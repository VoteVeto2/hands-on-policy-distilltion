r"""Stage 5 at scale: the same environments, trained with TRL's GRPOTrainer.

`opd/trainer.py` is written to be read. To scale past one device, hand the
same environment classes and the same reward to TRL, which adds vLLM
generation, multi-GPU training (accelerate / DeepSpeed / FSDP) and many GRPO
variants. Nothing about the environments changes: they already follow TRL's
`environment_factory` convention.

Usage (one A100, vLLM sharing the GPU with training):
  python -m opd.trl_grpo --config=opd/configs/stage5_grpo.py:a100 \
      --workdir=runs/stage5_trl --use_vllm

Several GPUs:
  accelerate launch -m opd.trl_grpo --config=... --workdir=... --use_vllm

Mapping to the from-scratch trainer: `samples_per_prompt` -> num_generations,
`num_minibatches` -> optimizer steps per generation batch, `clip_low/high` ->
epsilon/epsilon_high, `ref_kl_coef` -> beta, token-mean loss -> loss_type
"dapo", `normalize_std` -> scale_rewards.
"""

import os
import random

from absl import app
from absl import flags
import datasets
from ml_collections import config_flags
import peft
import transformers
import trl

from opd import chat
from opd import models
from opd.envs import base
from opd.envs import registry

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Stage config file, optionally ':<hardware>'."
)
_WORKDIR = flags.DEFINE_string("workdir", None, "Output directory.")
_USE_VLLM = flags.DEFINE_bool(
    "use_vllm",
    False,
    "Generate with vLLM (CUDA only, `pip install trl[vllm]`).",
)
flags.mark_flags_as_required(["config", "workdir"])


def correctness_reward(completions, answer, **kwargs) -> list[float]:
  """TRL reward function: grades the final assistant message of each rollout.

  Args:
    completions: One message list per rollout (assistant turns with tool
      calls, tool results, and the final assistant turn).
    answer: The dataset's `answer` column, one entry per rollout.
    **kwargs: Other dataset columns and trainer state, unused.

  Returns:
    1.0 or 0.0 per rollout, as in `opd.envs.base.grade`.
  """
  del kwargs
  rewards = []
  for messages, gold in zip(completions, answer):
    replies = [m for m in messages if m.get("role") == "assistant"]
    final = (replies[-1].get("content") or "") if replies else ""
    rewards.append(base.grade(final, gold))
  return rewards


def build_dataset(config, num_rows: int) -> datasets.Dataset:
  """Tasks as rows: a `prompt` column plus the task fields `reset` reads."""
  spec = registry.get(config.env)
  rng = random.Random(config.seed)
  rows = []
  for _ in range(num_rows):
    task = spec.sample_task(rng, "train")
    prompt = [
        {"role": "system", "content": chat.SYSTEM_PROMPT},
        {"role": "user", "content": task["question"]},
    ]
    rows.append({"prompt": prompt, **task})
  return datasets.Dataset.from_list(rows)


def load_model(config) -> str | transformers.PreTrainedModel:
  """The student, with a previous stage's LoRA adapter merged in if given."""
  checkpoint = config.model.init_checkpoint
  if not checkpoint:
    return config.model.student
  if not os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
    return checkpoint  # A full fine-tuned model.
  model = transformers.AutoModelForCausalLM.from_pretrained(
      config.model.student, dtype="auto"
  )
  return peft.PeftModel.from_pretrained(model, checkpoint).merge_and_unload()


def make_grpo_config(config, workdir: str, use_vllm: bool) -> trl.GRPOConfig:
  episodes = config.train.prompts_per_step * config.train.samples_per_prompt
  per_step = config.train.micro_batch_size * config.train.num_minibatches
  if episodes % per_step:
    raise ValueError(
        "prompts_per_step * samples_per_prompt must be divisible by"
        " micro_batch_size * num_minibatches."
    )
  accumulation = episodes // per_step
  return trl.GRPOConfig(
      output_dir=workdir,
      seed=config.seed,
      max_steps=config.train.num_steps * config.train.num_minibatches,
      learning_rate=config.train.lr,
      lr_scheduler_type="constant_with_warmup",
      warmup_steps=config.train.warmup_steps,
      weight_decay=config.train.weight_decay,
      max_grad_norm=config.train.max_grad_norm,
      per_device_train_batch_size=config.train.micro_batch_size,
      gradient_accumulation_steps=accumulation,
      # One generation batch of `episodes` feeds `num_minibatches` steps.
      steps_per_generation=accumulation * config.train.num_minibatches,
      num_generations=config.train.samples_per_prompt,
      max_completion_length=config.rollout.max_completion_tokens,
      max_tool_calling_iterations=config.rollout.max_turns - 1,
      chat_template_kwargs={"enable_thinking": config.rollout.enable_thinking},
      beta=config.loss.ref_kl_coef,
      epsilon=config.loss.clip_low,
      epsilon_high=config.loss.clip_high,
      loss_type="dapo",
      scale_rewards="group" if config.loss.normalize_std else "none",
      gradient_checkpointing=config.model.gradient_checkpointing,
      bf16=models.get_device().type != "cpu",
      use_vllm=use_vllm,
      vllm_mode="colocate",
      logging_steps=1,
      save_steps=config.log.save_every,
      report_to="wandb" if config.log.wandb_project else "none",
  )


def train(config, workdir: str, use_vllm: bool) -> None:
  """Runs GRPO with TRL on the config's environment."""
  if config.loss.teacher_coef > 0:
    raise ValueError(
        "This bridge runs GRPO only; use a config without teacher."
    )
  if config.log.wandb_project:
    os.environ.setdefault("WANDB_PROJECT", config.log.wandb_project)
  peft_config = None
  if config.model.lora.enabled:
    peft_config = peft.LoraConfig(
        r=config.model.lora.rank,
        lora_alpha=config.model.lora.alpha,
        target_modules="all-linear",
    )
  num_rows = config.train.num_steps * config.train.prompts_per_step
  trainer = trl.GRPOTrainer(
      model=load_model(config),
      reward_funcs=correctness_reward,
      args=make_grpo_config(config, workdir, use_vllm),
      train_dataset=build_dataset(config, num_rows),
      environment_factory=registry.get(config.env).env_cls,
      peft_config=peft_config,
  )
  trainer.train()
  trainer.save_model(os.path.join(workdir, "final"))


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  train(_CONFIG.value, _WORKDIR.value, _USE_VLLM.value)


if __name__ == "__main__":
  app.run(main)
