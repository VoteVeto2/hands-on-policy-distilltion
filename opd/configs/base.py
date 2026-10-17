"""Defaults shared by all stages, and the hardware presets.

A stage config is `get_config()` plus a few stage-specific overrides, finished
by `apply_hardware`. Pick the preset after the colon and override any field on
the command line:

  python -m opd.main --config=opd/configs/stage1_distill.py:a100 \\
      --config.train.lr=3e-5 --workdir=runs/stage1

Stage values are sized for the 48 GB Apple M5 Pro ("mps"). Memory budget
(bf16 weights, 2 bytes per parameter): teacher Qwen3-8B ~16.4 GB + student
Qwen3-1.7B ~3.4 GB + LoRA weights and Adam state < 1 GB. The rest of the GPU
share of unified memory (macOS caps it at roughly 75% of RAM, ~36 GB, by
default) holds activations, KV caches and logits.
"""

from ml_collections import config_dict

HARDWARE = ("mps", "a100", "debug")


def get_config() -> config_dict.ConfigDict:
  c = config_dict.ConfigDict()
  c.seed = 0
  c.env = "gsm8k"  # A key of opd.envs.registry.SPECS.

  c.model = config_dict.ConfigDict()
  c.model.student = "Qwen/Qwen3-1.7B"
  c.model.teacher = "Qwen/Qwen3-8B"  # "" means no teacher.
  # Start from a previous stage's output, e.g. "runs/stage1/final".
  c.model.init_checkpoint = ""
  c.model.gradient_checkpointing = True
  # LoRA on every linear layer. Thinking Machines ("LoRA Without Regret",
  # 2025) found it matches full fine-tuning for RL and small post-training
  # runs, at ~10x the full-fine-tuning learning rate.
  c.model.lora = config_dict.ConfigDict()
  c.model.lora.enabled = True
  c.model.lora.rank = 32
  c.model.lora.alpha = 32

  c.rollout = config_dict.ConfigDict()
  # Qwen3's non-thinking mode keeps episodes short enough for a laptop.
  c.rollout.enable_thinking = False
  c.rollout.max_turns = 1
  c.rollout.max_new_tokens = 512  # Per student turn.
  # Whole episode after the prompt: student turns plus tool results.
  c.rollout.max_completion_tokens = 512
  c.rollout.batch_size = 16  # Sequences generated in parallel.

  # Per-token advantage (see losses.py):
  #   teacher_coef * (log q - log p) + reward_coef * group_advantage.
  c.loss = config_dict.ConfigDict()
  c.loss.teacher_coef = 1.0
  # "sampled": single-sample reverse KL folded into the advantage (Tinker).
  # "full": exact reverse KL over the vocabulary, added to the loss (GKD).
  c.loss.teacher_estimator = "sampled"
  c.loss.reward_coef = 0.0
  c.loss.normalize_std = True  # GRPO: divide by the group's reward std.
  c.loss.clip_low = 0.2
  c.loss.clip_high = 0.28  # DAPO's "clip-higher".
  c.loss.ref_kl_coef = 0.0  # KL penalty to the initial student.

  c.train = config_dict.ConfigDict()
  c.train.num_steps = 200
  c.train.prompts_per_step = 8
  c.train.samples_per_prompt = 1  # GRPO group size.
  c.train.micro_batch_size = 2  # Sequences per forward/backward pass.
  c.train.num_minibatches = 1  # Optimizer steps per rollout batch.
  c.train.lr = 1e-4
  c.train.warmup_steps = 5
  c.train.weight_decay = 0.0
  c.train.max_grad_norm = 1.0

  c.eval = config_dict.ConfigDict()
  c.eval.every = 25  # Steps; 0 disables evaluation.
  c.eval.num_tasks = 64
  c.eval.temperature = 0.0  # Greedy.
  c.eval.seed = 1234  # Fixed, so every run sees the same eval tasks.
  c.eval.teacher_baseline = True  # Evaluate the teacher once: the ceiling.

  c.log = config_dict.ConfigDict()
  c.log.samples_every = 10  # Steps; 0 logs no samples.
  c.log.save_every = 50  # Steps; 0 saves only the final checkpoint.
  c.log.wandb_project = ""
  return c


def apply_hardware(
    c: config_dict.ConfigDict, hardware: str
) -> config_dict.ConfigDict:
  """Adapts a Mac-sized stage config to the chosen hardware.

  Args:
    c: A stage config.
    hardware: "mps" (Apple M5 Pro 48 GB, as written), "a100" (80 GB: bigger
      batches and longer episodes, still LoRA) or "debug" (minutes-long smoke
      test with the smallest Qwen3 models; works on any machine).

  Returns:
    The same config, modified in place.
  """
  if hardware not in HARDWARE:
    raise ValueError(f"Unknown hardware {hardware!r}; choose from {HARDWARE}.")
  if hardware == "a100":
    c.train.prompts_per_step *= 4
    c.train.micro_batch_size = 8
    c.rollout.batch_size = 64
    c.rollout.max_new_tokens *= 2
    c.rollout.max_completion_tokens *= 2
    c.eval.num_tasks = 256
  elif hardware == "debug":
    c.model.student = "Qwen/Qwen3-0.6B"
    if c.model.teacher:
      c.model.teacher = "Qwen/Qwen3-1.7B"
    c.train.num_steps = 4
    c.train.prompts_per_step = 2
    c.rollout.max_new_tokens = min(c.rollout.max_new_tokens, 128)
    c.rollout.max_completion_tokens = min(c.rollout.max_completion_tokens, 256)
    c.eval.every = 2
    c.eval.num_tasks = 4
    c.log.samples_every = 1
    c.log.save_every = 4
  return c
