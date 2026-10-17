"""Loading the student (trainable) and the frozen teacher / reference models.

Precision rules (the memory budget behind them is in the guide):
  * LoRA: the student's base weights are frozen bf16; PEFT keeps the small
    adapter matrices in fp32. This fits a 48 GB Mac.
  * Full fine-tuning: fp32 master weights, bf16 autocast for the math (see
    `rollout.autocast`). Pure-bf16 weights would silently drop updates: bf16
    keeps ~3 significant digits, so w + 1e-6 * g rounds back to w.
  * Frozen models (teacher, KL reference): bf16.
  * CPU (unit tests): everything fp32.
"""

import os

from absl import logging
from ml_collections import config_dict
import peft
import torch
import transformers


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def load_tokenizer(name: str) -> transformers.PreTrainedTokenizerBase:
  return transformers.AutoTokenizer.from_pretrained(name)


def check_same_tokenizer(
    student: transformers.PreTrainedTokenizerBase,
    teacher: transformers.PreTrainedTokenizerBase,
) -> None:
  """Token-level distillation compares log-probs of the *same* token ids."""
  if student.get_vocab() != teacher.get_vocab():
    raise ValueError(
        "Student and teacher tokenizers differ; token-level distillation needs"
        " a shared vocabulary (e.g. two models of the same family)."
    )


def is_adapter(path: str) -> bool:
  """True if `path` holds a LoRA adapter (an earlier LoRA stage's output)."""
  return bool(path) and os.path.exists(
      os.path.join(path, "adapter_config.json")
  )


def base_weights(config: config_dict.ConfigDict) -> str:
  """The full weights a stage starts from, before any LoRA adapter.

  An adapter records the weights it was trained on. They are not
  `config.student` when an earlier stage ran full fine-tuning.

  Args:
    config: The `model` section of the experiment config.
  """
  checkpoint = config.init_checkpoint
  if is_adapter(checkpoint):
    return peft.PeftConfig.from_pretrained(checkpoint).base_model_name_or_path
  return checkpoint or config.student


def load_policy(
    config: config_dict.ConfigDict, device: torch.device
) -> transformers.PreTrainedModel:
  """Loads the trainable student.

  `config.init_checkpoint` chains stages. A full checkpoint replaces the
  student's weights. A LoRA adapter is trained further when LoRA is on, and
  folded into the fp32 weights when it is off (full fine-tuning).

  Args:
    config: The `model` section of the experiment config.
    device: Where to put the model.

  Returns:
    A PEFT model when `config.lora.enabled`, else the full model.
  """
  checkpoint = config.init_checkpoint
  if device.type == "cpu" or not config.lora.enabled:
    dtype = torch.float32
  else:
    dtype = torch.bfloat16
  from_adapter = is_adapter(checkpoint)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      base_weights(config), dtype=dtype
  )
  if from_adapter and not config.lora.enabled:
    model = peft.PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    model.requires_grad_(True)  # PEFT froze the base weights when loading.
  model.to(device)
  if config.gradient_checkpointing:
    # Recompute activations in the backward pass: ~30% slower, far less memory.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
  if config.lora.enabled and from_adapter:
    model = peft.PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
  elif config.lora.enabled:
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules="all-linear",  # Attention *and* MLP layers.
            lora_dropout=0.0,
        ),
    )
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in model.parameters())
  logging.info(
      "Student %s: %.1fM / %.1fM params trainable, %s on %s.",
      checkpoint or config.student, trainable / 1e6, total / 1e6, dtype, device,
  )  # fmt: skip
  return model


def load_frozen(
    name: str, device: torch.device
) -> transformers.PreTrainedModel:
  """Loads a model for inference only (the teacher)."""
  dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
  model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
  model.to(device).eval().requires_grad_(False)
  return model


def load_reference(
    config: config_dict.ConfigDict, device: torch.device
) -> transformers.PreTrainedModel:
  """A frozen copy of the policy this stage starts from, for the KL penalty."""
  checkpoint = config.init_checkpoint
  model = load_frozen(base_weights(config), device)
  if not is_adapter(checkpoint):
    return model
  # Kept unmerged: folding a small LoRA update into bf16 weights rounds part of
  # it away.
  return peft.PeftModel.from_pretrained(model, checkpoint).eval()
