"""Loading the student (trainable) and the frozen teacher / reference models.

Precision rules (the memory budget behind them is in the guide):
  * LoRA: the student's base weights are frozen bf16; PEFT keeps the small
    adapter matrices in fp32. This fits a 48 GB Mac.
  * Full fine-tuning: fp32 master weights, bf16 autocast for the math (see
    `rollout.autocast`). Pure-bf16 weights would silently drop updates: bf16
    keeps ~3 significant digits, so w + 1e-6 * g rounds back to w.
  * Frozen models (teacher, reference): bf16.
  * CPU (unit tests): everything fp32.
"""

import os

from absl import logging
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


def load_policy(config, device: torch.device) -> transformers.PreTrainedModel:
  """Loads the trainable student.

  Args:
    config: The `model` section of the experiment config.
    device: Where to put the model.

  Returns:
    A PEFT model when `config.lora.enabled`, else the full model.
  """
  checkpoint = config.init_checkpoint
  is_adapter = bool(checkpoint) and os.path.exists(
      os.path.join(checkpoint, "adapter_config.json")
  )
  if device.type == "cpu" or not config.lora.enabled:
    dtype = torch.float32
  else:
    dtype = torch.bfloat16
  base_name = config.student if is_adapter or not checkpoint else checkpoint
  model = transformers.AutoModelForCausalLM.from_pretrained(
      base_name, dtype=dtype
  )
  model.to(device)
  if config.gradient_checkpointing:
    # Recompute activations in the backward pass: ~30% slower, far less memory.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
  if config.lora.enabled:
    if is_adapter:
      model = peft.PeftModel.from_pretrained(
          model, checkpoint, is_trainable=True
      )
    else:
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
  """Loads a model for inference only (teacher or reference)."""
  dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
  model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
  model.to(device).eval().requires_grad_(False)
  return model
