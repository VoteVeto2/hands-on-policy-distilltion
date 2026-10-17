"""Stage 1: plain on-policy distillation, Qwen3-8B -> Qwen3-1.7B, on GSM8K.

The student answers GSM8K questions; the teacher scores every token the
student wrote. No reward, no tools: advantage = log q(y_t) - log p(y_t).
Watch `teacher_kl` fall, and `eval/reward` (accuracy) rise without ever
training on the answers.
"""

from opd.configs import base


def get_config(hardware: str = "mps"):
  c = base.get_config()
  c.env = "gsm8k"
  c.loss.teacher_coef = 1.0
  c.loss.reward_coef = 0.0
  c.train.samples_per_prompt = 1
  c.train.prompts_per_step = 8
  c.train.lr = 1e-4  # Tinker's on-policy distillation recipe with LoRA.
  c.train.num_steps = 200
  return base.apply_hardware(c, hardware)
