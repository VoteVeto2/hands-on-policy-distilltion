"""Stage 5: GRPO agent RL from real environment feedback. No teacher.

The only learning signal is whether the final answer is right, and the only
information about *how* comes from the environment: SQL errors, empty JMESPath
results, calculator outputs. Without the teacher's 16 GB, groups get bigger.

Each rollout batch feeds two optimizer steps, so the second step trains on
slightly stale samples: that is where the PPO clip and `clip_frac` come alive.

To scale beyond one GPU, run the same environment with TRL's GRPOTrainer and
vLLM: `python -m opd.trl_train --config=opd/configs/stage5_grpo.py:a100`.
"""

from opd.configs import base


def get_config(hardware: str = "mps"):
  c = base.get_config()
  c.env = "multi_tool"
  c.model.teacher = ""
  c.rollout.max_turns = 6
  c.rollout.max_new_tokens = 256
  c.rollout.max_completion_tokens = 1536
  c.loss.teacher_coef = 0.0
  c.loss.reward_coef = 1.0
  c.loss.ref_kl_coef = 0.0  # Try 1e-3 (DeepSeek-R1's value) to stay close.
  c.train.samples_per_prompt = 8
  c.train.prompts_per_step = 4
  c.train.num_minibatches = 2
  c.train.lr = 1e-5  # Tinker's math RL recipe with LoRA rank 32.
  c.train.num_steps = 300
  c.eval.teacher_baseline = False
  return base.apply_hardware(c, hardware)
