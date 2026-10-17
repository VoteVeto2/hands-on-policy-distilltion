"""Stage 4: teacher imitation + task-success reward.

Each prompt is sampled `samples_per_prompt` times. The advantage of every
token is the teacher term (dense, per token) plus the GRPO group advantage of
its episode's reward (sparse, per episode). When a whole group gets the same
reward (`zero_advantage_groups`), the teacher term still teaches.
"""

from opd.configs import base


def get_config(hardware: str = "mps"):
  c = base.get_config()
  c.env = "multi_tool"
  c.rollout.max_turns = 6
  c.rollout.max_new_tokens = 256
  c.rollout.max_completion_tokens = 1536
  c.loss.teacher_coef = 1.0
  c.loss.reward_coef = 1.0
  c.train.samples_per_prompt = 4  # Tinker's math RL recipe uses groups of 4.
  c.train.prompts_per_step = 4
  # Between the distillation (1e-4) and RL (1e-5) learning rates: the dense
  # teacher term dominates early, the reward term later.
  c.train.lr = 5e-5
  c.train.num_steps = 150
  return base.apply_hardware(c, hardware)
