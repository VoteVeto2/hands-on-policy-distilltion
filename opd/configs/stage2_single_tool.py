"""Stage 2: single-tool agent distillation.

One tool per run: `calculator` (default), `sqlite` or `json`:

  --config=opd/configs/stage2_single_tool.py --config.env=sqlite

Same loss as stage 1, but episodes are now multi-turn: tool results enter the
context with action_mask = 0. The teacher scores the student's tool calls
(name, JSON arguments) and its use of the results, token by token.
"""

from opd.configs import base


def get_config(hardware: str = "mps"):
  c = base.get_config()
  c.env = "calculator"
  c.rollout.max_turns = 4
  c.rollout.max_new_tokens = 256
  c.rollout.max_completion_tokens = 1024
  c.loss.teacher_coef = 1.0
  c.loss.reward_coef = 0.0
  c.train.samples_per_prompt = 1
  c.train.prompts_per_step = 8
  c.train.lr = 1e-4
  c.train.num_steps = 150
  return base.apply_hardware(c, hardware)
