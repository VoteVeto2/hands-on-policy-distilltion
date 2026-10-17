"""Stage 3: multi-tool, multi-turn trajectories; the student decides.

All three tools are available on every task. Some tasks chain two tools, some
need none. Watch the per-kind eval metrics: `eval/tool_calls/no_tool` should
stay near 0 while `eval/reward/sql+calculator` climbs.
"""

from opd.configs import base


def get_config(hardware: str = "mps"):
  c = base.get_config()
  c.env = "multi_tool"
  c.rollout.max_turns = 6
  c.rollout.max_new_tokens = 256
  c.rollout.max_completion_tokens = 1536
  c.loss.teacher_coef = 1.0
  c.loss.reward_coef = 0.0
  c.train.samples_per_prompt = 1
  c.train.prompts_per_step = 8
  c.train.lr = 1e-4
  c.train.num_steps = 200
  return base.apply_hardware(c, hardware)
