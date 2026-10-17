import json
import os
import tempfile

from absl.testing import absltest

from opd import testing
from opd import trl_grpo
from opd.configs import stage5_grpo


class TrlGrpoTest(absltest.TestCase):

  def test_reward_reads_final_assistant_message(self):
    completions = [
        [
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "content": "6"},
            {"role": "assistant", "content": r"\boxed{6}"},
        ],
        [{"role": "assistant", "content": r"\boxed{5}"}],
        [{"role": "assistant", "content": None}],
    ]
    rewards = trl_grpo.correctness_reward(completions, answer=["6", "6", "6"])
    self.assertEqual(rewards, [1.0, 0.0, 0.0])

  def test_trains_with_trl(self):
    root = self.enterContext(tempfile.TemporaryDirectory())
    student, _ = testing.save_tiny_checkpoints(root)
    config = stage5_grpo.get_config("mps")
    config.model.student = student
    config.model.gradient_checkpointing = False
    config.train.num_steps = 2
    config.train.prompts_per_step = 2
    config.train.samples_per_prompt = 2
    config.train.micro_batch_size = 2
    config.rollout.max_completion_tokens = 32
    workdir = os.path.join(root, "run")
    trl_grpo.train(config, workdir, use_vllm=False)
    with open(os.path.join(workdir, "checkpoint-4", "trainer_state.json")) as f:
      history = json.load(f)["log_history"]
    self.assertEqual(history[-1]["step"], 4)  # 2 rollout batches x 2 steps.
    self.assertTrue(
        os.path.exists(os.path.join(workdir, "final", "adapter_config.json"))
    )


if __name__ == "__main__":
  absltest.main()
