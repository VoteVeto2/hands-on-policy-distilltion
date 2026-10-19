import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import pytest
import torch
from torch.nn import functional as F
import transformers

pytest.importorskip("trl")  # Optional: install it from the trl/ submodule.

# pylint: disable=g-import-not-at-top,g-bad-import-order
from opd import chat
from opd import testing
from opd import trl_train
from opd.configs import stage1_distill
from opd.configs import stage2_single_tool
from opd.configs import stage3_multi_tool
from opd.configs import stage4_distill_reward
from opd.configs import stage5_grpo
from opd.envs import calculator


class TrlTrainTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.root = tempfile.TemporaryDirectory()
    cls.student, cls.teacher = testing.save_tiny_checkpoints(cls.root.name)

  @classmethod
  def tearDownClass(cls):
    cls.root.cleanup()
    super().tearDownClass()

  def _tempdir(self):
    return self.enterContext(tempfile.TemporaryDirectory())

  def _tiny(self, config):
    """Points a stage config at the tiny checkpoints and shrinks everything."""
    config.model.student = self.student
    if config.model.teacher:
      config.model.teacher = self.teacher
    if config.env == "gsm8k":  # Needs the Hugging Face Hub.
      config.env = "calculator"
    config.model.gradient_checkpointing = False
    config.train.num_steps = 2
    config.train.prompts_per_step = 2
    config.train.micro_batch_size = 2
    config.rollout.max_completion_tokens = 32
    config.eval.every = 1
    config.eval.num_tasks = 2
    config.log.save_every = 2
    return config

  def test_reward_grades_final_message_of_finished_rollouts(self):
    stop, other = 1, 2
    called = {"role": "assistant", "content": "", "tool_calls": [{}]}
    answered = {"role": "assistant", "content": r"\boxed{6}"}
    completions = [
        [called, {"role": "tool", "content": "6"}, answered],
        [{"role": "assistant", "content": r"\boxed{5}"}],
        [{"role": "assistant", "content": None}],
        [answered],  # Ran out of tokens before the end of the turn.
        [called],  # Still calling tools at the turn limit.
    ]
    ids = [[other, stop], [stop], [stop], [other], [stop]]
    reward = trl_train.make_reward([stop])
    self.assertEqual(reward.__name__, "correctness_reward")
    rewards = reward(completions, completion_ids=ids, answer=["6"] * 5)
    self.assertEqual(rewards, [1.0, 0.0, 0.0, 0.0, 0.0])

  def test_advantage_is_minus_sampled_reverse_kl(self):
    config = self._tiny(stage2_single_tool.get_config("mps"))
    config.rollout.max_completion_tokens = 64  # Room for a tool round.
    config.eval.every = 0
    args = trl_train.make_config(config, self._tempdir(), use_vllm=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(self.student)
    dataset = trl_train.build_dataset(config, 4, "train", seed=0)
    opd_trainer = trl_train.OPDTrainer(
        model=self.student,
        reward_funcs=trl_train.make_reward(chat.stop_token_ids(tokenizer)),
        args=args,
        train_dataset=dataset,
        environment_factory=calculator.CalculatorEnv,
        teacher_model=transformers.AutoModelForCausalLM.from_pretrained(
            self.teacher
        ),
    )
    opd_trainer.model.train()
    output = opd_trainer._generate_and_score_completions(list(dataset))

    # Independently: full logits from fresh copies, on CPU.
    input_ids = torch.cat([output["prompt_ids"], output["completion_ids"]], 1)
    attention_mask = torch.cat(
        [output["prompt_mask"], output["completion_mask"]], 1
    )
    num_completion = output["completion_ids"].size(1)

    def logp(path):
      model = transformers.AutoModelForCausalLM.from_pretrained(path)
      with torch.no_grad():
        logits = model(
            input_ids=input_ids.cpu(), attention_mask=attention_mask.cpu()
        ).logits[:, -num_completion - 1 : -1]
      return (
          F.log_softmax(logits, -1)
          .gather(-1, output["completion_ids"].cpu()[..., None])
          .squeeze(-1)
      )

    expected = logp(self.teacher) - logp(self.student)  # reward_coef = 0.
    mask = output["completion_mask"].cpu().bool()
    advantages = output["advantages"].cpu()
    self.assertEqual(advantages.shape, mask.shape)
    # On GPU and MPS the trainer runs under bf16 autocast, which errs by
    # ~1e-3. A wrong sign, model or off-by-one alignment errs by ~0.2.
    torch.testing.assert_close(
        advantages[mask], expected[mask], atol=1e-2, rtol=0
    )

  def test_chained_stage_hands_on_the_policy_it_started_from(self):
    first = self._tiny(stage2_single_tool.get_config("mps"))
    first.train.lr = 1e-2  # Move the policy visibly.
    first.eval.every = 0
    first_dir = self._tempdir()
    trl_train.train(first, first_dir, use_vllm=False)
    second = self._tiny(stage3_multi_tool.get_config("mps"))
    second.model.init_checkpoint = os.path.join(first_dir, "final")
    second.train.lr = 0.0  # Saves exactly the policy it was handed.
    second.eval.every = 0
    second_dir = self._tempdir()
    trl_train.train(second, second_dir, use_vllm=False)

    def logits(checkpoint):
      config = self._tiny(stage3_multi_tool.get_config("mps"))
      config.model.init_checkpoint = checkpoint
      model = trl_train.load_model(config)
      if isinstance(model, str):
        model = transformers.AutoModelForCausalLM.from_pretrained(model)
      with torch.no_grad():
        return model(input_ids=torch.tensor([[5, 17, 42, 99, 7]])).logits

    started = logits(os.path.join(first_dir, "final"))
    self.assertFalse(torch.allclose(logits(""), started))
    torch.testing.assert_close(
        logits(os.path.join(second_dir, "final")), started
    )

  @parameterized.named_parameters(
      ("stage1", stage1_distill),
      ("stage2", stage2_single_tool),
      ("stage3", stage3_multi_tool),
      ("stage4", stage4_distill_reward),
      ("stage5", stage5_grpo),
  )
  def test_stage_runs(self, stage):
    config = self._tiny(stage.get_config("mps"))
    workdir = self._tempdir()
    trl_train.train(config, workdir, use_vllm=False)
    with open(os.path.join(workdir, "trainer_state.json")) as f:
      state = json.load(f)
    steps = config.train.num_steps * config.train.num_minibatches
    self.assertEqual(state["global_step"], steps)
    train_rows = [r for r in state["log_history"] if "loss" in r]
    self.assertEqual("teacher_kl" in train_rows[0], bool(config.model.teacher))
    eval_rows = [r for r in state["log_history"] if "eval_reward" in r]
    self.assertLen(eval_rows, config.train.num_steps + 1)  # + eval_on_start.
    self.assertTrue(
        os.path.exists(os.path.join(workdir, "final", "adapter_config.json"))
    )


if __name__ == "__main__":
  absltest.main()
