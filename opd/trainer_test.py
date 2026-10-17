"""End-to-end tests: every stage config trains on CPU with tiny models."""

import json
import os
import subprocess
import sys
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.nn import functional as F

from opd import losses
from opd import rollout
from opd import testing
from opd import trainer
from opd.configs import stage1_distill
from opd.configs import stage2_single_tool
from opd.configs import stage3_multi_tool
from opd.configs import stage4_distill_reward
from opd.configs import stage5_grpo


def _read_metrics(workdir):
  with open(os.path.join(workdir, "metrics.jsonl")) as f:
    return [json.loads(line) for line in f]


class TrainerTest(parameterized.TestCase):

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
    config.rollout.max_new_tokens = 8
    config.rollout.max_completion_tokens = 48
    config.rollout.batch_size = 4
    config.eval.every = 1
    config.eval.num_tasks = 2
    config.log.samples_every = 1
    config.log.save_every = 2
    return config

  @parameterized.named_parameters(
      ("stage1", stage1_distill),
      ("stage2", stage2_single_tool),
      ("stage3", stage3_multi_tool),
      ("stage4", stage4_distill_reward),
      ("stage5", stage5_grpo),
  )
  def test_stage_runs(self, stage):
    workdir = self._tempdir()
    config = self._tiny(stage.get_config("mps"))
    trainer.train(config, workdir)
    metrics = _read_metrics(workdir)
    train_rows = [m for m in metrics if "loss" in m]
    self.assertEqual([m["step"] for m in train_rows], [1, 2])
    self.assertEqual("teacher_kl" in train_rows[0], bool(config.model.teacher))
    self.assertTrue(any("eval/reward" in m for m in metrics))
    self.assertTrue(
        os.path.exists(os.path.join(workdir, "final", "adapter_config.json"))
    )

  def test_full_estimator_reference_kl_and_full_finetuning(self):
    workdir = self._tempdir()
    config = self._tiny(stage4_distill_reward.get_config("mps"))
    config.model.lora.enabled = False
    config.loss.teacher_estimator = "full"
    config.loss.ref_kl_coef = 0.1
    config.train.num_minibatches = 2
    trainer.train(config, workdir)
    last = [m for m in _read_metrics(workdir) if "loss" in m][-1]
    self.assertGreaterEqual(last["teacher_kl"], 0.0)  # Exact KL is >= 0.
    self.assertIn("ref_kl", last)
    self.assertTrue(
        os.path.exists(os.path.join(workdir, "final", "model.safetensors"))
    )

  @parameterized.parameters(True, False)
  def test_chained_stage_starts_at_zero_reference_kl(self, lora):
    # The KL penalty's reference must be where the stage starts (here: the
    # stage 3 adapter), not the original student. The large lr moves stage 3
    # far enough (ref_kl ~7e-3 against the original student) for the
    # assertion to tell the two apart.
    first = self._tempdir()
    config = self._tiny(stage3_multi_tool.get_config("mps"))
    config.train.lr = 1e-2
    config.train.warmup_steps = 0
    trainer.train(config, first)
    config = self._tiny(stage5_grpo.get_config("mps"))
    config.model.init_checkpoint = os.path.join(first, "final")
    config.model.lora.enabled = lora
    config.loss.ref_kl_coef = 0.1
    second = self._tempdir()
    trainer.train(config, second)
    step1 = [m for m in _read_metrics(second) if "loss" in m][0]
    self.assertAlmostEqual(step1["ref_kl"], 0.0, places=4)

  def test_fresh_adapter_reference_is_the_disabled_adapter(self):
    # Runs the `policy.disable_adapter()` branch of `score`.
    config = self._tiny(stage5_grpo.get_config("mps"))
    config.loss.ref_kl_coef = 0.1
    config.log.save_every = 0  # Only the final checkpoint.
    workdir = self._tempdir()
    trainer.train(config, workdir)
    step1 = [m for m in _read_metrics(workdir) if "loss" in m][0]
    self.assertAlmostEqual(step1["ref_kl"], 0.0, places=4)
    self.assertTrue(os.path.exists(os.path.join(workdir, "final")))

  def test_collate_and_forward_logits_align_with_sampled_tokens(self):
    tokenizer = testing.make_tokenizer()
    model = testing.make_model(len(tokenizer)).eval()
    episodes = [  # Different prompt lengths; tokens 10, 11 are a tool result.
        rollout.Episode(
            task={},
            group=0,
            prompt_ids=[5, 6, 7],
            completion_ids=[8, 9, 10, 11, 12, 13],
            action_mask=[1, 1, 0, 0, 1, 1],
        ),
        rollout.Episode(
            task={},
            group=0,
            prompt_ids=[5, 6, 7, 20, 21],
            completion_ids=[30, 31],
            action_mask=[1, 1],
        ),
    ]
    batch = trainer.collate(
        episodes, torch.zeros(2), tokenizer.pad_token_id, torch.device("cpu")
    )
    with torch.no_grad():
      logp, _ = losses.token_logprobs(
          trainer.forward_logits(model, batch), batch.targets
      )
    for i, episode in enumerate(episodes):
      # Reference: an unpadded forward pass; position t - 1 predicts token t.
      sequence = episode.prompt_ids + episode.completion_ids
      with torch.no_grad():
        full = F.log_softmax(model(torch.tensor([sequence])).logits[0], -1)
      expected_mask = torch.zeros(batch.action_mask.shape[1])
      for j, sampled in enumerate(episode.action_mask):
        t = len(episode.prompt_ids) + j
        expected_mask[t - batch.start] = sampled
        torch.testing.assert_close(
            logp[i, t - batch.start], full[t - 1, sequence[t]]
        )
      torch.testing.assert_close(batch.action_mask[i], expected_mask)

  def test_reference_kl_pulls_toward_the_reference(self):
    # Tokens the policy likes more than the reference does get a negative
    # advantage, so one update must make them less likely.
    tokenizer = testing.make_tokenizer()
    policy = testing.make_model(len(tokenizer))
    episode = rollout.Episode(
        task={},
        group=0,
        prompt_ids=[5, 6, 7],
        completion_ids=[8, 9, 10],
        action_mask=[1, 1, 1],
    )
    batch = trainer.collate(
        [episode], torch.zeros(1), tokenizer.pad_token_id, torch.device("cpu")
    )
    with torch.no_grad():
      before = trainer._logp(policy, batch)
    batch.ref_logp = before - 1.0
    config = stage5_grpo.get_config("mps")
    config.loss.ref_kl_coef = 1.0
    config.loss.reward_coef = 0.0
    config.train.num_minibatches = 1
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer.update(
        [batch], policy, None, optimizer, scheduler, config, torch.device("cpu")
    )
    with torch.no_grad():
      after = trainer._logp(policy, batch)
    self.assertLess(after.sum().item(), before.sum().item())

  @parameterized.parameters(
      # (estimator, prompts per step, tokens per turn, required KL ratio)
      ("full", 8, 8, 0.7),
      # Unbiased but noisy: needs more tokens per step to make progress.
      ("sampled", 32, 16, 0.9),
  )
  def test_distillation_reduces_teacher_kl(
      self, estimator, prompts, tokens, ratio
  ):
    # A teacher the student can actually reach in a few steps: the student
    # itself with its weights perturbed (initial per-token KL ~4 nats).
    tokenizer = testing.make_tokenizer()
    student = testing.make_model(len(tokenizer))
    teacher = testing.make_model(len(tokenizer))
    torch.manual_seed(1)
    with torch.no_grad():
      for param in teacher.parameters():
        param.add_(0.5 * torch.randn_like(param))
    root = self._tempdir()
    for name, model in (("student", student), ("teacher", teacher)):
      model.save_pretrained(os.path.join(root, name))
      tokenizer.save_pretrained(os.path.join(root, name))

    workdir = self._tempdir()
    config = self._tiny(stage1_distill.get_config("mps"))
    config.model.student = os.path.join(root, "student")
    config.model.teacher = os.path.join(root, "teacher")
    config.model.lora.enabled = False
    config.loss.teacher_estimator = estimator
    config.train.lr = 3e-3
    config.train.num_steps = 30
    config.train.prompts_per_step = prompts
    config.train.micro_batch_size = 16
    config.rollout.max_new_tokens = tokens
    config.rollout.max_completion_tokens = 64
    config.eval.every = 0
    config.log.samples_every = 0
    config.log.save_every = 1000
    trainer.train(config, workdir)
    kls = [m["teacher_kl"] for m in _read_metrics(workdir)]
    first, last = sum(kls[:5]) / 5, sum(kls[-5:]) / 5
    self.assertLess(last, ratio * first)

  def test_invalid_config_fails_loudly(self):
    config = self._tiny(stage4_distill_reward.get_config("mps"))
    config.train.samples_per_prompt = 1
    with self.assertRaisesRegex(ValueError, "samples_per_prompt"):
      trainer.train(config, self._tempdir())
    config = self._tiny(stage5_grpo.get_config("mps"))
    config.train.micro_batch_size = 64  # One micro-batch, two minibatches.
    with self.assertRaisesRegex(ValueError, "micro-batch"):
      trainer.train(config, self._tempdir())

  def test_command_line(self):
    workdir = self._tempdir()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        [
            sys.executable, "-m", "opd.main",
            "--config=opd/configs/stage3_multi_tool.py:debug",
            f"--config.model.student={self.student}",
            f"--config.model.teacher={self.teacher}",
            "--config.train.num_steps=1",
            "--config.rollout.max_new_tokens=8",
            "--config.eval.num_tasks=1",
            f"--workdir={workdir}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )  # fmt: skip
    self.assertLen([m for m in _read_metrics(workdir) if "loss" in m], 1)


if __name__ == "__main__":
  absltest.main()
