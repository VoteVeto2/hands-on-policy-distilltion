"""Tests for loading the policy and the KL reference when chaining stages."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import peft
import torch

from opd import models
from opd import testing
from opd.configs import base


class ChainingTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_adapter_loads_on_the_weights_it_was_trained_on(self, lora):
    # Stage A ran full fine-tuning (its output: `full`, weights unlike the
    # student's); stage B trained a LoRA adapter on top. Stage C starts from
    # that adapter and must see stage B's policy, not adapter + student.
    root = self.enterContext(tempfile.TemporaryDirectory())
    student, _ = testing.save_tiny_checkpoints(root)
    full = os.path.join(root, "full")
    vocab_size = len(testing.make_tokenizer())
    testing.make_model(vocab_size, seed=1).save_pretrained(full)
    stage_b = peft.get_peft_model(
        models.load_frozen(full, torch.device("cpu")),
        peft.LoraConfig(
            r=4, target_modules="all-linear", init_lora_weights=False
        ),
    )
    adapter = os.path.join(root, "adapter")
    stage_b.save_pretrained(adapter)

    config = base.get_config().model
    config.student = student
    config.init_checkpoint = adapter
    config.lora.enabled = lora
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
      expected = stage_b.eval()(ids).logits
      policy = models.load_policy(config, torch.device("cpu")).eval()
      reference = models.load_reference(config, torch.device("cpu"))
      torch.testing.assert_close(policy(ids).logits, expected)
      torch.testing.assert_close(reference(ids).logits, expected)


if __name__ == "__main__":
  absltest.main()
