from absl.testing import absltest
import torch
from torch.nn import functional as F

from opd import losses


class LossesTest(absltest.TestCase):

  def test_token_logprobs_matches_log_softmax(self):
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    logp, entropy = losses.token_logprobs(logits, targets)
    log_probs = F.log_softmax(logits, -1)
    expected = log_probs.gather(-1, targets[..., None]).squeeze(-1)
    torch.testing.assert_close(logp, expected)
    torch.testing.assert_close(entropy, -(log_probs.exp() * log_probs).sum(-1))

  def test_reverse_kl(self):
    p_logits, q_logits = torch.randn(1, 3, 7), torch.randn(1, 3, 9)
    kl = losses.reverse_kl(p_logits, q_logits)
    log_p = F.log_softmax(p_logits, -1)
    log_q = F.log_softmax(q_logits[..., :7], -1)
    torch.testing.assert_close(kl, (log_p.exp() * (log_p - log_q)).sum(-1))
    torch.testing.assert_close(
        losses.reverse_kl(p_logits, p_logits), torch.zeros(1, 3)
    )

  def test_distillation_advantage_gives_reverse_kl_gradient(self):
    # The identity behind on-policy distillation: with advantage
    # A(y) = log q(y) - log p(y) on sampled tokens y ~ p, the policy-gradient
    # loss has, in expectation, exactly the gradient of KL(p || q).
    theta = torch.randn(6, requires_grad=True)
    log_q = F.log_softmax(torch.randn(6), -1)

    log_p = F.log_softmax(theta, -1)
    kl = (log_p.exp() * (log_p - log_q)).sum()
    (expected,) = torch.autograd.grad(kl, theta)

    log_p = F.log_softmax(theta, -1)
    advantages = (log_q - log_p).detach()
    per_token, _ = losses.clipped_policy_loss(
        log_p, log_p.detach(), advantages, 0.2, 0.28
    )
    # Exact expectation over y ~ p instead of sampling.
    expected_loss = (log_p.exp().detach() * per_token).sum()
    (actual,) = torch.autograd.grad(expected_loss, theta)
    torch.testing.assert_close(actual, expected)

  def test_group_advantages(self):
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.5])
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    advantages = losses.group_advantages(rewards, groups, normalize_std=False)
    torch.testing.assert_close(
        advantages, torch.tensor([0.5, -0.5, 0.0, 0.0, -0.25, 0.25])
    )
    advantages = losses.group_advantages(rewards, groups, normalize_std=True)
    self.assertAlmostEqual(
        advantages[0].item(), 0.5 / (0.5**0.5 + 1e-4), places=5
    )
    self.assertEqual(advantages[2].item(), 0.0)

  def test_clipping(self):
    logp = torch.log(torch.tensor([1.5, 0.5, 1.5]))
    loss, clipped = losses.clipped_policy_loss(
        logp, torch.zeros(3), torch.tensor([1.0, -1.0, -1.0]), 0.2, 0.28
    )
    # Positive advantage: ratio capped at 1.28. Negative: floored at 0.8.
    # A ratio that moved against the advantage is never clipped.
    torch.testing.assert_close(loss, torch.tensor([-1.28, 0.8, 1.5]))
    torch.testing.assert_close(clipped, torch.tensor([1.0, 1.0, 0.0]))

  def test_k3_in_the_loss_follows_the_forward_kl(self):
    # Why the KL penalty lives in the advantage. GRPO adds Schulman's k3
    # estimate of KL(p || ref) to the loss. Its value is right, but its
    # expected gradient is that of KL(ref || p), the mass-covering direction.
    theta = torch.randn(6, requires_grad=True)
    log_ref = F.log_softmax(torch.randn(6), -1)

    def grad(fn):
      log_p = F.log_softmax(theta, -1)
      return torch.autograd.grad(fn(log_p), theta)[0]

    def k3_loss(log_p):
      log_ratio = log_ref - log_p
      k3 = torch.exp(log_ratio) - log_ratio - 1
      return (log_p.exp().detach() * k3).sum()  # Exact expectation, y ~ p.

    forward_kl = grad(lambda log_p: (log_ref.exp() * (log_ref - log_p)).sum())
    reverse_kl = grad(lambda log_p: (log_p.exp() * (log_p - log_ref)).sum())
    torch.testing.assert_close(grad(k3_loss), forward_kl)
    self.assertFalse(torch.allclose(grad(k3_loss), reverse_kl))

  def test_rows_are_recomputed_not_stored(self):
    # Under autograd, only the bf16 input logits may be kept for backward:
    # no float32 [T, V] intermediates.
    logits = torch.randn(4, 16, 1000, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, 1000, (4, 16))
    saved = []

    def pack(tensor):
      saved.append(tensor)
      return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
      logp, _ = losses.token_logprobs(logits, targets)
      kl = losses.reverse_kl(logits, logits.detach().flip(0))
    self.assertEmpty(
        [t for t in saved if t.dtype == torch.float32 and t.numel() >= 16000]
    )
    (logp.sum() + kl.sum()).backward()

    # Same gradients as the plain float32 computation.
    expected = logits.detach().float().requires_grad_()
    log_p = F.log_softmax(expected, -1)
    log_q = F.log_softmax(logits.detach().flip(0).float(), -1)
    plain = log_p.gather(-1, targets[..., None]).sum()
    plain = plain + (log_p.exp() * (log_p - log_q)).sum()
    plain.backward()
    torch.testing.assert_close(
        logits.grad.float(), expected.grad, atol=1e-2, rtol=1e-2
    )


if __name__ == "__main__":
  absltest.main()
