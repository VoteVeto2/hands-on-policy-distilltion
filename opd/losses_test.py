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
    self.assertAlmostEqual(advantages[0].item(), 0.5 / (0.5**0.5 + 1e-4), places=5)
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

  def test_k3_is_unbiased(self):
    torch.manual_seed(0)
    p = torch.distributions.Categorical(logits=torch.randn(5))
    q = torch.distributions.Categorical(logits=torch.randn(5))
    y = p.sample((200_000,))
    estimate = losses.k3_kl(p.log_prob(y), q.log_prob(y)).mean()
    exact = torch.distributions.kl_divergence(p, q)
    self.assertAlmostEqual(estimate.item(), exact.item(), delta=0.01)


if __name__ == "__main__":
  absltest.main()
