"""The math of every stage: log-probs, advantages, and the policy loss.

All five stages minimise the same per-token loss. Only the advantage changes:

  A_t = teacher_coef * (log q(y_t) - log p(y_t))      dense, from the teacher
      + reward_coef  * A_task                         sparse, from the env

where p is the student, q the teacher and y_t the token the student sampled.
The first term is minus the single-sample estimate of the per-token reverse
KL(p || q): on-policy distillation is policy-gradient RL with a dense reward
that the teacher computes for every token (Thinking Machines, "On-Policy
Distillation", 2025; their Tinker cookbook adds it to the advantages exactly
like this). A_task is GRPO's group-normalised outcome reward (Shao et al.,
2024, "DeepSeekMath").

Shapes: B = sequences, T = scored positions, V = vocabulary size.
"""

import torch
from torch.nn import functional as F


def token_logprobs(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Log-probability of each target token, and the entropy at each position.

  Args:
    logits: [B, T, V] next-token logits, any float dtype.
    targets: [B, T] token ids the logits should predict.

  Returns:
    logp: [B, T] float32, differentiable.
    entropy: [B, T] float32, detached (a diagnostic only).
  """
  # One row at a time: a float32 copy of [T, V] logits is ~0.6 GB at T=1k with
  # Qwen3's 152k vocabulary, so a whole batch at once can run out of memory.
  # Float32 also matters: bf16 log-softmax over 152k entries loses precision.
  logps, entropies = [], []
  for row_logits, row_targets in zip(logits, targets):
    row_logits = row_logits.float()
    log_z = torch.logsumexp(row_logits, dim=-1)
    target_logits = row_logits.gather(-1, row_targets[:, None]).squeeze(-1)
    logps.append(target_logits - log_z)
    with torch.no_grad():
      probs = torch.softmax(row_logits, dim=-1)
      entropies.append(log_z - (probs * row_logits).sum(-1))
  return torch.stack(logps), torch.stack(entropies)


def reverse_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
  """Exact per-position KL(student || teacher), summed over the vocabulary.

  The zero-variance alternative to the single-sample estimate used in the
  advantage. It costs a teacher forward pass inside the update and the full
  [T, V] distributions of both models.

  Args:
    student_logits: [B, T, V_s], differentiable.
    teacher_logits: [B, T, V_t].

  Returns:
    [B, T] float32.
  """
  # Checkpoints of one family can pad the vocabulary to different sizes
  # (e.g. Qwen2.5-7B has 152064 rows, Qwen2.5-0.5B 151936). The extra rows are
  # never-used padding, so compare the shared prefix.
  vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
  kls = []
  for student_row, teacher_row in zip(student_logits, teacher_logits):
    log_p = F.log_softmax(student_row[:, :vocab].float(), dim=-1)
    log_q = F.log_softmax(teacher_row[:, :vocab].float(), dim=-1)
    kls.append((log_p.exp() * (log_p - log_q)).sum(-1))
  return torch.stack(kls)


def group_advantages(
    rewards: torch.Tensor, groups: torch.Tensor, normalize_std: bool
) -> torch.Tensor:
  """GRPO advantages: each reward minus the mean of its group.

  A group is all samples for the same prompt, so the group mean is a baseline
  that needs no value network.

  Args:
    rewards: [N] one scalar per episode.
    groups: [N] group id of each episode.
    normalize_std: Also divide by the group's standard deviation (GRPO). Dr.
      GRPO (Liu et al., 2025) argues against it: it up-weights prompts that
      are nearly always solved or nearly always failed.

  Returns:
    [N] advantages. Zero for groups whose rewards are all equal: those
    prompts carry no learning signal from the reward.
  """
  advantages = torch.zeros_like(rewards)
  for group in groups.unique():
    members = groups == group
    centered = rewards[members] - rewards[members].mean()
    if normalize_std and members.sum() > 1:
      centered = centered / (rewards[members].std() + 1e-4)
    advantages[members] = centered
  return advantages


def clipped_policy_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float,
    clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """PPO's clipped surrogate, per token, with DAPO's asymmetric clip range.

  With one optimizer step per batch, logp == old_logp, the ratio is 1, the
  clip never binds, and the gradient is exactly REINFORCE: -A * grad(logp).
  Clipping matters once a batch is split into several optimizer steps: later
  steps see data sampled by an older policy.

  Args:
    logp: [B, T] current log-probs (differentiable).
    old_logp: [B, T] log-probs of the policy that sampled the tokens.
    advantages: [B, T] per-token advantages.
    clip_low: The ratio is clipped below at 1 - clip_low.
    clip_high: ... and above at 1 + clip_high (DAPO: 0.28 > 0.2 keeps
      low-probability tokens explorable).

  Returns:
    loss: [B, T] per-token loss.
    clipped: [B, T] 1.0 where the clipped branch was selected.
  """
  ratio = torch.exp(logp - old_logp)
  unclipped = ratio * advantages
  clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * advantages
  return -torch.minimum(unclipped, clipped), (clipped < unclipped).float()


def k3_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
  """Schulman's k3 estimator of KL(policy || reference): unbiased and >= 0."""
  log_ratio = ref_logp - logp
  return torch.exp(log_ratio) - log_ratio - 1
