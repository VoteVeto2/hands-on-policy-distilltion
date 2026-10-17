"""The math of every stage: log-probs, advantages, and the policy loss.

All five stages minimise the same per-token loss. Only the advantage changes:

  A_t = teacher_coef * (log q(y_t)   - log p(y_t))    dense, from the teacher
      + ref_kl_coef  * (log ref(y_t) - log p(y_t))    dense, a leash (optional)
      + reward_coef  * A_task                         sparse, from the env

where p is the student, q the teacher, ref the policy the stage started from,
and y_t the token the student sampled. The first term is minus the
single-sample estimate of the per-token reverse KL(p || q): on-policy
distillation is policy-gradient RL with a dense reward that the teacher
computes for every token (Thinking Machines, "On-Policy Distillation", 2025;
their Tinker cookbook adds it to the advantages exactly like this). The second
term is the same thing with your former self as the teacher: a KL penalty to
the reference. A_task is GRPO's group-normalised outcome reward (Shao et al.,
2024, "DeepSeekMath").

Shapes: B = sequences, T = scored positions, V = vocabulary size.
"""

from typing import Callable

import torch
from torch.nn import functional as F
from torch.utils import checkpoint


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
  logp = _by_row(_row_logprobs, logits, targets)
  with torch.no_grad():
    entropy = _by_row(_row_entropy, logits)
  return logp, entropy


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
  return _by_row(
      _row_reverse_kl, student_logits[..., :vocab], teacher_logits[..., :vocab]
  )


def _by_row(
    fn: Callable[..., torch.Tensor], *tensors: torch.Tensor
) -> torch.Tensor:
  """Applies `fn` to one sequence at a time and stacks the results.

  A float32 [T, V] row is ~0.6 GB at T=1k with Qwen3's 152k vocabulary, and
  float32 matters: bf16 log-softmax over 152k entries loses precision. Looping
  over rows alone does not bound memory under autograd, because every row's
  float32 intermediates would be kept for the backward pass. Activation
  checkpointing keeps only the input row and recomputes the rest in backward.
  """
  rows = zip(*tensors)
  if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
    return torch.stack(
        [checkpoint.checkpoint(fn, *row, use_reentrant=False) for row in rows]
    )
  return torch.stack([fn(*row) for row in rows])


def _row_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  logits = logits.float()
  target_logits = logits.gather(-1, targets[:, None]).squeeze(-1)
  return target_logits - torch.logsumexp(logits, dim=-1)


def _row_entropy(logits: torch.Tensor) -> torch.Tensor:
  logits = logits.float()
  log_z = torch.logsumexp(logits, dim=-1)
  return log_z - (torch.softmax(logits, dim=-1) * logits).sum(-1)


def _row_reverse_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
  log_p = F.log_softmax(student_logits.float(), dim=-1)
  log_q = F.log_softmax(teacher_logits.float(), dim=-1)
  return (log_p.exp() * (log_p - log_q)).sum(-1)


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
