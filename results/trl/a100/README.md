# Qwen3-8B → Qwen3-1.7B on one A100-80GB (TRL backend)

`BACKEND=trl EXTRA=--use_vllm scripts/run_stages.sh a100` on a Colab
A100-SXM4-80GB (high-RAM), 2026-10-19. LoRA rank 32, each stage starting from
the previous stage's `final/` adapter. About 23 Colab compute units in total.

Each stage directory holds `trainer_state.json` (every logged metric),
`completions/*.parquet` (sampled episodes) and `final/` (the LoRA adapter, in
Git LFS; the stock Qwen3 `tokenizer.json` is left out).

## Results

Held-out evaluation, sampled at temperature 1.0 (TRL has no greedy eval).

| Stage | Time | eval `teacher_kl` | eval reward |
| ----- | ---- | ----------------- | ----------- |
| 1. GSM8K distillation (256 tasks) | 45 min | 0.229 → 0.136 | 0.809 → 0.836 |
| 2. Calculator tool | 18 min | 0.328 → 0.057 | 0.473 → 1.000 |
| 3. Multi-tool, multi-turn | 52 min | 0.379 → 0.068 | 0.660 → 0.746 |
| 4. Distillation + task reward | 55 min | 0.065 → 0.067 | 0.754 → 0.758 |
| 5. GRPO, no teacher | stopped at step ~20 of 600 | — | — |

- Stage 2: the student learns the teacher's habit of calling the calculator
  instead of doing large arithmetic in its head; episodes shrink from 232 to
  65 tokens.
- Stage 3: failed tool calls drop from 12% to 7% of calls; most of the gain
  is in the first 50 steps.

## GRPO had no signal after distillation

The distilled student is nearly deterministic (policy entropy ~0.01–0.03
nats/token at temperature 1.0), so the samples of a prompt usually all get the
same reward and the group-normalized advantage is zero:

- Stage 4: 98% of groups (4 samples) had zero reward variance, so the task
  reward added almost nothing to the teacher term; eval reward stayed flat.
- Stage 5: 100% of groups (8 samples) had zero reward variance after the
  first batch; `loss` and `grad_norm` were exactly 0. The run was stopped to
  save compute. `runs/trl/a100/stage5_partial.tgz` (local only) holds its
  samples.

Getting a learning signal from GRPO here would need more diverse rollouts
(e.g. a higher sampling temperature for the RL stages) or prompts the student
does not always solve or always fail.
