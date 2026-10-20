# Qwen3.5-9B → Qwen3.5-2B on one A100-80GB (TRL backend)

`BACKEND=trl EXTRA=--use_vllm scripts/run_stages.sh a100 1 4` on a Colab
A100-SXM4-80GB (high-RAM), 2026-10-20, with the same configs as the Qwen3
run in `../a100/`. LoRA rank 32, each stage starting from the previous
stage's `final/` adapter, non-thinking mode throughout. About 50 Colab
compute units in total (the Qwen3 run: 23).

Each stage directory holds `trainer_state.json` (every logged metric),
`completions/*.parquet` (sampled episodes) and `final/` (the LoRA adapter, in
Git LFS; the stock Qwen3.5 `tokenizer.json` is left out).

## Results

Held-out evaluation, sampled at temperature 1.0 (TRL has no greedy eval).
In brackets: the Qwen3-8B → Qwen3-1.7B run.

| Stage | Time | eval `teacher_kl` | eval reward |
| ----- | ---- | ----------------- | ----------- |
| 1. GSM8K distillation (256 tasks) | 85 min (45) | 0.218 → 0.168 (0.229 → 0.136) | 0.656 → 0.781 (0.809 → 0.836) |
| 2. Calculator tool | 103 min (18) | 0.108 → 0.047 (0.328 → 0.057) | 0.645 → 0.902 (0.473 → 1.000) |
| 3. Multi-tool, multi-turn | 109 min (52) | 0.163 → 0.034 (0.378 → 0.068) | 0.754 → 0.957 (0.660 → 0.746) |
| 4. Distillation + task reward | 130 min (55) | 0.034 → 0.025 (0.065 → 0.067) | 0.945 → 0.953 (0.754 → 0.758) |

Stage 5 (GRPO without the teacher) was not rerun.

- Stage 1: 10–18% of answers run past the 1024-token limit and score 0.
  They are not loops: after an awkward intermediate result (say, a
  non-integer count) the model second-guesses itself in the answer. At step
  200, 91% of the answers that finish are right (Qwen3-1.7B: 84%).
- Stage 2: the student takes up the calculator (0.12 → 1.55 calls per
  episode) and episodes shrink from 1011 to 371 tokens (Qwen3: 232 → 65).
  Of the 129 failed training episodes in the last 50 steps, 89 used up the
  4 turns re-checking a result with more calls before giving an answer.
- Stage 3: failed tool calls drop from 21% to 4% of calls (Qwen3: 14% → 7%).
- Time: a token costs the same in both runs; Qwen3.5's episodes are 2–5x
  longer, with step-by-step explanations around every tool call.

## More GRPO signal, still too little

The distilled Qwen3.5 student keeps more randomness than the Qwen3 one
(policy entropy ~0.14 nats/token at temperature 1.0, against 0.01–0.03), so
more groups of 4 samples disagree on the reward: 11% in stage 4, against
1.5%. But it already solves ~95% of the stage 3/4 tasks, so 89% of groups
still have zero reward variance, and the stage 4 change in eval reward
(0.945 → 0.953) is within sampling noise: the same weights scored 0.957 at
the end of stage 3. Getting a real signal from the task reward needs harder
tasks than these.
