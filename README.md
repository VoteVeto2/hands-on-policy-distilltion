# Hands-on policy distillation

A from-scratch, readable post-training pipeline: on-policy distillation from
Qwen3-8B to Qwen3-1.7B, grown step by step into a tool-using agent trained
with GRPO. It runs on one Apple M5 Pro (48 GB) or one A100 (80 GB).

| Stage | What the student learns from |
| ----- | ---------------------------- |
| 1. On-policy distillation | The teacher's log-probs on every token the student writes (GSM8K prompts). |
| 2. Single-tool agent distillation | The same, in multi-turn episodes with a calculator, SQLite or JSON tool. |
| 3. Multi-tool, multi-turn | The same, with all tools offered; the student decides which to call. |
| 4. Distillation + reward | The teacher's per-token signal plus a task-success reward (GRPO groups). |
| 5. GRPO agent RL | The environment's reward alone; scale out with TRL + vLLM. |

All stages share one loop and one loss; only the config changes. Two trainers
run the same configs: `opd/trainer.py`, written from scratch to be read, and
TRL's `GRPOTrainer` with a teacher term added (`opd/trl_train.py`), for vLLM
and multiple GPUs.

- Setup and commands: [`ONBOARD.md`](ONBOARD.md)
- Step-by-step guide: open [`.onboard/guide.html`](.onboard/guide.html) in a browser
