# Onboarding

A hands-on post-training pipeline: on-policy distillation from Qwen3-8B to
Qwen3-1.7B, growing into tool-using agents and GRPO. Open
`.onboard/guide.html` in a browser for the step-by-step walkthrough; this file
is the short version.

## Set up the environment

From the project root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional, for the stage 5 scale-out path (`opd/trl_grpo.py`):

```bash
git clone https://github.com/huggingface/trl.git third_party/trl
uv pip install -e third_party/trl            # any machine
uv pip install -e "third_party/trl[vllm]"    # Linux + CUDA only (A100)
```

Notes:

- `-e` installs in editable mode, so edits take effect without reinstalling.
- vLLM needs Linux with a CUDA GPU. On the Mac, everything runs on PyTorch's
  MPS backend without vLLM.
- The first run downloads Qwen3-1.7B (~3.4 GB) and Qwen3-8B (~16.4 GB) from
  the Hugging Face Hub, and GSM8K for stage 1.

## Check the install

```bash
python -m pytest -q opd     # ~2 minutes on CPU, no downloads; TRL tests skip without TRL
```

Then a minutes-long smoke run with the smallest Qwen3 models:

```bash
python -m opd.main --config=opd/configs/stage1_distill.py:debug --workdir=runs/debug
```

## Run the stages

Pick the preset after the colon: `mps` (Apple M5 Pro, 48 GB) or `a100`
(80 GB). Chain stages with `--config.model.init_checkpoint=runs/<previous>/final`.

| Stage | Command |
| ----- | ------- |
| 1. On-policy distillation (GSM8K) | `python -m opd.main --config=opd/configs/stage1_distill.py:mps --workdir=runs/stage1` |
| 2. Single-tool agent distillation | `python -m opd.main --config=opd/configs/stage2_single_tool.py:mps --config.env=sqlite --workdir=runs/stage2_sqlite` |
| 3. Multi-tool, multi-turn | `python -m opd.main --config=opd/configs/stage3_multi_tool.py:mps --workdir=runs/stage3` |
| 4. Distillation + task reward | `python -m opd.main --config=opd/configs/stage4_distill_reward.py:mps --workdir=runs/stage4` |
| 5. GRPO agent RL | `python -m opd.main --config=opd/configs/stage5_grpo.py:mps --workdir=runs/stage5` |
| 5. At scale (TRL + vLLM) | `python -m opd.trl_grpo --config=opd/configs/stage5_grpo.py:a100 --use_vllm --workdir=runs/stage5_trl` |

Each run writes `config.json`, `metrics.jsonl` (one JSON line per step),
`samples.jsonl` (decoded episodes, tool results wrapped in `[env]...[/env]`)
and checkpoints (`step_N/`, `final/`) to its workdir.

## Code map

| File | What it does |
| ---- | ------------ |
| `opd/losses.py` | The math of every stage: log-probs, reverse KL, group advantages, clipped policy loss. |
| `opd/rollout.py` | Multi-turn sampling: student turns (action mask 1) and tool results (mask 0). |
| `opd/chat.py` | Qwen3 / ChatML prompt rendering and tool-call parsing. |
| `opd/trainer.py` | The loop: rollout, score with frozen models, update. |
| `opd/models.py` | Loading the student (LoRA or full), teacher and reference. |
| `opd/envs/` | Calculator, SQLite, JSON and multi-tool environments; GSM8K prompts. |
| `opd/configs/` | One config per stage plus shared defaults and hardware presets. |
| `opd/trl_grpo.py` | Stage 5 on TRL's GRPOTrainer, for vLLM and multiple GPUs. |
