# Onboarding

A hands-on post-training pipeline: on-policy distillation from Qwen3.5-9B to
Qwen3.5-2B, growing into tool-using agents and GRPO. Open
`.onboard/guide.html` in a browser for the step-by-step walkthrough; this file
is the short version.

## Set up the environment

From the project root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

TRL, for the TRL backend (`opd/trl_train.py`), is pinned as a git submodule in
`trl/`:

```bash
git submodule update --init trl
uv pip install -e trl --config-settings editable_mode=compat              # any machine
uv pip install -e "trl[vllm]" --config-settings editable_mode=compat      # Linux + CUDA only (A100)
uv pip install flash-linear-attention                                     # Linux + CUDA only (A100)
```

Notes:

- `-e` installs in editable mode, so edits take effect without reinstalling.
- `editable_mode=compat` matters: from the project root, the `trl/` checkout
  directory would otherwise shadow the installed package (`import trl` finds
  an empty namespace package).
- vLLM needs Linux with a CUDA GPU. On the Mac, everything runs on PyTorch's
  MPS backend without vLLM.
- Three in four Qwen3.5 layers are Gated DeltaNet (linear attention).
  `flash-linear-attention` gives transformers their Triton kernels; without it
  (always on the Mac) it falls back to a correct but much slower PyTorch
  implementation. vLLM ships its own kernels.
- The first run downloads Qwen3.5-2B (~4.5 GB) and Qwen3.5-9B (~19.3 GB) from
  the Hugging Face Hub, and GSM8K for stage 1. Both checkpoints include a
  vision tower; the tasks here are text only.

## Check the install

```bash
python -m pytest -q opd     # ~2 minutes on CPU, no downloads; TRL tests skip without TRL
```

Then a minutes-long smoke run with the smallest Qwen3.5 models:

```bash
python -m opd.main --config=opd/configs/stage1_distill.py:debug --workdir=runs/debug
python -m opd.trl_train --config=opd/configs/stage1_distill.py:debug --workdir=runs/trl/debug
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

Or run them all in order, each stage starting from the previous one's
`final/` (stages that already finished are skipped; see the script header for
options):

```bash
scripts/run_stages.sh mps        # or: debug, a100; optionally FIRST LAST
```

Each run writes `config.json`, `metrics.jsonl` (one JSON line per step),
`samples.jsonl` (decoded episodes, tool results wrapped in `[env]...[/env]`)
and checkpoints (`step_N/`, `final/`) to its workdir.

### On TRL

Every stage also runs on TRL's `GRPOTrainer`, from the same configs:
`OPDTrainer` (`opd/trl_train.py`) adds the teacher's per-token term to TRL's
advantage. Swap `opd.main` for `opd.trl_train`:

```bash
python -m opd.trl_train --config=opd/configs/stage1_distill.py:mps --workdir=runs/trl/stage1
python -m opd.trl_train --config=opd/configs/stage1_distill.py:a100 --use_vllm --workdir=runs/trl/stage1  # CUDA
accelerate launch -m opd.trl_train --config=... --use_vllm --workdir=...  # several GPUs
BACKEND=trl scripts/run_stages.sh mps  # all five, chained, under runs/trl/mps
```

A TRL run writes `trainer_state.json` (every logged metric, including
`teacher_kl` and `eval_reward`), `completions/*.parquet` (sampled episodes),
`checkpoint-N/` and `final/`. Differences from `opd.main`: only the "sampled"
teacher estimator; evaluation samples at the training temperature instead of
greedily, with no teacher baseline; and `ref_kl_coef` becomes TRL's k3 loss
term.

## Code map

| File | What it does |
| ---- | ------------ |
| `opd/losses.py` | The math of every stage: log-probs, reverse KL, group advantages, clipped policy loss. |
| `opd/rollout.py` | Multi-turn sampling: student turns (action mask 1) and tool results (mask 0). |
| `opd/chat.py` | Qwen3.5 / ChatML prompt rendering and tool-call parsing. |
| `opd/trainer.py` | The loop: rollout, score with frozen models, update. |
| `opd/models.py` | Loading the student (LoRA or full), teacher and reference. |
| `opd/envs/` | Calculator, SQLite, JSON and multi-tool environments; GSM8K prompts. |
| `opd/configs/` | One config per stage plus shared defaults and hardware presets. |
| `opd/trl_train.py` | Every stage on TRL's GRPOTrainer; `OPDTrainer` adds the teacher term to the advantage. For vLLM and multiple GPUs. |
