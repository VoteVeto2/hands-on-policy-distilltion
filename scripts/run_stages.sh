#!/bin/bash
#
# Runs stages FIRST..LAST in order, each starting from the previous stage's
# final checkpoint, and packs every finished stage into a tarball.
#
# Usage: scripts/run_stages.sh PRESET [FIRST] [LAST]
#   PRESET: debug, mps or a100 (see opd/configs/base.py).
#
# A stage whose workdir already has final/ is skipped, so rerunning after an
# interruption resumes at the first unfinished stage. That stage restarts from
# step 0: checkpoints hold no optimizer state.
#
# Environment:
#   STAGE2_ENV  Stage 2 environment: calculator (default), sqlite or json.
#   EXTRA       Extra flags for every stage, e.g. "--config.eval.every=50".
#   SAVE        Directory to copy each stage's tarball to (e.g. Google Drive).
#   PYTHON      Interpreter; default .venv/bin/python if present, else python3.
#   BACKEND     main (opd/trainer.py, the default) or trl (opd/trl_train.py;
#               add EXTRA=--use_vllm on CUDA).
#
# Outputs under runs/PRESET/ (runs/trl/PRESET/ for BACKEND=trl): one workdir
# per stage, STAGE.tgz (final/ plus metrics, samples and config) and, while
# running, run_stages.pid.

set -euo pipefail

readonly CONFIGS=(stage1_distill stage2_single_tool stage3_multi_tool
  stage4_distill_reward stage5_grpo)

main() {
  local preset="${1:?usage: $0 debug|mps|a100 [FIRST] [LAST]}"
  local first="${2:-1}" last="${3:-5}"
  local stage2_env="${STAGE2_ENV:-calculator}"
  cd "$(dirname "$0")/.."
  local python="${PYTHON:-python3}"
  if [[ -z "${PYTHON:-}" && -x .venv/bin/python ]]; then
    python=.venv/bin/python
  fi
  local module=opd.main root="runs/${preset}"
  if [[ "${BACKEND:-main}" == trl ]]; then
    module=opd.trl_train root="runs/trl/${preset}"
  fi
  local names=(stage1 "stage2_${stage2_env}" stage3 stage4 stage5)
  mkdir -p "${root}"
  echo $$ > "${root}/run_stages.pid"

  local stage i workdir previous
  for stage in $(seq "${first}" "${last}"); do
    i=$((stage - 1))
    workdir="${root}/${names[i]}"
    if [[ -d "${workdir}/final" ]]; then
      echo "=== stage ${stage}: ${workdir}/final exists, skipping"
      continue
    fi
    if [[ -d "${workdir}" ]]; then  # An interrupted run: keep it aside.
      mv "${workdir}" "${workdir}.partial.$(date +%s)"
    fi
    local args=(
      --config="opd/configs/${CONFIGS[i]}.py:${preset}"
      --workdir="${workdir}"
    )
    if (( stage == 2 )); then
      args+=(--config.env="${stage2_env}")
    fi
    if (( stage > 1 )); then
      previous="${root}/${names[i - 1]}/final"
      if [[ ! -d "${previous}" ]]; then
        echo "=== missing ${previous}; run stage $((stage - 1)) first" >&2
        exit 1
      fi
      args+=(--config.model.init_checkpoint="${previous}")
    fi
    echo "=== $(date -u +%FT%TZ) stage ${stage}:" \
      "${python} -m ${module} ${args[*]} ${EXTRA:-}"
    # shellcheck disable=SC2086  # EXTRA is a list of flags.
    "${python}" -m "${module}" "${args[@]}" ${EXTRA:-}
    tar czf "${root}/${names[i]}.tgz" --exclude='step_*' --exclude='checkpoint-*' \
      -C "${root}" "${names[i]}"
    if [[ -n "${SAVE:-}" ]]; then
      mkdir -p "${SAVE}"
      cp "${root}/${names[i]}.tgz" "${SAVE}/"
    fi
    echo "=== $(date -u +%FT%TZ) stage ${stage} done: ${root}/${names[i]}.tgz"
  done
  rm -f "${root}/run_stages.pid"
  echo "=== all done"
}

main "$@"
