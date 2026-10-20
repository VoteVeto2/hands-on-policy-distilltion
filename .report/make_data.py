"""Writes the evaluation curves plotted in the report, one file per run.

Usage, from the repository root:
  python .report/make_data.py

Reads results/trl/<run>/<stage>/trainer_state.json and writes
.report/data/<name>.dat: one row per evaluation, with the steps of stages 1-4
laid end to end, and a row of nan between stages so that the plotted lines
break there.
"""

import json
import os

RUNS = {"qwen3": "results/trl/a100", "qwen35": "results/trl/a100-qwen3.5"}
STAGES = ("stage1", "stage2_calculator", "stage3", "stage4")
COLUMNS = {  # Column in the .dat file: key in trainer_state.json.
    "kl": "eval_teacher_kl",
    "reward": "eval_reward",
    "length": "eval_completions/mean_length",
    "entropy": "eval_entropy",
}


def main() -> None:
  out_dir = os.path.join(os.path.dirname(__file__), "data")
  os.makedirs(out_dir, exist_ok=True)
  for name, run in RUNS.items():
    lines = ["step stage " + " ".join(COLUMNS)]
    offset = 0
    for stage, directory in enumerate(STAGES, start=1):
      with open(os.path.join(run, directory, "trainer_state.json")) as f:
        state = json.load(f)
      if stage > 1:
        lines.append(" ".join(["nan"] * (2 + len(COLUMNS))))
      for row in state["log_history"]:
        if "eval_reward" in row:
          values = " ".join(f"{row[key]:.4f}" for key in COLUMNS.values())
          lines.append(f"{offset + row['step']} {stage} {values}")
      offset += state["max_steps"]
    with open(os.path.join(out_dir, f"{name}.dat"), "w") as f:
      f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
  main()
