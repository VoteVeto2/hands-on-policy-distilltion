r"""Runs one training stage.

Usage:
  python -m opd.main --config=opd/configs/stage1_distill.py:mps \
      --workdir=runs/stage1

Everything in the config can be overridden from the command line, e.g.
`--config.train.lr=3e-5 --config.env=sqlite`.
"""

from absl import app
from absl import flags
from ml_collections import config_flags

from opd import trainer

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Stage config file, optionally ':<hardware>'."
)
_WORKDIR = flags.DEFINE_string(
    "workdir", None, "Output directory: metrics, samples, checkpoints."
)
flags.mark_flags_as_required(["config", "workdir"])


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  trainer.train(_CONFIG.value, _WORKDIR.value)


if __name__ == "__main__":
  app.run(main)
