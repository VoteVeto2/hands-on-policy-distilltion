"""Name -> environment lookup used by configs (`config.env`).

To add a task: write an `Env` subclass and a `sample_task` function in a new
module, then register its `EnvSpec` here.
"""

from opd.envs import base
from opd.envs import calculator
from opd.envs import gsm8k
from opd.envs import json_env
from opd.envs import multi_tool
from opd.envs import sqlite_env

SPECS = {
    "gsm8k": gsm8k.SPEC,
    "calculator": calculator.SPEC,
    "sqlite": sqlite_env.SPEC,
    "json": json_env.SPEC,
    "multi_tool": multi_tool.SPEC,
}


def get(name: str) -> base.EnvSpec:
  if name not in SPECS:
    raise ValueError(f"Unknown env {name!r}; choose from {sorted(SPECS)}.")
  return SPECS[name]
