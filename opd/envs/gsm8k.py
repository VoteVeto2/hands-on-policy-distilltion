"""Stage 1: GSM8K grade-school math, used as a prompt-only dataset.

On-policy distillation needs only prompts: the student writes the answers and
the teacher grades every token. The reference answers are used for evaluation
only, so you can watch accuracy improve without ever training on a reward.
"""

import functools
import random

import datasets

from opd.envs import base


@functools.cache
def _load(split: str) -> datasets.Dataset:
  return datasets.load_dataset(
      "openai/gsm8k", "main", split="train" if split == "train" else "test"
  )


def sample_task(rng: random.Random, split: str) -> base.Task:
  rows = _load(split)
  row = rows[rng.randrange(len(rows))]
  answer = row["answer"].split("####")[-1].strip().replace(",", "")
  return {"question": row["question"], "answer": answer, "kind": "gsm8k"}


SPEC = base.EnvSpec(base.Env, sample_task)
