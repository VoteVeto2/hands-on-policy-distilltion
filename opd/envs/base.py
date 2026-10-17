"""Environment conventions, tool dispatch, and answer grading.

An environment is the world one episode runs in. Its conventions match TRL's
`environment_factory`, so the same class trains with this repo's trainer and
with `trl.GRPOTrainer` at scale (see `opd/trl_grpo.py`):

  * `reset(**task)` prepares the state for one task. Overrides must accept and
    forward `**task` (`super().reset(**task)`), so that environments compose
    by multiple inheritance (see `multi_tool.py`).
  * Every other public method is a tool. Tools need type hints and a
    Google-style docstring: the chat template turns them into the JSON schema
    the model reads in its system prompt. The docstring *is* the tool's API
    documentation for the LLM, so write it for the LLM.

A task is a plain dict with at least "question", "answer" and "kind" keys, plus
whatever seeds the environment needs to rebuild its state.
"""

import dataclasses
import inspect
import math
import random
import re
from typing import Any, Callable

from opd import chat

Task = dict[str, Any]


class Env:
  """An environment with no tools: the policy just answers."""

  def reset(self, **task: Any) -> None:
    del task  # The base environment has no state.


@dataclasses.dataclass(frozen=True)
class EnvSpec:
  """How to build an environment and sample its tasks.

  Attributes:
    env_cls: Instantiated once per episode, then `reset(**task)`.
    sample_task: Maps `(rng, split)` to a task; split is "train" or "eval".
  """

  env_cls: type[Env]
  sample_task: Callable[[random.Random, str], Task]


def get_tools(env: Env) -> dict[str, Callable[..., str]]:
  """Returns the environment's tools by name: public methods except `reset`."""
  return {
      name: getattr(env, name)
      for name, _ in inspect.getmembers(type(env), inspect.isfunction)
      if not name.startswith("_") and name != "reset"
  }


def call_tool(tools: dict[str, Callable[..., str]], call: chat.ToolCall) -> str:
  """Runs one tool call. Any failure becomes an error string for the model.

  Errors are feedback, not crashes: a real API also answers a bad request with
  an error message, and the policy has to learn to read it and recover.
  """
  fn = tools.get(call.name)
  if fn is None:
    return f"Error: unknown tool {call.name!r}. Available: {sorted(tools)}."
  try:
    return str(fn(**call.arguments))
  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"Error: {type(e).__name__}: {e}"


def grade(completion: str, answer: str) -> float:
  """1.0 if the last `\\boxed{}` of the final turn matches `answer`, else 0.0."""
  prediction = chat.extract_boxed(chat.strip_thinking(completion))
  if prediction is None:
    return 0.0
  return float(answers_match(prediction, answer))


def answers_match(prediction: str, answer: str) -> bool:
  """Numeric comparison when both parse as numbers, else normalised strings."""
  prediction, answer = _normalize(prediction), _normalize(answer)
  try:
    return math.isclose(float(prediction), float(answer), abs_tol=1e-6)
  except ValueError:
    return prediction == answer


def _normalize(text: str) -> str:
  text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
  text = text.replace("\\$", "").replace("$", "").replace(",", "")
  return text.strip().strip(".'\"").strip().lower()
