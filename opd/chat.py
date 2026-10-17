"""Prompt rendering and tool-call parsing for ChatML models such as Qwen3.

Everything the policy reads or writes is a flat list of token ids. This module
is the only place that converts between chat messages and those ids.

Assumptions (true for Qwen2.5 and Qwen3; re-check them before switching
model families):
  * Turns look like `<|im_start|>{role}\\n{content}<|im_end|>\\n`.
  * The model calls tools by emitting `<tool_call>{"name": ..., "arguments":
    {...}}</tool_call>` blocks inside its own turn (Hermes format).
  * Tool results go back in a user turn, each wrapped in `<tool_response>`.
"""

import dataclasses
import json
import re
from typing import Any, Callable

import transformers

SYSTEM_PROMPT = (
    "You are a careful assistant. Solve the user's task. If tools are"
    " available, call them whenever they help you get an exact result; you may"
    " call tools several times. End with the final answer written as"
    " \\boxed{answer}."
)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclasses.dataclass(frozen=True)
class ToolCall:
  name: str
  arguments: dict[str, Any]


def stop_token_ids(tokenizer: transformers.PreTrainedTokenizerBase) -> list[int]:
  """Token ids that end an assistant turn."""
  return sorted({
      tokenizer.convert_tokens_to_ids("<|im_end|>"),
      tokenizer.convert_tokens_to_ids("<|endoftext|>"),
  })


def encode_prompt(
    tokenizer: transformers.PreTrainedTokenizerBase,
    question: str,
    tools: list[Callable[..., str]],
    enable_thinking: bool,
) -> list[int]:
  """Renders system and user turns (plus tool schemas) and the reply header.

  Args:
    tokenizer: A tokenizer with a ChatML chat template.
    question: The user's message.
    tools: Tool functions. The chat template converts their type hints and
      Google-style docstrings into JSON schemas inside the system prompt.
    enable_thinking: Qwen3 switch. When False the template pre-fills an empty
      `<think></think>` block so the model answers directly.

  Returns:
    Token ids ending with the assistant header, ready for generation.
  """
  messages = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": question},
  ]
  text = tokenizer.apply_chat_template(
      messages,
      tools=tools or None,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=enable_thinking,
  )
  return tokenizer.encode(text, add_special_tokens=False)


def encode_tool_results(
    tokenizer: transformers.PreTrainedTokenizerBase,
    results: list[str],
    enable_thinking: bool,
) -> list[int]:
  """Tokens the environment appends after a tool-calling assistant turn.

  The policy's turn ended with `<|im_end|>`. ChatML continues with a newline,
  one user turn holding a `<tool_response>` block per result, and a fresh
  assistant header.

  We append these tokens to the ids the policy sampled instead of re-rendering
  the whole conversation with the chat template. Re-rendering can rewrite
  earlier turns (Qwen3's template drops empty think blocks from past turns),
  and then the tokens we train on would no longer be the tokens the policy
  actually sampled.

  Args:
    tokenizer: A tokenizer with a ChatML chat template.
    results: One string per executed tool call, in call order.
    enable_thinking: Must match the value used for the prompt.

  Returns:
    Token ids to append; none of them are policy actions.
  """
  messages = [{"role": "tool", "content": result} for result in results]
  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=enable_thinking,
  )
  return tokenizer.encode("\n" + text, add_special_tokens=False)


def parse_tool_calls(text: str) -> tuple[list[ToolCall], int]:
  """Extracts tool calls from one assistant turn.

  Args:
    text: The decoded assistant turn.

  Returns:
    The well-formed calls, and how many `<tool_call>` blocks were malformed
    (unclosed, invalid JSON, or missing a string `name`).
  """
  bodies = _TOOL_CALL_RE.findall(text)
  num_malformed = text.count("<tool_call>") - len(bodies)
  calls = []
  for body in bodies:
    try:
      obj = json.loads(body)
    except json.JSONDecodeError:
      num_malformed += 1
      continue
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("name"), str)
        or not isinstance(obj.get("arguments", {}), dict)
    ):
      num_malformed += 1
      continue
    calls.append(ToolCall(obj["name"], obj.get("arguments", {})))
  return calls, num_malformed


def strip_thinking(text: str) -> str:
  """Drops everything up to the last `</think>`."""
  return text.rsplit("</think>", 1)[-1]


def extract_boxed(text: str) -> str | None:
  """Returns the content of the last `\\boxed{...}`, honouring nested braces."""
  start = text.rfind("\\boxed{")
  if start < 0:
    return None
  i = start + len("\\boxed{")
  depth = 1
  for j in range(i, len(text)):
    if text[j] == "{":
      depth += 1
    elif text[j] == "}":
      depth -= 1
      if depth == 0:
        return text[i:j]
  return None
