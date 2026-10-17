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
from typing import Any, Callable

import transformers

SYSTEM_PROMPT = (
    "You are a careful assistant. Solve the user's task. If tools are"
    " available, call them whenever they help you get an exact result; you may"
    " call tools several times. End with the final answer written as"
    " \\boxed{answer}."
)


@dataclasses.dataclass(frozen=True)
class ToolCall:
  name: str
  arguments: dict[str, Any]


def stop_token_ids(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> list[int]:
  """Token ids that end an assistant turn."""
  return sorted(
      {
          tokenizer.convert_tokens_to_ids("<|im_end|>"),
          tokenizer.convert_tokens_to_ids("<|endoftext|>"),
      }
  )


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
    tools: Tool functions. `apply_chat_template` converts their type hints
      and Google-style docstrings into JSON schemas
      (`transformers.utils.get_json_schema`), which the chat template renders
      into the system prompt.
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


def parse_tool_calls(text: str) -> list[ToolCall | None]:
  """Extracts the tool calls of one assistant turn, in the order written.

  Hermes-style calls carry no ids: the k-th `<tool_response>` answers the k-th
  `<tool_call>`, so malformed calls keep their place in the list.

  Args:
    text: The turn's content, without any thinking (see `strip_thinking`).

  Returns:
    One entry per `<tool_call>` block: a `ToolCall`, or None if the block is
    malformed (unclosed, invalid JSON, or no string `name`).
  """
  calls = []
  for chunk in text.split("<tool_call>")[1:]:
    # A block ends at its closing tag; an unclosed one ends at the next call.
    body, closed, _ = chunk.partition("</tool_call>")
    calls.append(_parse_call(body) if closed else None)
  return calls


def _parse_call(body: str) -> ToolCall | None:
  try:
    obj = json.loads(body)
  except json.JSONDecodeError:
    return None
  if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
    return None
  arguments = obj.get("arguments", {})
  if not isinstance(arguments, dict):
    return None
  return ToolCall(obj["name"], arguments)


def strip_thinking(text: str) -> str:
  """The part of a turn after its reasoning.

  Args:
    text: A decoded assistant turn.

  Returns:
    Everything after the last `</think>`; an empty string if a `<think>` was
    opened and never closed (the turn ended mid-thought).
  """
  if "</think>" in text:
    return text.rsplit("</think>", 1)[-1]
  return "" if "<think>" in text else text


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
