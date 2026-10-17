"""Sampling episodes: the student talks, the environment answers.

An episode is one flat token sequence:

  [prompt] [student turn] [tool results] [student turn] ... [final answer]
            mask = 1       mask = 0       mask = 1             mask = 1

Only mask-1 tokens are the policy's actions. Tool results are observations:
the loss never touches them, but they stay in the context, so both the student
and the teacher condition on them when scoring the next action.

Every stage uses this file. Stage 1 is the special case `max_turns = 1` with
an environment that has no tools.
"""

import contextlib
import dataclasses
from typing import Any

import torch
import transformers

from opd import chat
from opd.envs import base


@dataclasses.dataclass
class Episode:
  """One rollout. Token lists are what the model saw and is trained on.

  `turns` holds the decoded student turns; the last one is graded. Logs show
  `render(episode)` instead, which also includes the tool results.
  """

  task: base.Task
  group: int  # Index of the prompt; samples of one prompt form a GRPO group.
  prompt_ids: list[int]
  completion_ids: list[int] = dataclasses.field(default_factory=list)
  action_mask: list[int] = dataclasses.field(default_factory=list)
  turns: list[str] = dataclasses.field(default_factory=list)
  num_tool_calls: int = 0
  num_tool_errors: int = 0
  truncated: bool = False
  done: bool = False
  reward: float = 0.0


def autocast(device: torch.device) -> contextlib.AbstractContextManager[Any]:
  """bf16 autocast on CUDA, where full fine-tuning keeps fp32 master weights.

  On MPS we train LoRA on a bf16 base (adapters stay fp32 inside PEFT) and on
  CPU (tests) everything is fp32, so neither needs autocast.
  """
  return torch.autocast(
      device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
  )


@torch.no_grad()
def generate(
    model: transformers.PreTrainedModel,
    sequences: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    stop_ids: list[int],
    pad_id: int,
) -> list[list[int]]:
  """Samples one continuation per sequence, up to and including a stop token.

  Args:
    model: Policy or teacher.
    sequences: Unpadded token ids; they are left-padded here.
    max_new_tokens: Generation budget per sequence.
    temperature: 1.0 samples from the policy itself (required for training,
      see below); 0.0 is greedy decoding (used for evaluation).
    stop_ids: Generation ends after any of these tokens.
    pad_id: Padding token id.

  Returns:
    The new token ids of each sequence. A sequence that hit `max_new_tokens`
    does not end with a stop token.
  """
  device = model.device
  width = max(map(len, sequences))
  input_ids = torch.full((len(sequences), width), pad_id, device=device)
  attention_mask = torch.zeros_like(input_ids)
  for i, seq in enumerate(sequences):
    input_ids[i, width - len(seq) :] = torch.tensor(seq, device=device)
    attention_mask[i, width - len(seq) :] = 1
  if temperature > 0:
    # Sample from exactly the distribution we compute log-probs for. Without
    # these overrides transformers applies top_k=50 by default, and Qwen3's
    # generation_config.json adds top_k=20, top_p=0.95, temperature=0.6:
    # truncated sampling makes the "on-policy" data off-policy.
    sampling = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
  else:
    sampling = dict(do_sample=False)
  with autocast(device):
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=stop_ids,
        pad_token_id=pad_id,
        min_p=None,
        repetition_penalty=1.0,
        **sampling,
    )
  results = []
  for row in output[:, width:].tolist():
    stop = next((i for i, t in enumerate(row) if t in stop_ids), None)
    results.append(row if stop is None else row[: stop + 1])
  return results


def sample_episodes(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    spec: base.EnvSpec,
    tasks: list[base.Task],
    samples_per_task: int,
    config: Any,
    temperature: float = 1.0,
) -> list[Episode]:
  """Runs multi-turn episodes for all tasks in lockstep, batching generation.

  Args:
    model: The policy (or the teacher, for a baseline evaluation).
    tokenizer: Shared tokenizer.
    spec: Environment to run.
    tasks: One entry per prompt.
    samples_per_task: Episodes per task (the GRPO group size).
    config: The `rollout` section of the experiment config.
    temperature: Sampling temperature.

  Returns:
    `len(tasks) * samples_per_task` finished, graded episodes.
  """
  stop_ids, pad_id = chat.stop_token_ids(tokenizer), tokenizer.pad_token_id
  episodes, envs = [], []
  for group, task in enumerate(tasks):
    for _ in range(samples_per_task):
      env = spec.env_cls()
      env.reset(**task)
      tools = base.get_tools(env)
      prompt_ids = chat.encode_prompt(
          tokenizer,
          task["question"],
          list(tools.values()),
          config.enable_thinking,
      )
      episodes.append(Episode(task=task, group=group, prompt_ids=prompt_ids))
      envs.append(tools)

  for turn in range(config.max_turns):
    active = [i for i, e in enumerate(episodes) if not e.done]
    for start in range(0, len(active), config.batch_size):
      chunk = active[start : start + config.batch_size]
      rooms = [
          config.max_completion_tokens - len(episodes[i].completion_ids)
          for i in chunk
      ]
      new_ids = generate(
          model,
          [episodes[i].prompt_ids + episodes[i].completion_ids for i in chunk],
          min(config.max_new_tokens, max(rooms)),
          temperature,
          stop_ids,
          pad_id,
      )
      last_turn = turn == config.max_turns - 1
      for i, ids, room in zip(chunk, new_ids, rooms):
        _step(
            episodes[i], envs[i], ids[:room], tokenizer, config, stop_ids,
            last_turn,
        )  # fmt: skip

  for episode in episodes:
    # An unfinished episode earns nothing, even if a draft answer is in it.
    if not episode.truncated:
      episode.reward = base.grade(episode.turns[-1], episode.task["answer"])
  return episodes


def _step(
    episode: Episode,
    tools: dict[str, Any],
    new_ids: list[int],
    tokenizer: transformers.PreTrainedTokenizerBase,
    config: Any,
    stop_ids: list[int],
    last_turn: bool,
) -> None:
  """Appends one student turn and, if it called tools, their results."""
  episode.completion_ids += new_ids
  episode.action_mask += [1] * len(new_ids)
  text = tokenizer.decode(new_ids, skip_special_tokens=True)
  episode.turns.append(text)
  if not new_ids or new_ids[-1] not in stop_ids:  # Ran out of tokens.
    episode.truncated = episode.done = True
    return
  calls = chat.parse_tool_calls(chat.strip_thinking(text))
  # A turn without tool calls is the final answer. So is one that ends with
  # <|endoftext|> instead of <|im_end|>: the model ended the conversation.
  if not calls or new_ids[-1] != tokenizer.convert_tokens_to_ids("<|im_end|>"):
    episode.done = True
    return
  if last_turn:  # Wants another tool round but has no turns left.
    episode.truncated = episode.done = True
    return
  malformed = (
      "Error: malformed <tool_call>; expected JSON with name and arguments."
  )
  results = [
      malformed if c is None else base.call_tool(tools, c) for c in calls
  ]
  episode.num_tool_calls += len(calls)
  episode.num_tool_errors += sum(r.startswith("Error:") for r in results)
  result_ids = chat.encode_tool_results(
      tokenizer, results, config.enable_thinking
  )
  if (
      len(episode.completion_ids) + len(result_ids)
      >= config.max_completion_tokens
  ):
    episode.truncated = episode.done = True
    return
  episode.completion_ids += result_ids
  episode.action_mask += [0] * len(result_ids)


def render(
    episode: Episode, tokenizer: transformers.PreTrainedTokenizerBase
) -> str:
  """Decodes an episode, wrapping environment tokens in [env]...[/env]."""
  pieces, start = [], 0
  mask = episode.action_mask
  for i in range(1, len(mask) + 1):
    if i == len(mask) or mask[i] != mask[start]:
      text = tokenizer.decode(episode.completion_ids[start:i])
      pieces.append(text if mask[start] else f"[env]{text}[/env]")
      start = i
  return "".join(pieces)
