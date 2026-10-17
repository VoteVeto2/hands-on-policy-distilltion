"""Tiny offline stand-ins for Qwen3, used by the unit tests.

The real pipeline downloads Qwen3 checkpoints from the Hugging Face Hub. Tests
must run offline in seconds, so they use a byte-level BPE tokenizer trained on
the fly (with Qwen3's special tokens and Qwen3's verbatim chat template) and
randomly initialised `Qwen3ForCausalLM` models with ~100k parameters.
"""

import os
import pathlib

import tokenizers
import torch
import transformers

_TEMPLATE = pathlib.Path(__file__).parent / "testdata" / "qwen3_chat_template.jinja"

# Qwen3 marks these as special: `decode(skip_special_tokens=True)` drops them.
_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
# Qwen3 adds these as ordinary tokens: they survive decoding.
_ADDED_TOKENS = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]
_CORPUS = [
    "You are a careful assistant. Use tools when they help.",
    'Compute 48213 * 9127 - 3311. {"name": "calculator", "arguments": ',
    '{"expression": "(1234 + 4321) * 17"}} The answer is \\boxed{94435}.',
    "SELECT COUNT(*) FROM employees WHERE department = 'Sales';",
    "products[?name=='Oak Desk'].price | [0] customers orders items",
    "user assistant system tool function parameters description type",
] * 20


def make_tokenizer() -> transformers.PreTrainedTokenizerBase:
  """A ~500-token byte-level BPE tokenizer with Qwen3's chat format."""
  tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
  byte_level = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
  tokenizer.pre_tokenizer = byte_level
  tokenizer.decoder = tokenizers.decoders.ByteLevel()
  trainer = tokenizers.trainers.BpeTrainer(
      vocab_size=500,
      special_tokens=_SPECIAL_TOKENS,
      initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
      show_progress=False,
  )
  tokenizer.train_from_iterator(_CORPUS, trainer)
  hf_tokenizer = transformers.PreTrainedTokenizerFast(
      tokenizer_object=tokenizer,
      eos_token="<|im_end|>",
      pad_token="<|endoftext|>",
  )
  hf_tokenizer.add_tokens(_ADDED_TOKENS)
  hf_tokenizer.chat_template = _TEMPLATE.read_text()
  return hf_tokenizer


def make_model(
    vocab_size: int, hidden_size: int = 32, seed: int = 0
) -> transformers.PreTrainedModel:
  """A randomly initialised two-layer Qwen3."""
  torch.manual_seed(seed)
  config = transformers.Qwen3Config(
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      intermediate_size=2 * hidden_size,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=hidden_size // 4,
      max_position_embeddings=8192,
      tie_word_embeddings=True,
  )
  return transformers.Qwen3ForCausalLM(config)


def save_tiny_checkpoints(root: str) -> tuple[str, str]:
  """Writes a tiny student and a wider teacher that share one tokenizer.

  Args:
    root: Directory to write into.

  Returns:
    Paths of the student and teacher checkpoints.
  """
  tokenizer = make_tokenizer()
  paths = []
  for name, hidden_size, seed in (("student", 32, 0), ("teacher", 64, 1)):
    path = os.path.join(root, name)
    make_model(len(tokenizer), hidden_size, seed).save_pretrained(path)
    tokenizer.save_pretrained(path)
    paths.append(path)
  return paths[0], paths[1]
