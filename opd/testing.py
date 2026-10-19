"""Tiny offline stand-ins for Qwen3.5, used by the unit tests.

The real pipeline downloads Qwen3.5 checkpoints from the Hugging Face Hub.
Tests must run offline in seconds, so they use a byte-level BPE tokenizer
trained on the fly (with Qwen3.5's special tokens, pre-tokenizer and verbatim
chat template) and randomly initialised Qwen3.5 models with ~100k parameters:
one Gated DeltaNet layer and one attention layer.
"""

import os
import pathlib

import tokenizers
import torch
import transformers

_TEMPLATE = (
    pathlib.Path(__file__).parent / "testdata" / "qwen3_5_chat_template.jinja"
)

# Qwen3.5 marks these as special: `decode(skip_special_tokens=True)` drops them.
_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
# Qwen3.5 adds these as ordinary tokens: they survive decoding.
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
    "Compute 48213 * 9127 - 3311.\n<tool_call>\n<function=calculator>\n",
    "<parameter=expression>\n(1234 + 4321) * 17\n</parameter>\n</function>\n",
    "The answer is \\boxed{94435}.",
    "SELECT COUNT(*) FROM employees WHERE department = 'Sales';",
    "products[?name=='Oak Desk'].price | [0] customers orders items",
    "user assistant system tool function parameters description type",
] * 20


def make_tokenizer() -> transformers.PreTrainedTokenizerBase:
  """A ~500-token byte-level BPE tokenizer with Qwen3.5's chat format."""
  # Trained inside Qwen3.5's own normalizer and pre-tokenizer: `AutoTokenizer`
  # rebuilds those around a saved vocabulary (`Qwen3_5Tokenizer`), so the
  # tokenizer must encode the same way before and after a save.
  tokenizer = transformers.Qwen3_5Tokenizer().backend_tokenizer
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


def _text_config(
    vocab_size: int, hidden_size: int
) -> transformers.Qwen3_5TextConfig:
  head_dim = hidden_size // 4
  return transformers.Qwen3_5TextConfig(
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      intermediate_size=2 * hidden_size,
      num_hidden_layers=2,
      # Qwen3.5 has three Gated DeltaNet (linear attention) layers per full
      # attention layer; one of each here.
      layer_types=["linear_attention", "full_attention"],
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=head_dim,
      linear_num_key_heads=2,
      linear_num_value_heads=4,
      linear_key_head_dim=head_dim,
      linear_value_head_dim=head_dim,
      max_position_embeddings=8192,
      tie_word_embeddings=True,
  )


def make_model(
    vocab_size: int, hidden_size: int = 32, seed: int = 0
) -> transformers.PreTrainedModel:
  """A randomly initialised two-layer Qwen3.5 language model."""
  torch.manual_seed(seed)
  return transformers.Qwen3_5ForCausalLM(_text_config(vocab_size, hidden_size))


def save_tiny_checkpoints(root: str) -> tuple[str, str]:
  """Writes a tiny student and a wider teacher that share one tokenizer.

  Like the Hub's Qwen3.5 checkpoints, they hold a vision tower next to the
  language model: `AutoModelForCausalLM` loads the language model only, while
  TRL loads the whole `Qwen3_5ForConditionalGeneration`.

  Args:
    root: Directory to write into.

  Returns:
    Paths of the student and teacher checkpoints.
  """
  tokenizer = make_tokenizer()
  paths = []
  for name, hidden_size, seed in (("student", 32, 0), ("teacher", 64, 1)):
    torch.manual_seed(seed)
    config = transformers.Qwen3_5Config(
        text_config=_text_config(len(tokenizer), hidden_size).to_dict(),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=hidden_size,
        ),
        tie_word_embeddings=True,
    )
    path = os.path.join(root, name)
    transformers.Qwen3_5ForConditionalGeneration(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    paths.append(path)
  return paths[0], paths[1]
