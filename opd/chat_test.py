from absl.testing import absltest

from opd import chat
from opd import testing
from opd.envs import calculator


class EncodeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tokenizer = testing.make_tokenizer()

  def test_tool_round_trip_matches_chatml_format(self):
    prompt = chat.encode_prompt(self.tokenizer, "Q?", [], enable_thinking=False)
    sampled = self.tokenizer.encode(
        '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call><|im_end|>',
        add_special_tokens=False,
    )
    results = chat.encode_tool_results(
        self.tokenizer, ["4", "5"], enable_thinking=False
    )
    text = self.tokenizer.decode(prompt + sampled + results)
    self.assertEndsWith(
        text,
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call><|im_end|>\n'
        "<|im_start|>user\n<tool_response>\n4\n</tool_response>\n"
        "<tool_response>\n5\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )

  def test_rerendering_history_would_change_sampled_tokens(self):
    # Why rollouts append tokens instead of re-rendering the conversation:
    # Qwen3's template drops the empty think block of past assistant turns.
    messages = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "<think>\n\n</think>\n\nA."},
        {"role": "user", "content": "And?"},
    ]
    text = self.tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    self.assertIn("<|im_start|>assistant\nA.<|im_end|>", text)

  def test_prompt_lists_tool_schema(self):
    env = calculator.CalculatorEnv()
    ids = chat.encode_prompt(self.tokenizer, "Q?", [env.calculator], False)
    text = self.tokenizer.decode(ids)
    self.assertIn('"name": "calculator"', text)
    self.assertIn("Evaluates an arithmetic expression exactly.", text)


class ParseTest(absltest.TestCase):

  def test_parse_tool_calls(self):
    text = (
        'I will compute. <tool_call>\n{"name": "calculator", "arguments":'
        ' {"expression": "2*3"}}\n</tool_call>'
        "<tool_call>not json</tool_call><tool_call>{\"name\": 1}</tool_call>"
        "<tool_call>{unclosed"
    )
    calls, num_malformed = chat.parse_tool_calls(text)
    self.assertEqual(calls, [chat.ToolCall("calculator", {"expression": "2*3"})])
    self.assertEqual(num_malformed, 3)

  def test_extract_boxed(self):
    self.assertEqual(chat.extract_boxed(r"a \boxed{1} b \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
    self.assertIsNone(chat.extract_boxed("no answer"))
    self.assertIsNone(chat.extract_boxed(r"\boxed{unclosed"))

  def test_strip_thinking(self):
    self.assertEqual(chat.strip_thinking(r"<think>\boxed{1}</think>\boxed{2}"), r"\boxed{2}")


if __name__ == "__main__":
  absltest.main()
