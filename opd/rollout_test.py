import random

from absl.testing import absltest
from ml_collections import config_dict

from opd import chat
from opd import rollout
from opd import testing
from opd.envs import calculator
from opd.envs import multi_tool


def _rollout_config(**overrides) -> config_dict.ConfigDict:
  config = config_dict.ConfigDict(dict(
      enable_thinking=False,
      max_turns=3,
      max_new_tokens=16,
      max_completion_tokens=200,
      batch_size=4,
  ))  # fmt: skip
  config.update(overrides)
  return config


class GenerateTest(absltest.TestCase):

  def test_samples_from_the_full_distribution(self):
    # A random tiny model is close to uniform over ~500 tokens. Top-k=50
    # truncation (the transformers default) would cap distinct tokens at 50.
    tokenizer = testing.make_tokenizer()
    model = testing.make_model(len(tokenizer))
    outputs = rollout.generate(
        model, [[1, 2, 3]] * 400, 1, 1.0, stop_ids=[-1],
        pad_id=tokenizer.pad_token_id,
    )  # fmt: skip
    self.assertGreater(len({ids[0] for ids in outputs}), 100)


class StepTest(absltest.TestCase):
  """Drives `_step` with scripted turns: a random model never calls tools."""

  def setUp(self):
    super().setUp()
    self.tokenizer = testing.make_tokenizer()
    self.stop_ids = chat.stop_token_ids(self.tokenizer)
    self.env = calculator.CalculatorEnv()
    self.tools = {"calculator": self.env.calculator}

  def _step(self, episode, text, last_turn=False):
    ids = self.tokenizer.encode(text, add_special_tokens=False)
    rollout._step(  # pylint: disable=protected-access
        episode, self.tools, ids, self.tokenizer, _rollout_config(),
        self.stop_ids, last_turn,
    )  # fmt: skip
    return ids

  def test_tool_turn_then_answer(self):
    episode = rollout.Episode(task={"answer": "6"}, group=0, prompt_ids=[1])
    call = self._step(
        episode,
        "<tool_call>\n<function=calculator>\n<parameter=expression>\n2*3\n"
        "</parameter>\n</function>\n</tool_call><|im_end|>",
    )
    self.assertFalse(episode.done)
    self.assertEqual(episode.num_tool_calls, 1)
    self.assertIn(
        "[env]\n<|im_start|>user\n<tool_response>\n6\n</tool_response>",
        rollout.render(episode, self.tokenizer),
    )
    self.assertEqual(episode.action_mask.count(1), len(call))

    answer = self._step(episode, r"\boxed{6}<|im_end|>")
    self.assertTrue(episode.done)
    self.assertFalse(episode.truncated)
    self.assertEqual(episode.action_mask[-len(answer) :], [1] * len(answer))
    self.assertLen(episode.completion_ids, len(episode.action_mask))

  def test_tool_call_on_last_turn_truncates(self):
    episode = rollout.Episode(task={}, group=0, prompt_ids=[1])
    self._step(
        episode,
        "<tool_call>\n<function=calculator>\n<parameter=expression>\n1\n"
        "</parameter>\n</function>\n</tool_call><|im_end|>",
        last_turn=True,
    )
    self.assertTrue(episode.truncated)
    self.assertNotIn(0, episode.action_mask)

  def test_results_follow_call_order(self):
    episode = rollout.Episode(task={}, group=0, prompt_ids=[1])
    self._step(
        episode,
        "<tool_call>{oops}</tool_call>"
        "<tool_call>\n<function=calculator>\n<parameter=expression>\n2*3\n"
        "</parameter>\n</function>\n</tool_call><|im_end|>",
    )
    text = rollout.render(episode, self.tokenizer)
    self.assertLess(text.index("Error: malformed"), text.index("\n6\n"))

  def test_turns_that_end_the_episode(self):
    call = (
        "<tool_call>\n<function=calculator>\n<parameter=expression>\n1\n"
        "</parameter>\n</function>\n"
    )
    for turn in (
        # Ends with <|endoftext|>: the conversation is over.
        call + "</tool_call><|endoftext|>",
        # The call is only a thought.
        "<think>" + call + "</tool_call></think>\\boxed{1}<|im_end|>",
    ):
      episode = rollout.Episode(task={}, group=0, prompt_ids=[1])
      self._step(episode, turn)
      self.assertTrue(episode.done)
      self.assertFalse(episode.truncated)
      self.assertEqual(episode.num_tool_calls, 0)

  def test_malformed_call_gets_error_feedback(self):
    episode = rollout.Episode(task={}, group=0, prompt_ids=[1])
    self._step(episode, "<tool_call>{oops}</tool_call><|im_end|>")
    self.assertEqual(episode.num_tool_errors, 1)
    self.assertIn("Error: malformed", rollout.render(episode, self.tokenizer))


class SampleEpisodesTest(absltest.TestCase):

  def test_invariants_with_random_model(self):
    tokenizer = testing.make_tokenizer()
    model = testing.make_model(len(tokenizer))
    rng = random.Random(0)
    tasks = [multi_tool.sample_task(rng, "train") for _ in range(3)]
    config = _rollout_config(max_completion_tokens=40, batch_size=2)
    episodes = rollout.sample_episodes(
        model, tokenizer, multi_tool.SPEC, tasks, 2, config
    )
    self.assertLen(episodes, 6)
    self.assertEqual([e.group for e in episodes], [0, 0, 1, 1, 2, 2])
    for episode in episodes:
      self.assertTrue(episode.done)
      self.assertLen(episode.completion_ids, len(episode.action_mask))
      self.assertLessEqual(len(episode.completion_ids), 40)
      self.assertIn(episode.reward, (0.0, 1.0))
      if episode.truncated:
        self.assertEqual(episode.reward, 0.0)


if __name__ == "__main__":
  absltest.main()
