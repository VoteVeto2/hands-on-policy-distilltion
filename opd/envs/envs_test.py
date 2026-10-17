"""Tests for the environments: tools, task generators and grading."""

import random

from absl.testing import absltest
from absl.testing import parameterized

from opd import chat
from opd.envs import base
from opd.envs import calculator
from opd.envs import json_env
from opd.envs import multi_tool
from opd.envs import registry
from opd.envs import sqlite_env


class GradeTest(parameterized.TestCase):

  @parameterized.parameters(
      (r"so \boxed{1,234}", "1234", 1.0),
      (r"\boxed{\$1234.0}", "1234", 1.0),
      (r"\boxed{ada abe}", "Ada Abe", 1.0),
      (r"<think>\boxed{7}</think> \boxed{8}", "7", 0.0),
      ("7", "7", 0.0),  # The answer must be boxed.
  )
  def test_grade(self, completion, answer, expected):
    self.assertEqual(base.grade(completion, answer), expected)


class CalculatorTest(parameterized.TestCase):

  @parameterized.parameters(
      ("(1234 + 4321) * 17", "94435"),
      ("12,345 * 2", "24690"),
      ("2 ^ 10", "1024"),
      ("7 / 2", "3.5"),
      ("99999999999 * 99999999999", "9999999999800000000001"),
  )
  def test_calculator(self, expression, expected):
    self.assertEqual(calculator.CalculatorEnv().calculator(expression), expected)

  @parameterized.parameters("__import__('os')", "len('a')", "2 ** 1000", "x + 1")
  def test_rejects_non_arithmetic(self, expression):
    tools = base.get_tools(calculator.CalculatorEnv())
    result = base.call_tool(tools, chat.ToolCall("calculator", {"expression": expression}))
    self.assertStartsWith(result, "Error:")


class SqliteTest(absltest.TestCase):

  def test_database_is_read_only_and_bounded(self):
    env = sqlite_env.SqliteEnv()
    env.reset(db_seed=3)
    tools = base.get_tools(env)
    self.assertEqual(env.query_sql("SELECT COUNT(*) FROM employees"), "COUNT(*)\n40")
    for sql, error in (
        ("DELETE FROM employees", "readonly"),
        ("SELECT COUNT(*) FROM employees a, employees b, employees c, employees d", "expensive"),
    ):
      result = base.call_tool(tools, chat.ToolCall("query_sql", {"sql": sql}))
      self.assertStartsWith(result, "Error:")
      self.assertIn(error, result)

  def test_same_seed_same_database(self):
    query = "SELECT name, salary FROM employees ORDER BY id"
    first, second = sqlite_env.SqliteEnv(), sqlite_env.SqliteEnv()
    first.reset(db_seed=7)
    second.reset(db_seed=7)
    self.assertEqual(first.query_sql(query), second.query_sql(query))


class JsonTest(absltest.TestCase):

  def test_query(self):
    env = json_env.JsonEnv()
    env.reset(doc_seed=1)
    doc = json_env.build_document(1)
    name = doc["products"][0]["name"]
    price = env.json_query(f"products[?name=='{name}'].price | [0]")
    self.assertEqual(price, str(doc["products"][0]["price"]))
    self.assertEqual(env.json_query("length(orders)"), "20")


class MultiToolTest(absltest.TestCase):

  def test_inherits_all_tools_and_state(self):
    env = multi_tool.MultiToolEnv()
    env.reset(**multi_tool.sample_task(random.Random(0), "train"))
    self.assertEqual(
        sorted(base.get_tools(env)), ["calculator", "json_query", "query_sql"]
    )
    self.assertEqual(env.query_sql("SELECT COUNT(*) FROM projects"), "COUNT(*)\n12")
    self.assertEqual(env.json_query("length(products)"), "16")

  def test_every_kind_is_sampled(self):
    rng = random.Random(0)
    kinds = {multi_tool.sample_task(rng, "train")["kind"] for _ in range(300)}
    self.assertLen(kinds, 7)


class RegistryTest(parameterized.TestCase):

  @parameterized.parameters("calculator", "sqlite", "json", "multi_tool")
  def test_tasks_have_consistent_keys(self, name):
    # Consistent keys let a list of tasks become a `datasets.Dataset`.
    spec = registry.get(name)
    rng = random.Random(0)
    keys = {tuple(sorted(spec.sample_task(rng, "train"))) for _ in range(50)}
    self.assertLen(keys, 1)
    self.assertContainsSubset({"question", "answer", "kind"}, next(iter(keys)))


if __name__ == "__main__":
  absltest.main()
