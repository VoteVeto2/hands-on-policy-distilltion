"""Stage 2: a calculator tool and arithmetic word problems that need it.

The numbers are large enough that a 1.7B model makes mistakes doing the
arithmetic "in its head", so calling the tool is the winning strategy.
"""

import ast
import operator
import random
import re

from opd.envs import base

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate(expression: str) -> int | float:
  """Evaluates arithmetic over number literals. No names, calls or attributes.

  Args:
    expression: For example "(1,234 + 4321) * 17". Thousands separators and
      `^` (as power) are accepted because models write them.

  Returns:
    The exact result: Python ints stay exact at any size.

  Raises:
    ValueError: The expression uses anything but numbers and operators.
    SyntaxError: The expression does not parse.
  """
  expression = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression).replace("^", "**")
  return _evaluate(ast.parse(expression, mode="eval").body)


def _evaluate(node: ast.AST) -> int | float:
  if isinstance(node, ast.Constant) and type(node.value) in (int, float):
    return node.value
  if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
    return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
  if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
    left, right = _evaluate(node.left), _evaluate(node.right)
    if isinstance(node.op, ast.Pow) and abs(right) > 64:
      raise ValueError("exponent larger than 64")
    return _BINARY_OPS[type(node.op)](left, right)
  raise ValueError(f"unsupported expression: {ast.unparse(node)}")


def format_number(x: int | float) -> str:
  if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
    x = int(x)
  return str(x) if isinstance(x, int) else f"{x:.10g}"


class CalculatorEnv(base.Env):
  """Exposes one stateless tool: `calculator`."""

  def calculator(self, expression: str) -> str:
    """Evaluates an arithmetic expression exactly.

    Args:
      expression: Numbers combined with + - * / // % ** and parentheses, for
        example "(1234 + 4321) * 17".

    Returns:
      The result, or an error message.
    """
    return format_number(evaluate(expression))


def _product_minus(rng: random.Random) -> tuple[str, int]:
  a, b, c = (
      rng.randint(1000, 99999),
      rng.randint(100, 9999),
      rng.randint(1000, 99999),
  )
  return f"Compute {a} * {b} - {c}.", a * b - c


def _factory(rng: random.Random) -> tuple[str, int]:
  rate, bad, days = (
      rng.randint(1200, 9800),
      rng.randint(10, 300),
      rng.randint(12, 365),
  )
  question = (
      f"A factory makes {rate} widgets per day, and {bad} of them are"
      f" defective and thrown away. How many good widgets does it make in"
      f" {days} days?"
  )
  return question, (rate - bad) * days


def _tickets(rng: random.Random) -> tuple[str, int]:
  people, price = rng.randint(150, 4000), rng.randint(15, 240)
  cost = rng.randint(1000, people * price - 1)
  question = (
      f"{people} people each buy a ticket for ${price}. The organizers spend"
      f" ${cost} on the venue. How many dollars of profit do they make?"
  )
  return question, people * price - cost


def _remainder(rng: random.Random) -> tuple[str, int]:
  a, m = rng.randint(10**6, 10**8), rng.randint(13, 997)
  return f"What is the remainder when {a} is divided by {m}?", a % m


def _sum(rng: random.Random) -> tuple[str, int]:
  xs = [rng.randint(10**4, 10**6) for _ in range(4)]
  return f"What is {' + '.join(map(str, xs))}?", sum(xs)


def _discount(rng: random.Random) -> tuple[str, int]:
  while True:  # Resample until the discounted total is a whole number.
    units, price = rng.randint(20, 900), rng.randint(12, 480)
    pct = rng.choice([10, 15, 20, 25, 30, 40])
    if units * price * (100 - pct) % 100 == 0:
      break
  question = (
      f"A school buys {units} chairs at ${price} each and gets a {pct}%"
      f" discount on the total. How many dollars does it pay?"
  )
  return question, units * price * (100 - pct) // 100


_PROBLEMS = (_product_minus, _factory, _tickets, _remainder, _sum, _discount)


def sample_problem(rng: random.Random) -> tuple[str, str]:
  """Returns a (question, answer) pair."""
  question, answer = rng.choice(_PROBLEMS)(rng)
  return question, str(answer)


def sample_task(rng: random.Random, split: str) -> base.Task:
  del split  # Procedural tasks: train and eval differ only in the rng seed.
  question, answer = sample_problem(rng)
  return {"question": question, "answer": answer, "kind": "calculator"}


SPEC = base.EnvSpec(CalculatorEnv, sample_task)
