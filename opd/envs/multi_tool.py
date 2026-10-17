"""Stages 3-5: all three tools at once; the policy decides what to call.

The task mix includes single-tool questions, questions that chain two tools,
and easy questions that need no tool at all. A good agent calls tools on the
first two and answers the last directly; the per-kind metrics logged during
evaluation show whether it does.
"""

import random
import sqlite3
from typing import Any

from opd.envs import base
from opd.envs import calculator
from opd.envs import json_env
from opd.envs import sqlite_env


class MultiToolEnv(
    calculator.CalculatorEnv, sqlite_env.SqliteEnv, json_env.JsonEnv
):
  """`calculator`, `query_sql` and `json_query`.

  No code needed: tools are inherited, and the cooperative `reset` chain
  (SqliteEnv -> JsonEnv -> Env) builds both the database and the document.
  """


def _sql_then_calculator(
    rng: random.Random, db: sqlite3.Connection, doc: dict[str, Any]
) -> tuple[str, str]:
  del doc
  department = rng.choice(sqlite_env.DEPARTMENTS)
  pct = rng.choice([3, 4, 5, 6, 8, 10, 12])
  (total,) = db.execute(
      "SELECT COALESCE(SUM(salary), 0) FROM employees WHERE department = ?",
      (department,),
  ).fetchone()
  question = (
      f"Every employee in the {department} department gets a {pct}% raise."
      " What will the department's total annual salary be after the raise?"
      " Round to the nearest integer."
  )
  return question, str((total * (100 + pct) + 50) // 100)


def _json_then_calculator(
    rng: random.Random, db: sqlite3.Connection, doc: dict[str, Any]
) -> tuple[str, str]:
  del db
  product = rng.choice(doc["products"])
  question = (
      f"If the store sold its entire stock of '{product['name']}' at the listed"
      " price, how many dollars of revenue would that bring in?"
  )
  return question, str(product["price"] * product["stock"])


def _sql_and_json(
    rng: random.Random, db: sqlite3.Connection, doc: dict[str, Any]
) -> tuple[str, str]:
  category = rng.choice(doc["products"])["category"]
  city = rng.choice(sqlite_env.CITIES)
  stock = sum(p["stock"] for p in doc["products"] if p["category"] == category)
  (employees,) = db.execute(
      "SELECT COUNT(*) FROM employees WHERE city = ?", (city,)
  ).fetchone()
  question = (
      f"Take the total stock of '{category}' products in the online store and"
      f" subtract the number of company employees based in {city}. What is the"
      " result?"
  )
  return question, str(stock - employees)


def _no_tool(
    rng: random.Random, db: sqlite3.Connection, doc: dict[str, Any]
) -> tuple[str, str]:
  del db, doc
  a, b = rng.randint(2, 12), rng.randint(2, 12)
  return rng.choice(
      (
          (f"What is {a} + {b}?", str(a + b)),
          (f"What is {a} times {b}?", str(a * b)),
          (f"How many days are there in {a} weeks?", str(7 * a)),
      )
  )


_KINDS = {
    # kind: (sampler, weight)
    "calculator": (lambda rng, db, doc: calculator.sample_problem(rng), 0.15),
    "sql": (lambda rng, db, doc: sqlite_env.sample_question(rng, db), 0.15),
    "json": (lambda rng, db, doc: json_env.sample_question(rng, doc), 0.15),
    "sql+calculator": (_sql_then_calculator, 0.15),
    "json+calculator": (_json_then_calculator, 0.15),
    "sql+json": (_sql_and_json, 0.1),
    "no_tool": (_no_tool, 0.15),
}


def sample_task(rng: random.Random, split: str) -> base.Task:
  del split  # Procedural tasks: train and eval differ only in the rng seed.
  db_seed, doc_seed = rng.randrange(2**31), rng.randrange(2**31)
  kinds = list(_KINDS)
  kind = rng.choices(kinds, weights=[_KINDS[k][1] for k in kinds])[0]
  sampler = _KINDS[kind][0]
  question, answer = sampler(
      rng, sqlite_env.build_database(db_seed), json_env.build_document(doc_seed)
  )
  return {
      "question": question,
      "answer": answer,
      "kind": kind,
      "db_seed": db_seed,
      "doc_seed": doc_seed,
  }


SPEC = base.EnvSpec(MultiToolEnv, sample_task)
