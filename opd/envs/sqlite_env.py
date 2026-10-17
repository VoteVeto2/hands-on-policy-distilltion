"""Stage 2: a read-only SQL tool over a small, randomly generated company DB.

Every task carries a `db_seed`; `reset` rebuilds the same in-memory database,
so a group of rollouts for one task all see identical data.
"""

import random
import sqlite3

from opd.envs import base

DEPARTMENTS = ("Engineering", "Sales", "Marketing", "Finance", "Support", "Research")
CITIES = ("Berlin", "Tokyo", "Austin", "Toronto", "Paris", "Sydney")
_STATUSES = ("active", "completed", "paused")
_FIRST_NAMES = (
    "Ada", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goro", "Hana", "Ivan",
    "Jun", "Kofi", "Lena", "Mateo", "Nia", "Omar", "Priya", "Quinn", "Rosa",
    "Sven", "Tara",
)  # fmt: skip
_LAST_NAMES = (
    "Abe", "Brooks", "Costa", "Dubois", "Evans", "Fischer", "Garcia", "Hale",
    "Ito", "Jensen", "Kim", "Lopez", "Moreau", "Novak", "Okafor", "Park",
    "Quist", "Rossi", "Silva", "Tanaka",
)  # fmt: skip
_PROJECT_NAMES = (
    "Atlas", "Beacon", "Comet", "Delta", "Ember", "Falcon", "Granite",
    "Harbor", "Ion", "Juniper", "Keystone", "Lumen", "Meridian", "Nimbus",
)  # fmt: skip
_NUM_EMPLOYEES = 40
_NUM_PROJECTS = 12
_MAX_ROWS = 20
# The progress handler runs every 1000 VM instructions; 1000 calls ~ 1M ops.
_MAX_PROGRESS_CALLS = 1000

# (question template, ground-truth SQL). Placeholders are sampled per task.
_QUESTIONS = (
    (
        "How many employees work in the {department} department?",
        "SELECT COUNT(*) FROM employees WHERE department = :department",
    ),
    (
        "What is the total annual salary of the {department} department?",
        "SELECT SUM(salary) FROM employees WHERE department = :department",
    ),
    (
        "What is the average salary of employees based in {city}, rounded to"
        " the nearest integer?",
        "SELECT CAST(ROUND(AVG(salary)) AS INTEGER) FROM employees"
        " WHERE city = :city",
    ),
    (
        "Who is the highest-paid employee in the {department} department?"
        " Answer with their full name.",
        "SELECT name FROM employees WHERE department = :department"
        " ORDER BY salary DESC LIMIT 1",
    ),
    (
        "How many employees based in {city} were hired after {year}?",
        "SELECT COUNT(*) FROM employees WHERE city = :city AND hire_year > :year",
    ),
    (
        "What is the total budget of the {status} projects run by the"
        " {department} department?",
        "SELECT COALESCE(SUM(budget), 0) FROM projects"
        " WHERE department = :department AND status = :status",
    ),
)


def build_database(seed: int) -> sqlite3.Connection:
  """Creates the in-memory database for `seed` and makes it read-only."""
  rng = random.Random(seed)
  db = sqlite3.connect(":memory:", check_same_thread=False)
  db.executescript("""
      CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT,
          department TEXT, city TEXT, salary INTEGER, hire_year INTEGER);
      CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT,
          department TEXT, budget INTEGER, status TEXT);
  """)
  names = rng.sample(
      [f"{f} {l}" for f in _FIRST_NAMES for l in _LAST_NAMES], _NUM_EMPLOYEES
  )
  # Distinct salaries make "highest-paid" questions unambiguous.
  salaries = rng.sample(range(40_000, 200_000, 250), _NUM_EMPLOYEES)
  for i, (name, salary) in enumerate(zip(names, salaries), start=1):
    db.execute(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)",
        (i, name, rng.choice(DEPARTMENTS), rng.choice(CITIES), salary,
         rng.randint(2012, 2025)),
    )  # fmt: skip
  projects = rng.sample(_PROJECT_NAMES, _NUM_PROJECTS)
  for i, project in enumerate(projects, start=1):
    db.execute(
        "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
        (i, f"Project {project}", rng.choice(DEPARTMENTS),
         rng.randrange(10_000, 500_000, 1_000), rng.choice(_STATUSES)),
    )  # fmt: skip
  db.commit()
  db.execute("PRAGMA query_only = ON")
  return db


def run_query(db: sqlite3.Connection, sql: str) -> str:
  """Executes one statement with a compute budget; formats rows as text."""
  calls = 0

  def over_budget() -> bool:
    nonlocal calls
    calls += 1
    return calls > _MAX_PROGRESS_CALLS

  db.set_progress_handler(over_budget, 1000)
  try:
    cursor = db.execute(sql)
    rows = cursor.fetchmany(_MAX_ROWS + 1)
  except sqlite3.OperationalError as e:
    if calls > _MAX_PROGRESS_CALLS:
      raise RuntimeError("query too expensive; filter or aggregate more") from e
    raise
  finally:
    db.set_progress_handler(None, 0)
  lines = [" | ".join(column[0] for column in cursor.description or ())]
  lines += [" | ".join(map(str, row)) for row in rows[:_MAX_ROWS]]
  if not rows:
    lines.append("(no rows)")
  elif len(rows) > _MAX_ROWS:
    lines.append(f"... (only the first {_MAX_ROWS} rows are shown)")
  return "\n".join(lines)


class SqliteEnv(base.Env):
  """Exposes `query_sql` over a per-task database."""

  def reset(self, **task) -> None:
    super().reset(**task)
    self._db = build_database(task["db_seed"])

  def query_sql(self, sql: str) -> str:
    """Runs a read-only SQL query on the company's SQLite database.

    The database has two tables: employees(id, name, department, city, salary,
    hire_year) and projects(id, name, department, budget, status).

    Args:
      sql: One SQLite SELECT statement.

    Returns:
      Up to 20 result rows as text, or an error message.
    """
    return run_query(self._db, sql)


def sample_question(rng: random.Random, db: sqlite3.Connection) -> tuple[str, str]:
  """Returns a (question, answer) pair whose answer exists in `db`."""
  while True:
    template, sql = rng.choice(_QUESTIONS)
    params = {
        "department": rng.choice(DEPARTMENTS),
        "city": rng.choice(CITIES),
        "status": rng.choice(_STATUSES),
        "year": rng.randint(2014, 2022),
    }
    (answer,) = db.execute(sql, params).fetchone() or (None,)
    if answer is not None:  # E.g. AVG over an empty city: try again.
      return template.format(**params), str(answer)


def sample_task(rng: random.Random, split: str) -> base.Task:
  del split  # Procedural tasks: train and eval differ only in the rng seed.
  db_seed = rng.randrange(2**31)
  question, answer = sample_question(rng, build_database(db_seed))
  return {"question": question, "answer": answer, "kind": "sql", "db_seed": db_seed}


SPEC = base.EnvSpec(SqliteEnv, sample_task)
