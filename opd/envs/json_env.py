"""Stage 2: a JMESPath query tool over a randomly generated store document.

The document is too large to paste into the prompt, so the policy has to query
it. JMESPath (https://jmespath.org) is the query language of the AWS CLI; the
teacher has seen plenty of it.
"""

import json
import random
from typing import Any

import jmespath

from opd.envs import base

_ADJECTIVES = (
    "Oak",
    "Steel",
    "Linen",
    "Copper",
    "Maple",
    "Glass",
    "Velvet",
    "Bamboo",
)
_NOUNS = ("Desk", "Lamp", "Chair", "Kettle", "Shelf", "Planter", "Rug", "Clock")
_CATEGORIES = ("furniture", "lighting", "kitchen", "garden", "decor")
_TAGS = ("eco", "bestseller", "new", "sale", "handmade", "premium")
_COUNTRIES = ("Germany", "Japan", "Brazil", "Canada", "Kenya", "Vietnam")
_SUPPLIERS = ("Northwind", "Acme", "Globex", "Initech", "Umbrella", "Stark")
_CUSTOMERS = (
    "Avery",
    "Blake",
    "Casey",
    "Devon",
    "Emery",
    "Finley",
    "Harper",
    "Jordan",
    "Kai",
    "Logan",
    "Morgan",
    "Rowan",
)
_CITIES = ("Lisbon", "Seoul", "Denver", "Lagos", "Oslo", "Lima")
_NUM_PRODUCTS = 16
_NUM_CUSTOMERS = 10
_NUM_ORDERS = 20
_MAX_CHARS = 1500


def build_document(seed: int) -> dict[str, Any]:
  """Creates the store document for `seed`."""
  rng = random.Random(seed)
  suppliers = [
      {"name": name, "country": rng.choice(_COUNTRIES)} for name in _SUPPLIERS
  ]
  names = rng.sample(
      [f"{a} {n}" for a in _ADJECTIVES for n in _NOUNS], _NUM_PRODUCTS
  )
  products = [
      {
          "id": f"P{i:02d}",
          "name": name,
          "category": rng.choice(_CATEGORIES),
          "price": rng.randint(5, 900),
          "stock": rng.randint(0, 400),
          "supplier": dict(rng.choice(suppliers)),
          "tags": rng.sample(_TAGS, rng.randint(1, 3)),
      }
      for i, name in enumerate(names, start=1)
  ]
  customers = [
      {
          "id": f"C{i:02d}",
          "name": name,
          "city": rng.choice(_CITIES),
          "tier": rng.choice(("gold", "silver", "basic")),
      }
      for i, name in enumerate(rng.sample(_CUSTOMERS, _NUM_CUSTOMERS), start=1)
  ]
  orders = [
      {
          "id": f"O{i:03d}",
          "customer_id": rng.choice(customers)["id"],
          "status": rng.choice(("shipped", "pending", "returned")),
          "items": [
              {"product_id": p["id"], "quantity": rng.randint(1, 5)}
              for p in rng.sample(products, rng.randint(1, 3))
          ],
      }
      for i in range(1, _NUM_ORDERS + 1)
  ]
  return {"products": products, "customers": customers, "orders": orders}


class JsonEnv(base.Env):
  """Exposes `json_query` over a per-task store document."""

  def reset(self, **task) -> None:
    super().reset(**task)
    self._doc = build_document(task["doc_seed"])

  def json_query(self, expression: str) -> str:
    """Queries the online store's JSON document with a JMESPath expression.

    Top-level keys: products (id, name, category, price, stock, supplier
    {name, country}, tags), customers (id, name, city, tier) and orders (id,
    customer_id, status, items [{product_id, quantity}]).

    Args:
      expression: A JMESPath expression, for example
        "products[?name=='Oak Desk'].price | [0]".

    Returns:
      The JSON-encoded result (truncated when long), or an error message.
    """
    text = json.dumps(jmespath.search(expression, self._doc))
    if len(text) > _MAX_CHARS:
      return text[:_MAX_CHARS] + " ... (truncated)"
    return text


def sample_question(rng: random.Random, doc: dict[str, Any]) -> tuple[str, str]:
  """Returns a (question, answer) pair about `doc`."""
  products, orders = doc["products"], doc["orders"]
  product, order = rng.choice(products), rng.choice(orders)
  category = rng.choice(products)["category"]
  country = rng.choice(products)["supplier"]["country"]
  tag = rng.choice(rng.choice(products)["tags"])
  customer = next(
      c for c in doc["customers"] if c["id"] == order["customer_id"]
  )
  choices = (
      (
          f"What is the price of the product named '{product['name']}'?",
          product["price"],
      ),
      (
          f"How many products are in the '{category}' category?",
          sum(p["category"] == category for p in products),
      ),
      (
          "What is the total stock of all products whose supplier is based"
          f" in {country}?",
          sum(
              p["stock"]
              for p in products
              if p["supplier"]["country"] == country
          ),
      ),
      (
          f"What is the name of the customer who placed order {order['id']}?",
          customer["name"],
      ),
      (
          f"How many units in total were ordered in order {order['id']}?",
          sum(item["quantity"] for item in order["items"]),
      ),
      (
          f"How many products have the tag '{tag}'?",
          sum(tag in p["tags"] for p in products),
      ),
  )
  question, answer = rng.choice(choices)
  return question, str(answer)


def sample_task(rng: random.Random, split: str) -> base.Task:
  del split  # Procedural tasks: train and eval differ only in the rng seed.
  doc_seed = rng.randrange(2**31)
  question, answer = sample_question(rng, build_document(doc_seed))
  return {
      "question": question,
      "answer": answer,
      "kind": "json",
      "doc_seed": doc_seed,
  }


SPEC = base.EnvSpec(JsonEnv, sample_task)
