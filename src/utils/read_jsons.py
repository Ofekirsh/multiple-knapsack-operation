# -*- coding: utf-8 -*-
"""
I/O helpers for loading MKP problem definitions.

This module includes lightweight JSON readers that match the
`problems/problem_*/{items.json, knapsacks.json}` structure.

JSON formats:
- items.json      : [{"id": "...", "profit": <number>, "weight": <number>}, ...]
- knapsacks.json  : [{"id": "...", "capacity": <number>}, ...]

These map directly to:
- business_objects.items.Item
- business_objects.knapsacks.KnapsackSpec
"""

from __future__ import annotations
import json
from typing import List

from src.business_objects.items import Item
from src.business_objects.knapsacks import KnapsackSpec


class SchemaError(ValueError):
    """Raised when a JSON file violates the expected schema."""


def _require(obj: dict, key: str, path: str) -> object:
    if key not in obj:
        raise SchemaError(f"{path}: missing required key '{key}' in object {obj}")
    return obj[key]


def read_items_json(path: str) -> List[Item]:
    """
    Load items from a JSON array. Each element must have:
      - id (str)
      - profit (number)
      - weight (number)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise SchemaError(f"{path}: failed to read/parse JSON: {e}") from e

    if not isinstance(data, list):
        raise SchemaError(f"{path}: expected a JSON array.")

    items: List[Item] = []
    for idx, obj in enumerate(data, start=1):
        if not isinstance(obj, dict):
            raise SchemaError(f"{path}[{idx}]: expected an object.")
        try:
            iid = str(_require(obj, "id", path))
            profit = float(_require(obj, "profit", path))
            weight = float(_require(obj, "weight", path))
        except Exception as e:
            raise SchemaError(f"{path}[{idx}]: {e}") from e
        items.append(Item(id=iid, profit=profit, weight=weight))
    return items


def read_knapsacks_json(path: str) -> List[KnapsackSpec]:
    """
    Load knapsacks from a JSON array. Each element must have:
      - id (str)
      - capacity (number)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise SchemaError(f"{path}: failed to read/parse JSON: {e}") from e

    if not isinstance(data, list):
        raise SchemaError(f"{path}: expected a JSON array.")

    knaps: List[KnapsackSpec] = []
    for idx, obj in enumerate(data, start=1):
        if not isinstance(obj, dict):
            raise SchemaError(f"{path}[{idx}]: expected an object.")
        try:
            kid = str(_require(obj, "id", path))
            capacity = float(_require(obj, "capacity", path))
        except Exception as e:
            raise SchemaError(f"{path}[{idx}]: {e}") from e
        knaps.append(KnapsackSpec(id=kid, capacity=capacity))
    return knaps
