# -*- coding: utf-8 -*-
"""
Phase 1: Select Next (item sequencing).

You provide an `order` list defining priority left→right.

Direction rules (fixed):
  - profit -> descending (higher first)
  - ratio  -> descending (higher first)
  - weight -> descending by default
  - -weight -> ascending (lighter first)
  - id -> ascending (only used as final deterministic tiebreaker)

We always append 'id' at the end if missing.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Sequence, Tuple

from src.planning.state import SelectionState
from .features import build_feature_table

# Allowed keys for user order
_ALLOWED_KEYS = {"profit", "ratio", "weight", "-weight", "id"}


def _normalize_order(order: Sequence[str] | None) -> list[str]:
    """
    Normalize the user-provided order:
      - Default to ["ratio", "profit", "-weight"] if None/empty
      - Keep first occurrence only (deduplicate while preserving order)
      - Validate keys against the allowed set
      - Ensure 'id' is present as the final key
    """
    if not order:
        norm = ["ratio", "profit", "-weight"]
    else:
        seen: set[str] = set()
        norm: list[str] = []
        for raw in order:
            key = str(raw).strip()
            if key not in _ALLOWED_KEYS:
                raise ValueError(
                    f"Unknown order key '{key}'. "
                    f"Allowed: {sorted(_ALLOWED_KEYS)}"
                )
            if key not in seen:
                seen.add(key)
                norm.append(key)

    if "id" not in norm:
        norm.append("id")
    return norm


def _sort_key_for_item(
    item_id: str,
    features: Dict[str, float],
    order_keys: list[str],
) -> Tuple:
    """
    Build a Python sort key tuple following our direction rules.

    Implementation trick:
      - Python sorts ascending; for "descending" we negate numeric values.
      - 'id' is kept as string (ascending).
    """
    key: list = []
    for k in order_keys:
        if k == "id":
            key.append(item_id)  # ascending
            continue

        # profit and ratio: descending
        if k == "profit":
            val = features.get("profit", float("-inf"))
            key.append(-val)
            continue
        if k == "ratio":
            val = features.get("ratio", float("-inf"))
            key.append(-val)
            continue

        # weight variants
        if k == "weight":
            # descending (heavier first)
            val = features.get("weight", float("-inf"))
            key.append(-val)
            continue
        if k == "-weight":
            # ascending (lighter first)
            val = features.get("weight", float("inf"))
            key.append(val)
            continue

        # Should never reach here due to validation
        raise AssertionError(f"Unhandled order key: {k}")

    return tuple(key)


def select_next(
    state: SelectionState,
    order: Sequence[str] | None = None,
) -> list[str]:
    """
    Return an ordered list of item IDs according to `order`.

    Parameters
    ----------
    state : SelectionState
        Immutable snapshot with items & knapsack specs.
    order : Sequence[str] | None
        Feature priority list. Examples:
          ["ratio", "profit", "-weight"]
          ["profit", "-weight"]
          ["-weight"]
        If None/empty, defaults to ["ratio", "profit", "-weight"].

    Notes
    -----
    - No mutation occurs here.
    - Sorting is deterministic; 'id' is always the final tiebreaker (ascending).
    """
    order_keys = _normalize_order(order)
    feat_table = build_feature_table(state.items)

    sortable = []
    for it in state.items:
        row = feat_table[it.id]
        sortable.append((it.id, _sort_key_for_item(it.id, row, order_keys)))

    sortable.sort(key=lambda t: t[1])
    return [iid for iid, _ in sortable]
