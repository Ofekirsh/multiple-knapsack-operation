# -*- coding: utf-8 -*-
"""
Derived item features for Phase 1 sequencing.

This module computes per-item features used by ordering schemes.
It is intentionally pure/stateless and performs no mutation or I/O.
"""

from __future__ import annotations
from typing import Dict
from src.business_objects.items import Item


def compute_item_features(item: Item) -> Dict[str, float]:
    """
    Compute derived features for a single item.

    Features:
      - profit: raw profit
      - weight: raw weight
      - ratio:  profit/weight; convention:
                if weight == 0:
                   - profit > 0  -> +inf   (extremely desirable)
                   - profit == 0 -> 0.0
                   - profit < 0  -> -inf   (shouldn't happen under our BO validation)
    """
    p = float(item.profit)
    w = float(item.weight)

    if w == 0.0:
        if p > 0.0:
            ratio = float("inf")
        elif p == 0.0:
            ratio = 0.0
        else:
            ratio = float("-inf")
    else:
        ratio = p / w

    return {
        "profit": p,
        "weight": w,
        "ratio": ratio,
    }


def build_feature_table(items: list[Item]) -> dict[str, dict[str, float]]:
    """
    Vectorized convenience: return {item_id: {feature_name: value, ...}, ...}
    """
    return {it.id: compute_item_features(it) for it in items}
