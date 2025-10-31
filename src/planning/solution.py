# -*- coding: utf-8 -*-
"""
Solution and decision models for MKP planning results.

These data classes define the shape of outputs produced by Phase 2
and consumed by the metrics/reporting layers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PlacementDecision:
    """
    Outcome of placing a single item (or failing to place).

    Attributes
    ----------
    item_id : str
        The item acted upon.
    knapsack_id : str | None
        The knapsack chosen, or None if no feasible knapsack was found.
    reason : str | None
        Optional short reason/label (e.g., "placed_best_fit", "no_feasible_knapsack").
    """
    item_id: str
    knapsack_id: Optional[str]
    reason: Optional[str] = None


@dataclass(frozen=True)
class Solution:
    """
    Aggregated placement results for a full run.

    Attributes
    ----------
    assignments : dict[item_id, knapsack_id | None]
        Final mapping of items to knapsacks (None means not placed).
    total_profit : float
        Sum of profits for all placed items.
    remaining : dict[knapsack_id, remaining_capacity]
        Post-run remaining capacity per knapsack.
    """
    assignments: Dict[str, Optional[str]]
    total_profit: float
    remaining: Dict[str, float]
