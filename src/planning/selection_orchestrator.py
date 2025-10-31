# -*- coding: utf-8 -*-
"""
Phase 1 Orchestrator: Selection (Select Next)

Thin wrapper that connects Policy → Heuristics, and (optionally) writes a CSV
report of the sequencing via planning.Tracker.

- Reads feature order from Policy.select_order (e.g., ["ratio","profit","-weight"])
- Calls the pure heuristic select_next() to produce a deterministic item queue
- Optionally writes selection_queue.csv if a Tracker is provided
"""

from __future__ import annotations
from typing import List, Optional

from src.planning.state import SelectionState
from src.planning.policy import Policy
from src.heuristics.select_next.selector import select_next
from src.planning.tracker import Tracker


def run_selection_phase(
    state: SelectionState,
    policy: Policy,
    tracker: Optional[Tracker] = None,
) -> List[str]:
    """
    Execute Phase 1 to compute (and optionally log) the ordered list of item IDs.

    Parameters
    ----------
    state : SelectionState
        Immutable problem input (items + knapsack specs).
    policy : Policy
        Must provide `select_order` (tuple[str, ...] | None).
        If None/empty, the underlying heuristic applies its default:
          ["ratio", "profit", "-weight"] and appends "id" as final tiebreaker.
    tracker : Tracker | None
        If provided, writes selection_queue.csv into tracker.out_dir.

    Returns
    -------
    List[str]
        Ordered item IDs (first = next to place).
    """
    ordered_ids = select_next(state, order=policy.select_order)

    if tracker is not None:
        tracker.write_selection_queue_csv(state, ordered_ids)

    return ordered_ids


# Optional alias mirroring “queue” terminology (use whichever reads better upstream)
build_item_queue = run_selection_phase
