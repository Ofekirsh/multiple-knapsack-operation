# -*- coding: utf-8 -*-
"""
Greedy end-to-end solver (Phase 1 + Phase 2).

Pipeline:
  1) Phase 1 (selection): build item queue (and write items.csv via Tracker).
  2) Phase 2 (placement): iterate queue, place items greedily (and append placement_log.csv).
  3) Build Solution and write final CSVs:
       - assignments.csv      (per-item final assignment)
       - per_knapsack.csv     (per-knapsack KPIs)
       - problem_summary.csv  (global KPIs)
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from src.business_objects.items import Item
from src.planning import SelectionState, RuntimeState, Policy, Solution, PlacementDecision
from src.planning.selection_orchestrator import run_selection_phase
from src.planning.place_orchestrator import place_next
from src.planning.tracker import Tracker


def run_greedy(
    state: SelectionState,
    policy: Policy,
    tracker: Tracker | None = None,
) -> Solution:
    """
    Run selection → placement greedily and return the final Solution.
    If a Tracker is provided, writes items.csv, placement_log.csv, assignments.csv,
    per_knapsack.csv, and problem_summary.csv under tracker.out_dir.
    """
    # Phase 1: selection (items.csv written by orchestrator if tracker provided)
    ordered_ids: List[str] = run_selection_phase(state, policy, tracker=tracker)

    # Fresh mutable runtime (remaining capacities)
    rt: RuntimeState = state.to_runtime()

    # Quick lookup for items
    items_by_id: Dict[str, Item] = {it.id: it for it in rt.items}

    # Phase 2: iterate and place
    decisions: List[PlacementDecision] = []
    for iid in ordered_ids:
        it = items_by_id[iid]
        dec = place_next(item=it, state=rt, policy=policy, tracker=tracker)
        decisions.append(dec)

    # Build Solution
    assignments: Dict[str, str | None] = {d.item_id: d.knapsack_id for d in decisions}
    total_profit: float = sum(items_by_id[i].profit for i, k in assignments.items() if k is not None)
    remaining: Dict[str, float] = {k.id: float(k.remaining) for k in rt.knapsacks}
    sol = Solution(assignments=assignments, total_profit=total_profit, remaining=remaining)

    # Final artifacts
    if tracker is not None:
        tracker.write_assignments_csv(state=state, decisions=decisions)
        tracker.write_per_knapsack_csv(state=state, solution=sol)
        tracker.write_problem_summary_csv(state=state, solution=sol)
        tracker.write_fairness_items_csv(state=state, solution=sol)

    return sol
