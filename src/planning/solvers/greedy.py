# -*- coding: utf-8 -*-
"""
Greedy end-to-end solver with iterative resequencing (Phase 1 + Phase 2).

New behavior:
  - Outer loop with (max_iters, patience) early-stopping on Total Profit.
  - Per-iteration subfolders: OUT_DIR/iter_000, iter_001, ...
  - Optional resequencer hook to build the next iteration's item queue
    from the current solution + placement log.
  - Writes iteration-level metrics via Tracker.write_iteration_metrics(...).

Pipeline per iteration i:
  1) Build item queue:
        - i == 0: run_selection_phase(state, policy, tracker_i)  (your existing Phase-1)
        - i > 0 : if `resequencer` provided → use it; else fallback to run_selection_phase
  2) Phase 2 (placement): iterate queue with place_next (policy.place_strategy)
        - placement_log.csv enriched with remaining_before, closest_fit_margin, feasible_knaps_count
  3) Build Solution and write artifacts (assignments, per_knapsack, problem_summary, fairness)
  4) Write iteration-level metrics (iteration_summary, per_knapsack_iter, per_item_iter, gap_histogram)
  5) Update best-so-far solution; apply patience rule

Return:
  - The best Solution observed across all iterations.

This file assumes:
  - Tracker has append_placement_decision( ..., remaining_before, remaining_after )
  - Tracker has write_iteration_metrics(...)
"""

from __future__ import annotations
import os
import csv
from typing import Dict, List, Tuple, Optional, Callable, Any

from src.business_objects.items import Item
from src.planning import (
    SelectionState,
    RuntimeState,
    Policy,
    Solution,
    PlacementDecision,
)
from src.planning.selection_orchestrator import run_selection_phase
from src.planning.place_orchestrator import place_next
from src.planning.tracker import Tracker


# Type signature for a resequencer hook:
#   Inputs:
#     - state: immutable SelectionState (items + knapsacks)
#     - policy: current Policy (you may read select_order/place_strategy)
#     - solution: the Solution from the previous iteration
#     - placement_rows: parsed rows from placement_log.csv of the previous iteration
#   Output:
#     - ordered list of item_ids for the NEXT iteration
ResequencerFn = Callable[
    [SelectionState, Policy, Solution, List[Dict[str, Any]]],
    List[str],
]


def _read_placement_rows(log_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if os.path.exists(log_path):
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    return rows


def run_greedy(
    state: SelectionState,
    policy: Policy,
    tracker: Tracker | None = None,  # kept for backward-compat (unused in iterative mode)
    *,
    out_dir: Optional[str] = None,     # override output root; else uses tracker.out_dir if provided
    max_iters: int = 20,
    patience: int = 5,
    resequencer: Optional[ResequencerFn] = None,
) -> Solution:
    """
    Run iterative greedy with resequencing and early stopping.

    Backward-compatible behavior:
      - If max_iters == 1, patience <= 0, and no resequencer: behaves like your original single-pass.
      - If `tracker` is provided but `out_dir` is None, we use tracker.out_dir as the base folder.

    Returns
    -------
    Solution
        Best solution found across iterations.
    """
    if max_iters < 1:
        max_iters = 1
    if patience < 0:
        patience = 0

    # Determine base output directory
    if out_dir is None:
        if tracker is not None:
            out_dir = tracker.out_dir
        else:
            out_dir = "reports/iter_solver"
    os.makedirs(out_dir, exist_ok=True)

    # Book-keeping for best-so-far
    best_solution: Optional[Solution] = None
    best_profit: float = float("-inf")
    best_iter: int = -1
    no_improve: int = 0

    # Initial queue (iter 0) is produced by your existing Phase-1 selection
    # We re-run Phase-1 each iteration unless a resequencer is provided for i>0
    prev_solution: Optional[Solution] = None
    prev_placement_rows: List[Dict[str, Any]] = []

    for i in range(max_iters):
        iter_dir = os.path.join(out_dir, f"iter_{i:03d}")
        os.makedirs(iter_dir, exist_ok=True)

        # Fresh per-iteration tracker (resets step index and writes into iter subfolder)
        t = Tracker(out_dir=iter_dir)

        # ---- Phase 1: build queue ----
        if i == 0 or resequencer is None or prev_solution is None:
            # Default Phase-1 ordering (ratio, profit, -weight) or whatever policy.select_order is
            ordered_ids: List[str] = run_selection_phase(state, policy, tracker=t)
        else:
            # Use resequencer hook to build the next queue from prev iteration signals
            ordered_ids = resequencer(state, policy, prev_solution, prev_placement_rows)
            # Persist the resequenced queue as items.csv for transparency
            # Reuse Tracker's writer
            t.write_selection_queue_csv(state=state, ordered_item_ids=ordered_ids, filename="items.csv")

        # ---- Phase 2: placement with fixed place_next heuristic ----
        rt: RuntimeState = state.to_runtime()
        items_by_id: Dict[str, Item] = {it.id: it for it in rt.items}

        decisions: List[PlacementDecision] = []
        for iid in ordered_ids:
            it = items_by_id[iid]
            dec = place_next(item=it, state=rt, policy=policy, tracker=t)  # tracker logs remaining_before/after
            decisions.append(dec)

        # ---- Build Solution ----
        assignments: Dict[str, str | None] = {d.item_id: d.knapsack_id for d in decisions}
        total_profit: float = sum(items_by_id[iid].profit for iid, kid in assignments.items() if kid is not None)
        remaining: Dict[str, float] = {k.id: float(k.remaining) for k in rt.knapsacks}
        sol = Solution(assignments=assignments, total_profit=total_profit, remaining=remaining)

        # ---- Write per-iteration artifacts ----
        t.write_assignments_csv(state=state, decisions=decisions)
        t.write_per_knapsack_csv(state=state, solution=sol)
        t.write_problem_summary_csv(state=state, solution=sol)
        t.write_fairness_items_csv(state=state, solution=sol)
        t.write_iteration_metrics(state=state, solution=sol, placement_log_path=t._placement_log_path, iter_index=i)

        # ---- Early stopping / best-so-far ----
        improved = total_profit > best_profit
        if improved:
            best_profit = total_profit
            best_solution = sol
            best_iter = i
            no_improve = 0
        else:
            no_improve += 1

        # Prepare inputs for next iteration (resequencer)
        prev_solution = sol
        prev_placement_rows = _read_placement_rows(t._placement_log_path)

        # Stop on patience
        if no_improve >= patience:
            break

    # Return best found (or last if none set for some reason)
    if best_solution is None:
        # This only happens if max_iters==0 (guarded) or something very odd;
        # fall back to last built solution.
        best_solution = prev_solution
    return best_solution  # type: ignore[return-value]
