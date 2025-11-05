#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run iterative greedy MKP pipeline (Phase 1 + Phase 2 + resequencing hook)
on problems/problem_2.

Per-iteration outputs under OUT_DIR/iter_000, iter_001, ...
  - items.csv              (Phase-1 / resequenced ranking)
  - placement_log.csv      (step-by-step placements with diagnostics)
  - assignments.csv        (final per-item assignment)
  - per_knapsack.csv       (per-knapsack KPIs)
  - problem_summary.csv    (global KPIs)
  - iteration_summary.csv  (global KPIs + near-miss diagnostics)
  - per_knapsack_iter.csv  (per-bag KPIs incl. pain score)
  - per_item_iter.csv      (per-item signals for resequencing)
  - gap_histogram.csv      (leftover-capacity distribution)

Usage:
  python scripts/run_problem_2_greedy.py
"""

from __future__ import annotations
import os
from typing import List, Optional

# ====== CONFIGURATION ======
ITEMS_PATH = "../problems/problem_2/items.json"
KNAPS_PATH = "../problems/problem_2/knapsacks.json"
OUT_DIR    = "reports/problem_2"

# Phase-1 feature priority (left→right); allowed: "profit", "ratio", "weight", "-weight",
# "log-linear", "sweet-spot", "capacity-spot"
SELECT_ORDER = ["ratio", "profit", "-weight"]

# Phase-2 strategy (fixed across iterations):
#   "first_fit", "best_fit", "max_remaining", "smallest_fit",
#   "largest_fit", "balance_utilization", "best_fit_then_smallest"
PLACE_STRATEGY = "best_fit"

# Outer-loop controls
MAX_ITERS = 20       # hard cap
PATIENCE  = 5        # stop if no TP improvement for this many consecutive iters

# Feasibility tolerance & RNG seed
EPS  = 1e-12
SEED = 42
# ===========================

# Project imports
from src.utils.read_jsons import read_items_json, read_knapsacks_json
from src.business_objects.items import Item
from src.business_objects.knapsacks import KnapsackSpec
from src.planning import SelectionState, Policy, Solution
from src.planning.tracker import Tracker
from src.planning.solvers.greedy import run_greedy

# Optional: plug a resequencer here later. For now, keep None (fallback to Phase-1 each iter).
from src.planning.resequencing.resequencer_v1 import resequencer_v1 as ResequencerFn


def main() -> None:
    # Load problem definition
    items: List[Item] = read_items_json(ITEMS_PATH)
    knaps: List[KnapsackSpec] = read_knapsacks_json(KNAPS_PATH)

    # Immutable input snapshot
    state = SelectionState(items=items, knapsacks=knaps)

    # Policy (selection order + placement strategy)
    policy = Policy(
        select_order=tuple(SELECT_ORDER),
        place_strategy=PLACE_STRATEGY,
        eps=EPS,
        seed=SEED,
    )

    # We pass a tracker only to provide the z-score weights and a base out_dir.
    # The iterative solver will create a fresh per-iteration Tracker under OUT_DIR/iter_xxx.
    tracker = Tracker(out_dir=OUT_DIR)

    # Run iterative greedy solver
    solution: Solution = run_greedy(
        state=state,
        policy=policy,
        tracker=tracker,            # supplies weights + base folder
        out_dir=OUT_DIR,            # explicit base folder (mirrors tracker.out_dir)
        max_iters=MAX_ITERS,
        patience=PATIENCE,
        resequencer=ResequencerFn,  # keep None for now; plug in later
    )

    # Console summary (best solution found)
    print("\n=== Iterative Greedy Solve Complete (problem_2) ===")
    print(f"Best Total profit: {solution.total_profit:.2f}")
    print("Remaining capacities per knapsack:")
    for kid, rem in solution.remaining.items():
        print(f"  - {kid}: {rem:.4f}")

    # Point to outputs
    print("\nArtifacts written per iteration under:")
    print(f"  - {os.path.abspath(OUT_DIR)}")
    print("Each iter_* folder includes:")
    print("  items.csv, placement_log.csv, assignments.csv, per_knapsack.csv, problem_summary.csv")
    print("  iteration_summary.csv, per_knapsack_iter.csv, per_item_iter.csv, gap_histogram.csv")


if __name__ == "__main__":
    main()
