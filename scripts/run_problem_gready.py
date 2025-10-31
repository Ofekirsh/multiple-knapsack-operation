#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run greedy MKP pipeline (Phase 1 + Phase 2) on problems/problem_1.

Outputs under OUT_DIR:
  - items.csv             (Phase-1 ranking)
  - placement_log.csv     (step-by-step placements)
  - assignments.csv       (final per-item assignment)
  - per_knapsack.csv      (per-knapsack KPIs)
  - problem_summary.csv   (global KPIs)

Usage:
  python scripts/run_problem_1_greedy.py
"""

from __future__ import annotations
import os
from typing import List

# ====== CONFIGURATION ======
ITEMS_PATH = "../problems/problem_1/items.json"
KNAPS_PATH = "../problems/problem_1/knapsacks.json"
OUT_DIR    = "reports/problem_1"

# Phase-1 feature priority (left→right); allowed: "profit", "ratio", "weight", "-weight"
SELECT_ORDER = ["ratio", "profit", "-weight"]

# Phase-2 strategy: one of
#   "first_fit", "best_fit", "max_remaining", "smallest_fit",
#   "largest_fit", "balance_utilization", "best_fit_then_smallest"
PLACE_STRATEGY = "best_fit"

# Feasibility tolerance & optional seed
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

    # Tracker writes all CSV artifacts to OUT_DIR
    tracker = Tracker(out_dir=OUT_DIR)

    # Run end-to-end greedy solver
    solution: Solution = run_greedy(state=state, policy=policy, tracker=tracker)

    # Console summary
    print("\n=== Greedy Solve Complete (problem_1) ===")
    print(f"Total profit: {solution.total_profit:.2f}")
    print("Remaining capacities per knapsack:")
    for kid, rem in solution.remaining.items():
        print(f"  - {kid}: {rem:.4f}")

    # Point to outputs
    print("\nArtifacts written:")
    for fname in ["items.csv", "placement_log.csv", "assignments.csv", "per_knapsack.csv", "problem_summary.csv"]:
        print(f"  - {os.path.join(OUT_DIR, fname)}")


if __name__ == "__main__":
    main()
