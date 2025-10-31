#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Phase-1 (selection) on problems/problem_1 and export the sequencing CSV.

This version does NOT use argparse.
Just set the variables at the top of the file and run:

    python scripts/run_problem_1_selection.py
"""

from __future__ import annotations
import os
from typing import List

# ====== CONFIGURATION ======
ITEMS_PATH = "../problems/problem_1/items.json"
KNAPS_PATH = "../problems/problem_1/knapsacks.json"
OUT_DIR = "reports/problem_1"

# Feature priority list (left→right)
# Allowed: "profit", "ratio", "weight", "-weight"
SELECT_ORDER = ["ratio", "profit", "-weight"]
# ============================

# Business objects & planning layer
from src.business_objects.items import Item
from src.business_objects.knapsacks import KnapsackSpec
from src.planning import SelectionState, Policy
from src.planning.selection_orchestrator import run_selection_phase
from src.planning.tracker import Tracker
from src.utils.read_jsons import read_items_json, read_knapsacks_json


def main() -> None:
    # Load problem
    items: List[Item] = read_items_json(ITEMS_PATH)
    knaps: List[KnapsackSpec] = read_knapsacks_json(KNAPS_PATH)

    # Build immutable SelectionState
    state = SelectionState(items=items, knapsacks=knaps)

    # Build Policy (convert list → tuple)
    policy = Policy(select_order=tuple(SELECT_ORDER))

    # Set up output tracker
    tracker = Tracker(out_dir=OUT_DIR)

    # Run Phase-1 (also writes selection_queue.csv via tracker)
    ordered_ids: List[str] = run_selection_phase(state, policy, tracker=tracker)

    # Print the resulting order
    print("\n=== Phase-1 Item Order (first → next to place) ===")
    print(ordered_ids)

    print(f"\nSelection CSV written to: {os.path.join(OUT_DIR, 'selection_queue.csv')}")


if __name__ == "__main__":
    main()
