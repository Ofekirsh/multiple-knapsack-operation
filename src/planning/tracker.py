# -*- coding: utf-8 -*-
"""
Planning tracker: CSV artifacts for Phase 1 (selection) and Phase 2 (placement).

Files produced (when Tracker is used):
  - items.csv              (Phase-1 ranking; written by write_selection_queue_csv)
  - placement_log.csv      (append-as-you-go during Phase-2)
  - assignments.csv        (final per-item assignment snapshot)
  - per_knapsack.csv       (per-knapsack KPIs)
  - problem_summary.csv    (global KPIs)

Notes
-----
- Column choices are MKP-appropriate (no zone/lane/layer concepts).
- Callers decide when to invoke these writers; greedy solver calls them at the end.
"""

from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

from src.planning import SelectionState, PlacementDecision, Solution
from src.business_objects.items import Item

from src.quality_metrics.fairness import compute_per_item_fairness
from src.quality_metrics.core import (
    compute_global_metrics,
    build_gap_histogram,
    count_near_misses,
    per_knapsack_metrics,
    per_item_signals,
)

@dataclass
class Tracker:
    """
    Thin, opt-in artifact writer. Callers control when/where to dump.
    """
    out_dir: str
    _placement_log_path: str = field(init=False, repr=False)
    _placement_started: bool = field(default=False, init=False, repr=False)
    _placement_step_idx: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:  # type: ignore[override]
        os.makedirs(self.out_dir, exist_ok=True)
        self._placement_log_path = os.path.join(self.out_dir, "placement_log.csv")

    # -----------------------------
    # Phase 1: selection queue CSV
    # -----------------------------
    def write_selection_queue_csv(
        self,
        state: SelectionState,
        ordered_item_ids: List[str],
        filename: str = "items.csv",
    ) -> str:
        """
        Persist the Phase-1 item ordering to CSV.

        Columns:
          order_index, item_id, profit, weight, ratio
        """
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, filename)

        items_by_id = {it.id: it for it in state.items}

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["order_index", "item_id", "profit", "weight", "ratio"])
            for idx, iid in enumerate(ordered_item_ids):
                it = items_by_id[iid]
                p = float(it.profit)
                wgt = float(it.weight)
                if wgt == 0.0:
                    ratio = float("inf") if p > 0.0 else 0.0
                else:
                    ratio = p / wgt
                w.writerow([idx, iid, p, wgt, ratio])

        return path

    # -----------------------------
    # Phase 2: placement log CSV
    # -----------------------------
    def _ensure_placement_header(self) -> None:
        if self._placement_started:
            return
        with open(self._placement_log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "step_index",
                "item_id",
                "knapsack_id",
                "profit",
                "weight",
                "reason",
                "remaining_after_json",
            ])
        self._placement_started = True

    def append_placement_decision(
            self,
            item: Item,
            decision: PlacementDecision,
            remaining_before: Dict[str, float],
            remaining_after: Dict[str, float],
    ) -> str:
        """
        Append a single placement decision row.

        New columns:
          - remaining_before_json: snapshot of all knapsacks before placing this item
          - closest_fit_margin: min non-negative (remaining_before[k] - item.weight) across all knapsacks;
                                if none feasible, use the least-negative margin (max of negatives).
          - feasible_knaps_count: number of knapsacks with remaining_before[k] >= item.weight
        """
        # Ensure header exists
        if not self._placement_started:
            with open(self._placement_log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "step_index",
                    "item_id",
                    "knapsack_id",
                    "profit",
                    "weight",
                    "reason",
                    "remaining_before_json",  # NEW
                    "remaining_after_json",
                    "closest_fit_margin",  # NEW
                    "feasible_knaps_count",  # NEW
                ])
            self._placement_started = True

        # Compute margins across ALL knapsacks (based on 'before' snapshot)
        wj = float(item.weight)
        margins = [(float(rem_before) - wj) for rem_before in remaining_before.values()]

        # Count feasible knapsacks (non-negative margin)
        feasible_knaps_count = sum(1 for m in margins if m >= 0.0)

        # Tightest feasible margin if any; else, least-negative (closest miss)
        if feasible_knaps_count > 0:
            closest_fit_margin = min(m for m in margins if m >= 0.0)
        else:
            closest_fit_margin = max(margins) if margins else None  # None only if no knaps exist

        # Write row
        with open(self._placement_log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                self._placement_step_idx,
                item.id,
                "" if decision.knapsack_id is None else decision.knapsack_id,
                float(item.profit),
                wj,
                decision.reason or "",
                json.dumps(remaining_before, ensure_ascii=False),  # NEW
                json.dumps(remaining_after, ensure_ascii=False),
                "" if closest_fit_margin is None else round(float(closest_fit_margin), 5),  # NEW
                feasible_knaps_count,  # NEW
            ])

        self._placement_step_idx += 1
        return self._placement_log_path

    # -----------------------------
    # Final artifacts after solve
    # -----------------------------
    def write_assignments_csv(
        self,
        state: SelectionState,
        decisions: List[PlacementDecision],
        filename: str = "assignments.csv",
    ) -> str:
        """
        Final per-item assignment snapshot.

        Columns:
          item_id, knapsack_id, profit, weight, placed (0/1), reason
        """
        path = os.path.join(self.out_dir, filename)
        items_by_id = {it.id: it for it in state.items}

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "knapsack_id", "profit", "weight", "placed", "reason"])
            for dec in decisions:
                it = items_by_id[dec.item_id]
                placed = 1 if dec.knapsack_id is not None else 0
                w.writerow([
                    dec.item_id,
                    "" if dec.knapsack_id is None else dec.knapsack_id,
                    float(it.profit),
                    float(it.weight),
                    placed,
                    dec.reason or "",
                ])
        return path

    def write_per_knapsack_csv(
        self,
        state: SelectionState,
        solution: Solution,
        filename: str = "per_knapsack.csv",
    ) -> str:
        """
        Per-knapsack KPIs.

        Columns:
          knapsack_id, capacity, used, remaining, utilization_pct, items_placed, profit_sum
        """
        path = os.path.join(self.out_dir, filename)

        # Capacity lookup
        cap_by_id: Dict[str, float] = {k.id: float(k.capacity) for k in state.knapsacks}

        # Aggregate per knapsack
        profit_by_knap: Dict[str, float] = {k: 0.0 for k in cap_by_id}
        count_by_knap: Dict[str, int] = {k: 0 for k in cap_by_id}
        for item_id, knap_id in solution.assignments.items():
            if knap_id is None:
                continue
            # Profit lookup requires items; build it once
            # (We don’t have items directly here; create fast map)
            # Safer: pass in state for item lookup
            # Build map:
        items_by_id = {it.id: it for it in state.items}
        for item_id, knap_id in solution.assignments.items():
            if knap_id is None:
                continue
            profit_by_knap[knap_id] += float(items_by_id[item_id].profit)
            count_by_knap[knap_id] += 1

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "knapsack_id",
                "capacity",
                "used",
                "remaining",
                "utilization_pct",
                "items_placed",
                "profit_sum",
            ])
            for kid, cap in cap_by_id.items():
                rem = float(solution.remaining.get(kid, 0.0))
                used = max(0.0, cap - rem)
                util = 0.0 if cap == 0.0 else (used / cap) * 100.0
                w.writerow([
                    kid,
                    cap,
                    used,
                    rem,
                    util,
                    int(count_by_knap.get(kid, 0)),
                    float(profit_by_knap.get(kid, 0.0)),
                ])
        return path

    def write_problem_summary_csv(
            self,
            state: SelectionState,
            solution: Solution,
            filename: str = "problem_summary.csv",
    ) -> str:
        """
        Global KPIs across all knapsacks with weighted z_score.

        Args:
            weights_of_z_score: List of 4 weights [w_TP, w_OU, w_SR, w_PWE] for calculating weighted z_score

        Columns (in order):
          1. z_score: Weighted combination of quality metrics
          2. Problem features: total_items, num_knapsacks, capacity_sum, total_possible_profit
          3. Quality metrics: TP, OU, SR, PWE, APK, BL
          4. Other details: placed_items, skipped_items, used_sum, remaining_sum
        """
        import os
        import csv
        import math

        path = os.path.join(self.out_dir, filename)

        # Base counts
        total_items = len(state.items)
        placed_items = sum(1 for _, k in solution.assignments.items() if k is not None)
        skipped_items = total_items - placed_items

        # Problem features
        num_knapsacks = len(state.knapsacks)
        capacity_sum = sum(float(k.capacity) for k in state.knapsacks)
        total_possible_profit = sum(float(it.profit) for it in state.items)

        # Capacity aggregates
        rem_sum = sum(float(r) for r in solution.remaining.values())
        used_sum = max(0.0, capacity_sum - rem_sum)

        # Quality Metrics
        # TP: Total Profit
        TP = float(solution.total_profit)

        # OU: Overall Utilization (percent)
        OU = 0.0 if capacity_sum == 0.0 else (used_sum / capacity_sum) * 100.0

        # SR: Selection Rate (percent)
        SR = 0.0 if total_items == 0 else (placed_items / total_items) * 100.0

        # PWE: Profit-Weight Efficiency
        items_by_id = {it.id: it for it in state.items}
        total_weight_used = sum(
            float(items_by_id[item_id].weight)
            for item_id, knap_id in solution.assignments.items()
            if knap_id is not None
        )
        PWE = 0.0 if total_weight_used == 0.0 else TP / total_weight_used

        # APK: Avg Profit per Knapsack
        APK = 0.0 if num_knapsacks == 0 else TP / num_knapsacks

        # BL: Balance Load (Std Dev of knapsack loads)
        loads = [
            float(k.capacity) - float(solution.remaining.get(k.id, 0.0))
            for k in state.knapsacks
        ]
        if loads:
            mean_load = sum(loads) / len(loads)
            BL = math.sqrt(sum((x - mean_load) ** 2 for x in loads) / len(loads))
        else:
            BL = 0.0


        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Header row
            w.writerow([
                # 2. Problem Features
                "Total Items",
                "Num Knapsacks",
                "Capacity Sum",
                "Total Possible Profit",
                # 3. Quality Metrics
                "Total Profit",
                "OU",
                "% Placed",
                "PWE",
                "APK",
                "BL",
                # 4. Other Details
                "Placed Items",
                "# Left Out items",
                "Used Sum",
                "Remaining Sum",
            ])
            # Data row
            w.writerow([
                # 2. Problem Features
                total_items,
                num_knapsacks,
                f"{capacity_sum:.3f}",
                f"{total_possible_profit:.3f}",
                # 3. Quality Metrics
                f"{TP:.3f}",
                f"{OU:.3f}",
                f"{SR:.3f}",
                f"{PWE:.3f}",
                f"{APK:.3f}",
                f"{BL:.3f}",
                # 4. Other Details
                placed_items,
                skipped_items,
                f"{used_sum:.3f}",
                f"{rem_sum:.3f}",
            ])
        return path

    def write_fairness_items_csv(
            self,
            state: SelectionState,
            solution: Solution,
            filename: str = "fairness_items.csv",
    ) -> str:
        """
        Write fairness_items.csv with the subset-based fairness model.

        Columns:
          item_id, placed, merit, s2_count, s3_sum, s3_max, s3_avg, subsets_json, gains_json

        Notes:
          - 'placed' is 1 if the item is assigned, else 0.
          - 'merit' = p_j / w_j, using +inf if w_j==0 and p_j>0, else 0.
          - subsets_json: list of subsets ONLY for unpacked items (packed = [])
          - gains_json: aligned gains per subset (packed = [])
          - Numeric values formatted to 3 decimals via _fmt()
        """

        import os
        import csv
        import json
        from src.quality_metrics.fairness import compute_per_item_fairness

        path = os.path.join(self.out_dir, filename)

        # Helper for merit
        def merit_of(p: float, w: float) -> float:
            if w == 0.0:
                return float("inf") if p > 0.0 else 0.0
            return float(p) / float(w)

        items_by_id = {it.id: it for it in state.items}

        # Compute fairness once
        fair = compute_per_item_fairness(state, solution)

        # Stable ordering by item_id
        item_ids = sorted(items_by_id.keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "item_id",
                "placed",
                "merit",
                "s2_count",
                "s3_sum",
                "s3_max",
                "s3_avg",
                "subsets_json",
                "gains_json",
            ])

            for iid in item_ids:
                it = items_by_id[iid]
                placed = 0 if solution.assignments.get(iid) is None else 1
                m = merit_of(float(it.profit), float(it.weight))

                fair_item = fair[iid]

                subsets_json = json.dumps(fair_item.subsets, ensure_ascii=False, separators=(",", ":"))
                gains_json = json.dumps([float(g) for g in fair_item.gains], ensure_ascii=False, separators=(",", ":"))

                w.writerow([
                    iid,
                    placed,
                    _fmt(m),
                    _fmt(fair_item.s2),
                    _fmt(fair_item.s3_sum),
                    _fmt(fair_item.s3_max),
                    _fmt(fair_item.s3_avg),
                    subsets_json,
                    gains_json,
                ])

        return path

    def write_iteration_metrics(
            self,
            state: SelectionState,
            solution: Solution,
            placement_log_path: str | None = None,
            *,
            iter_index: int = 0,
            bin_size: int = 1,
            near_miss_tau: float = 3.0,
    ) -> None:
        """
        Produce iteration-level CSVs to support resequencing:

          - iteration_summary.csv
          - per_knapsack_iter.csv
          - per_item_iter.csv
          - gap_histogram.csv

        Reads placement_log.csv (already written during Phase-2) to compute diagnostics.
        """
        # 1) Load placement rows
        if placement_log_path is None:
            placement_log_path = self._placement_log_path

        placement_rows: List[Dict[str, Any]] = []
        if os.path.exists(placement_log_path):
            with open(placement_log_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    placement_rows.append(row)

        # 2) Compute metrics via pure helpers
        global_metrics = compute_global_metrics(state, solution)
        gap_hist = build_gap_histogram(placement_rows, bin_size=bin_size)
        nm = count_near_misses(placement_rows, tau=near_miss_tau)
        per_knap = per_knapsack_metrics(state, solution, placement_rows)
        per_item = per_item_signals(state, solution, placement_rows, gap_hist)

        # 3) Write iteration_summary.csv
        summary_path = os.path.join(self.out_dir, "iteration_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "iter",
                "Total Profit",
                "OU",
                "% Placed",
                "PWE",
                "APK",
                "BL",
                "Used Sum",
                "Remaining Sum",
                "tight_fits",
                "just_missed",
                "Total Items",
                "Num Knapsacks",
                "Capacity Sum",
                "Total Possible Profit",
            ])


            w.writerow([
                int(iter_index),
                f'{global_metrics["TP"]:.3f}',
                f'{global_metrics["OU"]:.3f}',
                f'{global_metrics["SR"]:.3f}',
                f'{global_metrics["PWE"]:.3f}',
                f'{global_metrics["APK"]:.3f}',
                f'{global_metrics["BL"]:.3f}',
                f'{global_metrics["Used Sum"]:.3f}',
                f'{global_metrics["Remaining Sum"]:.3f}',
                int(nm["tight_fits"]),
                int(nm["just_missed"]),
                int(global_metrics["Total Items"]),
                int(global_metrics["Num Knapsacks"]),
                f'{global_metrics["Capacity Sum"]:.3f}',
                f'{global_metrics["Total Possible Profit"]:.3f}',
            ])

        # 4) Write per_knapsack_iter.csv
        knap_iter_path = os.path.join(self.out_dir, "per_knapsack_iter.csv")
        with open(knap_iter_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "iter",
                "knapsack_id",
                "capacity",
                "used",
                "remaining",
                "utilization_pct",
                "items_placed",
                "profit_sum",
                "pain_score",
            ])
            for row in per_knap:
                w.writerow([
                    int(iter_index),
                    row["knapsack_id"],
                    f'{float(row["capacity"]):.3f}',
                    f'{float(row["used"]):.3f}',
                    f'{float(row["remaining"]):.3f}',
                    f'{float(row["utilization_pct"]):.3f}',
                    int(row["items_placed"]),
                    f'{float(row["profit_sum"]):.3f}',
                    f'{float(row["pain_score"]):.6f}',
                ])

        # 5) Write per_item_iter.csv
        item_iter_path = os.path.join(self.out_dir, "per_item_iter.csv")
        with open(item_iter_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "iter",
                "item_id",
                "placed",
                "knapsack_id",
                "profit",
                "weight",
                "ratio",
                "rarity_by_capacity",
                "gap_closure_score",
                "unplaced_prev_iter",
            ])
            for row in per_item:
                w.writerow([
                    int(iter_index),
                    row["item_id"],
                    int(row["placed"]),
                    row["knapsack_id"],
                    f'{float(row["profit"]):.3f}',
                    f'{float(row["weight"]):.3f}',
                    f'{float(row["ratio"]):.6f}' if row["ratio"] != float("inf") else "inf",
                    int(row["rarity_by_capacity"]),
                    int(row["gap_closure_score"]),
                    int(row["unplaced_prev_iter"]),
                ])

        # 6) Write gap_histogram.csv
        gap_hist_path = os.path.join(self.out_dir, "gap_histogram.csv")
        with open(gap_hist_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iter", "bucket", "count"])
            for bucket, count in gap_hist:
                w.writerow([int(iter_index), int(bucket), int(count)])


def _fmt(x: float) -> str:
    return f"{float(x):.3f}"
