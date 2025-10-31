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
from typing import Dict, List, Optional

from src.planning import SelectionState, PlacementDecision, Solution
from src.business_objects.items import Item

from src.quality_metrics.fairness import compute_per_item_fairness


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
        remaining_after: Dict[str, float],
    ) -> str:
        """
        Append a single placement decision row.

        Columns:
          step_index, item_id, knapsack_id, profit, weight, reason, remaining_after_json
        """
        self._ensure_placement_header()

        with open(self._placement_log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                self._placement_step_idx,
                item.id,
                "" if decision.knapsack_id is None else decision.knapsack_id,
                float(item.profit),
                float(item.weight),
                decision.reason or "",
                json.dumps(remaining_after, ensure_ascii=False),
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
        Global KPIs across all knapsacks.

        Columns:
          total_items, placed_items, skipped_items, placed_pct,
          total_profit, capacity_sum, used_sum, remaining_sum, utilization_pct,
          selection_rate, profit_weight_efficiency, balance_load_std, avg_profit_per_knapsack
        """
        import os
        import csv
        import math

        path = os.path.join(self.out_dir, filename)

        # Base counts
        total_items = len(state.items)
        placed_items = sum(1 for _, k in solution.assignments.items() if k is not None)
        skipped_items = total_items - placed_items
        placed_pct = 0.0 if total_items == 0 else (placed_items / total_items) * 100.0  # percent

        # Capacity aggregates
        cap_sum = sum(float(k.capacity) for k in state.knapsacks)
        rem_sum = sum(float(r) for r in solution.remaining.values())
        used_sum = max(0.0, cap_sum - rem_sum)
        util_pct = 0.0 if cap_sum == 0.0 else (used_sum / cap_sum) * 100.0  # percent

        # ---- New metrics ----
        # Selection Rate (percent), same definition as placed_pct but with explicit name
        selection_rate = placed_pct

        # Profit-Weight Efficiency (PWE) = total_profit / total_weight_used (packed items only)
        items_by_id = {it.id: it for it in state.items}
        total_weight_used = sum(
            float(items_by_id[item_id].weight)
            for item_id, knap_id in solution.assignments.items()
            if knap_id is not None
        )
        profit_weight_efficiency = 0.0 if total_weight_used == 0.0 else float(solution.total_profit) / total_weight_used

        # Avg Profit per Knapsack (APK) = total_profit / m
        m = len(state.knapsacks)
        avg_profit_per_knapsack = 0.0 if m == 0 else float(solution.total_profit) / m

        # Balance Load (Std Dev of knapsack loads)
        # load_k = used capacity in knapsack k = C_k - remaining_k
        loads = [
            float(k.capacity) - float(solution.remaining.get(k.id, 0.0))
            for k in state.knapsacks
        ]
        if loads:
            mean_load = sum(loads) / len(loads)
            balance_load_std = math.sqrt(sum((x - mean_load) ** 2 for x in loads) / len(loads))
        else:
            balance_load_std = 0.0
        # ---------------------

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "total_items",
                "placed_items",
                "skipped_items",
                "placed_pct",
                "total_profit",
                "capacity_sum",
                "used_sum",
                "remaining_sum",
                "utilization_pct",
                "profit_weight_efficiency",
                "balance_load_std",
                "avg_profit_per_knapsack",
            ])
            w.writerow([
                total_items,
                placed_items,
                skipped_items,
                f"{placed_pct:.3f}",
                f"{solution.total_profit:.3f}",
                f"{cap_sum:.3f}",
                f"{used_sum:.3f}",
                f"{rem_sum:.3f}",
                f"{util_pct:.3f}",
                f"{profit_weight_efficiency:.3f}",
                f"{balance_load_std:.3f}",
                f"{avg_profit_per_knapsack:.3f}",
            ])
        return path

    def write_fairness_items_csv(
            self,
            state: SelectionState,
            solution: Solution,
            filename: str = "fairness_items.csv",
    ) -> str:
        """
        Per-item fairness CSV with all three aggregations in columns.

        Columns:
          item_id, placed, merit, s2_count, s3_sum, s3_max, s3_avg

        Notes:
          - 'placed' is 1 if the item is assigned, else 0.
          - 'merit' = p_j / w_j, using +inf if w_j==0 and p_j>0, else 0 when both 0.
          - Numeric values formatted to 3 decimals via _fmt().
        """
        import os
        import csv

        path = os.path.join(self.out_dir, filename)

        # Helpers
        def merit_of(p: float, w: float) -> float:
            if w == 0.0:
                return float("inf") if p > 0.0 else 0.0
            return float(p) / float(w)

        items_by_id = {it.id: it for it in state.items}

        fair_sum = compute_per_item_fairness(state, solution, mode="sum")
        fair_max = compute_per_item_fairness(state, solution, mode="max")
        fair_avg = compute_per_item_fairness(state, solution, mode="avg")

        # Write rows; keep a stable order (by item_id ascending)
        item_ids = sorted(items_by_id.keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "placed", "merit", "s2_count", "s3_sum", "s3_max", "s3_avg"])
            for iid in item_ids:
                it = items_by_id[iid]
                placed = 0 if solution.assignments.get(iid) is None else 1
                m = merit_of(float(it.profit), float(it.weight))

                # s2_count should be consistent across modes; take from "sum"
                s2 = fair_sum[iid].s2_count
                s3_s = fair_sum[iid].s3
                s3_mx = fair_max[iid].s3
                s3_av = fair_avg[iid].s3

                # Use _fmt for all floats (assumes _fmt exists in Tracker; if not, replace with f"{...:.3f}")
                w.writerow([
                    iid,
                    placed,
                    _fmt(m),
                    s2,
                    _fmt(s3_s),
                    _fmt(s3_mx),
                    _fmt(s3_av),
                ])

        return path


def _fmt(x: float) -> str:
    return f"{float(x):.3f}"
