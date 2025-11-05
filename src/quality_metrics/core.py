# -*- coding: utf-8 -*-
"""
quality_metrics/core.py

Pure helpers to compute iteration-level KPIs for MKP runs.
- No side effects
- No external dependencies (pandas not required)
- Works off SelectionState, Solution, and a placement_log parsed as rows (Dicts)

Public API (MVP):
  - compute_global_metrics(state, solution) -> Dict[str, float]
  - build_gap_histogram(placement_rows, bin_size=1) -> List[Tuple[int,int]]
  - count_near_misses(placement_rows, tau=3.0) -> Dict[str, int]
  - per_knapsack_metrics(state, solution, placement_rows) -> List[Dict]
  - per_item_signals(state, solution, placement_rows, gap_hist) -> List[Dict]
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import json
import math

# ---- Type hints (import only for typing; avoid heavy deps at runtime) ----
try:
    from src.planning import SelectionState, Solution
    from src.business_objects.items import Item
except Exception:  # pragma: no cover - typing only
    SelectionState = Any  # type: ignore
    Solution = Any  # type: ignore
    Item = Any  # type: ignore


# ---------------------------------------------------------------------------
# Utility: safe float parsing (CSV values may be strings or blanks)
# ---------------------------------------------------------------------------
def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return default
    try:
        return float(s)
    except Exception:
        return default


def _safe_ratio(p: float, w: float) -> float:
    if w == 0.0:
        return float("inf") if p > 0.0 else 0.0
    return p / w


# ---------------------------------------------------------------------------
# 1) Global metrics (TP, OU, SR, PWE, APK, BL, Used/Remaining)
# ---------------------------------------------------------------------------
def compute_global_metrics(state: SelectionState, solution: Solution) -> Dict[str, float]:
    """
    Returns:
      {
        "TP": ...,
        "OU": ...,           # percent (0..100)
        "SR": ...,           # percent (0..100)
        "PWE": ...,
        "APK": ...,
        "BL": ...,
        "Used Sum": ...,
        "Remaining Sum": ...,
        "Total Items": ...,
        "Num Knapsacks": ...,
        "Capacity Sum": ...,
        "Total Possible Profit": ...
      }
    """
    total_items = len(state.items)
    placed_items = sum(1 for _, k in solution.assignments.items() if k is not None)
    num_knapsacks = len(state.knapsacks)

    capacity_sum = sum(float(k.capacity) for k in state.knapsacks)
    rem_sum = sum(float(r) for r in solution.remaining.values())
    used_sum = max(0.0, capacity_sum - rem_sum)

    TP = float(solution.total_profit)
    OU = 0.0 if capacity_sum == 0.0 else (used_sum / capacity_sum) * 100.0
    SR = 0.0 if total_items == 0 else (placed_items / total_items) * 100.0

    items_by_id: Dict[str, Item] = {it.id: it for it in state.items}
    total_weight_used = sum(
        float(items_by_id[iid].weight) for iid, kid in solution.assignments.items() if kid is not None
    )
    PWE = 0.0 if total_weight_used == 0.0 else TP / total_weight_used
    APK = 0.0 if num_knapsacks == 0 else TP / num_knapsacks

    loads = [float(k.capacity) - float(solution.remaining.get(k.id, 0.0)) for k in state.knapsacks]
    if loads:
        mean_load = sum(loads) / len(loads)
        BL = math.sqrt(sum((x - mean_load) ** 2 for x in loads) / len(loads))
    else:
        BL = 0.0

    total_possible_profit = sum(float(it.profit) for it in state.items)

    return {
        "TP": float(TP),
        "OU": float(OU),
        "SR": float(SR),
        "PWE": float(PWE),
        "APK": float(APK),
        "BL": float(BL),
        "Used Sum": float(used_sum),
        "Remaining Sum": float(rem_sum),
        "Total Items": float(total_items),
        "Num Knapsacks": float(num_knapsacks),
        "Capacity Sum": float(capacity_sum),
        "Total Possible Profit": float(total_possible_profit),
    }


# ---------------------------------------------------------------------------
# 2) Gap histogram from placement log
# ---------------------------------------------------------------------------
def build_gap_histogram(
    placement_rows: List[Dict[str, Any]],
    bin_size: int = 1,
) -> List[Tuple[int, int]]:
    """
    Build a histogram of leftover capacities after each step (all knapsacks).
    - Reads the 'remaining_after_json' column from placement log rows.
    - Buckets by floor(leftover / bin_size) * bin_size (default 1).

    Returns sorted list of (bucket, count).
    """
    buckets: Dict[int, int] = defaultdict(int)

    for row in placement_rows:
        s = row.get("remaining_after_json", "")
        if not s:
            continue
        try:
            rem_after = json.loads(s)
        except Exception:
            # If the CSV already serialized dict (not JSON string), try direct
            rem_after = s if isinstance(s, dict) else {}
        for _kid, leftover in (rem_after or {}).items():
            val = _to_float(leftover, 0.0)
            b = int(math.floor(val / max(1, bin_size)) * max(1, bin_size))
            buckets[b] += 1

    out = sorted(buckets.items(), key=lambda t: t[0])
    return out


# ---------------------------------------------------------------------------
# 3) Near-miss counters (tight fits & just missed)
# ---------------------------------------------------------------------------
def count_near_misses(
    placement_rows: List[Dict[str, Any]],
    tau: float = 3.0,
) -> Dict[str, int]:
    """
    Counts:
      - tight_fits: closest_fit_margin in [0, tau]
      - just_missed: reason='no_feasible_knapsack' and closest_fit_margin in [-tau, 0)

    Returns: {"tight_fits": int, "just_missed": int}
    """
    tight = 0
    miss = 0
    for row in placement_rows:
        margin = _to_float(row.get("closest_fit_margin"), default=float("nan"))
        reason = (row.get("reason") or "").strip()

        if not math.isnan(margin):
            if 0.0 <= margin <= tau:
                tight += 1
            elif reason == "no_feasible_knapsack" and (-tau) <= margin < 0.0:
                miss += 1

    return {"tight_fits": int(tight), "just_missed": int(miss)}


# ---------------------------------------------------------------------------
# 4) Per-knapsack metrics (extend your existing per_knapsack.csv logic)
# ---------------------------------------------------------------------------
def per_knapsack_metrics(
    state: SelectionState,
    solution: Solution,
    placement_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Returns one dict per knapsack with:
      knapsack_id, capacity, used, remaining, utilization_pct, items_placed, profit_sum,
      pain_score

    pain_score (MVP): remaining / capacity  (higher = worse)
      - Later you can blend in more signals (near_miss_hits, early_fill, etc.)
    """
    items_by_id: Dict[str, Item] = {it.id: it for it in state.items}
    cap_by_id: Dict[str, float] = {k.id: float(k.capacity) for k in state.knapsacks}

    profit_by_knap: Dict[str, float] = {k: 0.0 for k in cap_by_id}
    count_by_knap: Dict[str, int] = {k: 0 for k in cap_by_id}

    for iid, kid in solution.assignments.items():
        if kid is None:
            continue
        profit_by_knap[kid] += float(items_by_id[iid].profit)
        count_by_knap[kid] += 1

    rows: List[Dict[str, Any]] = []
    for kid, cap in cap_by_id.items():
        rem = float(solution.remaining.get(kid, 0.0))
        used = max(0.0, cap - rem)
        util = 0.0 if cap == 0.0 else (used / cap) * 100.0
        pain = 0.0 if cap == 0.0 else (rem / cap)  # normalized remaining

        rows.append({
            "knapsack_id": kid,
            "capacity": float(cap),
            "used": float(used),
            "remaining": float(rem),
            "utilization_pct": float(util),
            "items_placed": int(count_by_knap.get(kid, 0)),
            "profit_sum": float(profit_by_knap.get(kid, 0.0)),
            "pain_score": float(pain),
        })
    return rows


# ---------------------------------------------------------------------------
# 5) Per-item signals (for resequencing rules)
# ---------------------------------------------------------------------------
def per_item_signals(
    state: SelectionState,
    solution: Solution,
    placement_rows: List[Dict[str, Any]],
    gap_hist: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """
    Returns one dict per item with:
      item_id, placed(0/1), knapsack_id, profit, weight, ratio,
      rarity_by_capacity, gap_closure_score, unplaced_prev_iter

    Notes:
      - rarity_by_capacity: # knapsacks with capacity >= weight (problem-static rarity)
      - gap_closure_score: histogram count at bucket ≈ round(weight)
      - unplaced_prev_iter: 0 for iter-1 (you can set it to 1 starting iter-2)
    """
    items_by_id: Dict[str, Item] = {it.id: it for it in state.items}
    knap_caps: List[float] = [float(k.capacity) for k in state.knapsacks]

    # Precompute histogram lookup
    gap_map: Dict[int, int] = {int(b): int(c) for b, c in gap_hist}

    out: List[Dict[str, Any]] = []
    for iid, it in items_by_id.items():
        placed_flag = 0
        placed_knap = ""
        kid = solution.assignments.get(iid)
        if kid is not None:
            placed_flag = 1
            placed_knap = str(kid)

        p = float(it.profit)
        w = float(it.weight)
        r = _safe_ratio(p, w)

        rarity = sum(1 for cap in knap_caps if cap >= w)
        # Use nearest-integer bucket for weight
        bucket = int(round(w))
        gap_score = int(gap_map.get(bucket, 0))

        out.append({
            "item_id": iid,
            "placed": int(placed_flag),
            "knapsack_id": placed_knap,
            "profit": float(p),
            "weight": float(w),
            "ratio": float(r),
            "rarity_by_capacity": int(rarity),
            "gap_closure_score": int(gap_score),
            "unplaced_prev_iter": 0,  # will be updated by the driver from iter-2 onward
        })
    # Stable order by item_id for readability
    out.sort(key=lambda d: d["item_id"])
    return out
