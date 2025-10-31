# -*- coding: utf-8 -*-
"""
Fairness metrics for MKP solutions based on signed 'spoiled / deprived' severity.

UPDATE: A blocking pair (j ∈ U, k ∈ P) is counted ONLY IF the swap is feasible
in k's knapsack under a simple sufficient condition:
    s_j > s_k  AND  w_j < w_k
(i.e., the unpacked item is strictly lighter than the packed item it would replace).
This ensures we don't count 'unfairness' that could not be corrected by a direct swap.

Definitions:
- Merit: s_j = p_j / w_j  (with s_j = +inf if w_j == 0 and p_j > 0; 0 if p_j == 0 == w_j)
- Packed set P: items assigned to any knapsack
- Unpacked set U: items not assigned
- Blocking relation: item k (packed) blocks j (unpacked) if s_j > s_k AND w_j < w_k

Per-item scores:
- S2(j): signed count
    * For j ∈ U (deprived):  +|{k ∈ P : s_j > s_k, w_j < w_k}|
    * For k ∈ P (spoiled):   -|{j ∈ U : s_j > s_k, w_j < w_k}|
- S3_list(j): list of signed severities
    * j ∈ U:  add  (s_j - s_k)  for each feasible blocking k
    * k ∈ P:  add -(s_j - s_k)  for each feasible blocked j
- S3(j): aggregation over S3_list(j) using mode ∈ {"sum","max","avg"}; empty list -> 0

Global property (unchanged): for mode="sum", pairwise contributions cancel in total
(though with the w_j < w_k gate, counts can decrease; still sums to ~0 numerically).
"""

from __future__ import annotations
from typing import Dict, List, Literal
from dataclasses import dataclass

from src.business_objects.items import Item
from src.planning import SelectionState, Solution


AggMode = Literal["sum", "max", "avg"]


@dataclass(frozen=True)
class PerItemFairness:
    s2_count: int
    s3_list: List[float]
    s3: float


def _merit(item: Item) -> float:
    p = float(item.profit)
    w = float(item.weight)
    if w == 0.0:
        return float("inf") if p > 0.0 else 0.0
    return p / w


def _aggregate(vals: List[float], mode: AggMode) -> float:
    if not vals:
        return 0.0
    if mode == "sum":
        return float(sum(vals))
    if mode == "max":
        return float(max(vals))
    if mode == "avg":
        return float(sum(vals) / len(vals))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def compute_per_item_fairness(
    state: SelectionState,
    solution: Solution,
    mode: AggMode = "sum",
) -> Dict[str, PerItemFairness]:
    """
    Compute S2, S3_list, and S3(j) for every item j with feasibility-gated pairs:
      (j ∈ U, k ∈ P) contributes iff s_j > s_k and w_j < w_k.

    Returns
    -------
    Dict[item_id, PerItemFairness]
    """
    items_by_id: Dict[str, Item] = {it.id: it for it in state.items}
    merit: Dict[str, float] = {iid: _merit(it) for iid, it in items_by_id.items()}
    weight: Dict[str, float] = {iid: float(it.weight) for iid, it in items_by_id.items()}

    # Partition items by placement
    packed: List[str] = [iid for iid, kid in solution.assignments.items() if kid is not None]
    unpacked: List[str] = [iid for iid, kid in solution.assignments.items() if kid is None]

    res: Dict[str, PerItemFairness] = {}

    # Precompute (id, merit, weight) tuples for speed
    packed_stats = [(k, merit[k], weight[k]) for k in packed]
    unpacked_stats = [(j, merit[j], weight[j]) for j in unpacked]

    # Deprived items j ∈ U
    for j, s_j, w_j in unpacked_stats:
        s3_list: List[float] = []
        count = 0
        for k, s_k, w_k in packed_stats:
            # Feasibility-gated blocking: s_j > s_k and w_j < w_k
            if (s_j > s_k) and (w_j < w_k):
                count += 1
                s3_list.append(s_j - s_k)
        s3 = _aggregate(s3_list, mode)
        res[j] = PerItemFairness(s2_count=count, s3_list=s3_list, s3=s3)

    # Spoiled items k ∈ P
    for k, s_k, w_k in packed_stats:
        s3_list: List[float] = []
        count = 0
        for j, s_j, w_j in unpacked_stats:
            # Feasibility-gated blocked: s_j > s_k and w_j < w_k
            if (s_j > s_k) and (w_j < w_k):
                count -= 1
                s3_list.append(-(s_j - s_k))
        s3 = _aggregate(s3_list, mode)
        res[k] = PerItemFairness(s2_count=count, s3_list=s3_list, s3=s3)

    return res


def compute_global_fairness(
    state: SelectionState,
    solution: Solution,
    mode: AggMode = "sum",
) -> float:
    """
    Global signed severity: sum_j S3(j) (with feasibility-gated pairs).
    For mode="sum" this remains ~0 (numerical noise aside).
    """
    per_item = compute_per_item_fairness(state, solution, mode=mode)
    return float(sum(v.s3 for v in per_item.values()))


def compute_knapsack_fairness(
    state: SelectionState,
    solution: Solution,
    mode: AggMode = "sum",
) -> Dict[str, float]:
    """
    Per-knapsack signed severity: sum of S3(j) over items assigned to that knapsack.
    Interpretation: how much 'spoiled' advantage accumulated inside each knapsack (negative values).
    """
    per_item = compute_per_item_fairness(state, solution, mode=mode)
    score_by_knap: Dict[str, float] = {k.id: 0.0 for k in state.knapsacks}
    for iid, kid in solution.assignments.items():
        if kid is not None:
            score_by_knap[kid] += per_item[iid].s3
    return score_by_knap


def export_fairness_per_item_rows(
    state: SelectionState,
    solution: Solution,
    mode: AggMode = "sum",
) -> List[List[str]]:
    """
    Build CSV rows for per-item fairness:
      item_id, placed, merit, s2_count, s3_mode
    (s3_list omitted for compactness; can be added if you want verbose mode.)
    """
    items_by_id: Dict[str, Item] = {it.id: it for it in state.items}
    per_item = compute_per_item_fairness(state, solution, mode=mode)

    rows: List[List[str]] = [["item_id", "placed", "merit", "s2_count", f"s3_{mode}"]]
    for iid, fair in per_item.items():
        placed = 0 if solution.assignments[iid] is None else 1
        rows.append([
            iid,
            str(placed),
            f"{_merit(items_by_id[iid]):.6f}",
            str(fair.s2_count),
            f"{fair.s3:.6f}",
        ])
    return rows
