# -*- coding: utf-8 -*-
"""
Knapsack ranking strategies for Phase 2 ("place next").

Public entry point:
    choose_best_knapsack(item, knapsacks, strategy="best_fit", eps=1e-12)
returns a RuntimeKnapsack (no mutation) or None if infeasible.

Supported strategies (all deterministic; tie-break by input order unless noted):
  1) first_fit
  2) best_fit                       -> argmin_k (r_k - w_j)
  3) max_remaining                  -> argmax_k r_k
  4) smallest_fit                   -> argmin_k C_k
  5) largest_fit                    -> argmax_k C_k
  6) balance_utilization            -> argmin_k Var({ (C_i - r_i')/C_i })
  7) best_fit_then_smallest         -> argmin_k ( (r_k - w_j), C_k ) (lexicographic)
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple, List

from src.business_objects.items import Item
from src.planning import RuntimeKnapsack

DEFAULT_EPS = 1e-12


# -----------------------------
# Feasibility / residual utils
# -----------------------------

def _can_fit(k: RuntimeKnapsack, item: Item, eps: float = DEFAULT_EPS) -> bool:
    return item.weight <= k.remaining + eps


def _residual_after(k: RuntimeKnapsack, item: Item) -> float:
    return k.remaining - item.weight


# -----------------------------
# Individual strategies (pure)
# -----------------------------

def _first_fit(
    item: Item, cands: List[RuntimeKnapsack], eps: float
) -> Optional[RuntimeKnapsack]:
    for k in cands:
        if _can_fit(k, item, eps):
            return k
    return None


def _best_fit(
    item: Item, feas: List[RuntimeKnapsack]
) -> Optional[RuntimeKnapsack]:
    # argmin residual >= -eps; feas already filtered by feasibility
    best: Optional[Tuple[RuntimeKnapsack, float]] = None
    for k in feas:
        res = _residual_after(k, item)  # >= -eps
        if best is None or res < best[1]:
            best = (k, res)
    return best[0] if best else None


def _max_remaining(
    item: Item, feas: List[RuntimeKnapsack]
) -> Optional[RuntimeKnapsack]:
    # argmax current remaining r_k
    best: Optional[Tuple[RuntimeKnapsack, float]] = None
    for k in feas:
        r = k.remaining
        if best is None or r > best[1]:
            best = (k, r)
    return best[0] if best else None


def _smallest_fit(
    item: Item, feas: List[RuntimeKnapsack]
) -> Optional[RuntimeKnapsack]:
    # argmin capacity C_k
    best: Optional[Tuple[RuntimeKnapsack, float]] = None
    for k in feas:
        C = k.capacity
        if best is None or C < best[1]:
            best = (k, C)
    return best[0] if best else None


def _largest_fit(
    item: Item, feas: List[RuntimeKnapsack]
) -> Optional[RuntimeKnapsack]:
    # argmax capacity C_k
    best: Optional[Tuple[RuntimeKnapsack, float]] = None
    for k in feas:
        C = k.capacity
        if best is None or C > best[1]:
            best = (k, C)
    return best[0] if best else None


def _balance_utilization(
    item: Item, feas: List[RuntimeKnapsack], all_knaps: List[RuntimeKnapsack]
) -> Optional[RuntimeKnapsack]:
    """
    Choose k minimizing utilization variance after hypothetically placing item j:
      u_i' = (C_i - r_i') / C_i, where
        r_i' = r_i for i != k and r_k' = r_k - w_j
      objective: Var({u_i'}) across i=1..m
    Ties resolved by input order of 'feas'.
    """
    import math

    wj = float(item.weight)

    def variance_after(chosen: RuntimeKnapsack) -> float:
        # Compute utilizations after hypothetical placement
        u_vals: List[float] = []
        for kk in all_knaps:
            Ci = float(kk.capacity)
            if kk is chosen:
                ri_prime = kk.remaining - wj
            else:
                ri_prime = kk.remaining
            # Guard against zero capacity (shouldn't happen, but be safe)
            ui = 0.0 if Ci == 0.0 else (Ci - ri_prime) / Ci
            u_vals.append(ui)
        # Var = mean(x^2) - mean(x)^2
        n = len(u_vals)
        mean = sum(u_vals) / n
        mean_sq = sum(u * u for u in u_vals) / n
        return max(0.0, mean_sq - mean * mean)

    best_k: Optional[RuntimeKnapsack] = None
    best_var: Optional[float] = None
    for k in feas:
        v = variance_after(k)
        if best_var is None or v < best_var:
            best_var = v
            best_k = k
    return best_k


def _best_fit_then_smallest(
    item: Item, feas: List[RuntimeKnapsack]
) -> Optional[RuntimeKnapsack]:
    """
    Lexicographic (residual, capacity):
      - minimize residual r_k - w_j (tightest fit),
      - break ties by smaller capacity C_k,
      - then by input order implicitly.
    """
    best: Optional[Tuple[RuntimeKnapsack, float, float]] = None
    for k in feas:
        res = _residual_after(k, item)
        cap = float(k.capacity)
        if best is None or (res < best[1]) or (res == best[1] and cap < best[2]):
            best = (k, res, cap)
    return best[0] if best else None


# -----------------------------
# Strategy dispatcher
# -----------------------------

def choose_best_knapsack(
    item: Item,
    knapsacks: Iterable[RuntimeKnapsack],
    strategy: str = "best_fit",
    eps: float = DEFAULT_EPS,
) -> Optional[RuntimeKnapsack]:
    """
    Decide which knapsack is best *right now* for placing 'item', or None if none fits.

    Parameters
    ----------
    item : Item
        The item to place.
    knapsacks : Iterable[RuntimeKnapsack]
        Candidate knapsacks (their order is used for stable tie-breaking).
    strategy : str
        One of: "first_fit", "best_fit", "max_remaining",
                "smallest_fit", "largest_fit",
                "balance_utilization", "best_fit_then_smallest".
    eps : float
        Feasibility tolerance for capacity checks.

    Returns
    -------
    Optional[RuntimeKnapsack]
        The chosen knapsack object (no mutation), or None if infeasible.
    """
    cands: List[RuntimeKnapsack] = list(knapsacks)
    feas = [k for k in cands if _can_fit(k, item, eps)]
    if not feas:
        return None

    if strategy == "first_fit":
        return _first_fit(item, cands, eps)
    if strategy == "best_fit":
        return _best_fit(item, feas)
    if strategy == "max_remaining":
        return _max_remaining(item, feas)
    if strategy == "smallest_fit":
        return _smallest_fit(item, feas)
    if strategy == "largest_fit":
        return _largest_fit(item, feas)
    if strategy == "balance_utilization":
        return _balance_utilization(item, feas, cands)
    if strategy == "best_fit_then_smallest":
        return _best_fit_then_smallest(item, feas)

    raise ValueError(
        f"Unknown placement strategy: {strategy}. "
        "Expected one of: first_fit, best_fit, max_remaining, "
        "smallest_fit, largest_fit, balance_utilization, best_fit_then_smallest."
    )
