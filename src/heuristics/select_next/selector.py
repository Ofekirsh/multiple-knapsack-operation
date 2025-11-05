# -*- coding: utf-8 -*-
"""
Phase 1: Select Next (item sequencing).

You provide an `order` list defining priority left→right.

Direction rules (fixed):
  - profit        -> descending (higher first)
  - ratio         -> descending (higher first)
  - weight        -> descending by default
  - -weight       -> ascending (lighter first)
  - log-linear    -> descending (higher score first; see formula below)
  - sweet-spot    -> descending (higher score first; fixed target w0)
  - capacity-spot -> descending (higher score first; target w0 from knapsack capacities)
  - id            -> ascending (only used as final deterministic tiebreaker)

We always append 'id' at the end if missing.

New: "log-linear" score (general, monotone resequencing)
--------------------------------------------------------
score(j) = gamma * ln(ratio_j) - lambda * ln(weight_j) + delta * ln(profit_j)

Defaults:
  gamma  = 0.70   # compresses merit gaps (smaller than 1 = gentler differences)
  lambda = 0.30   # soft preference to lighter items
  delta  = 0.15   # small absolute-profit bias to break ties

New: "sweet-spot" score (lift items near target weight w0 you set)
------------------------------------------------------------------
score(j) = ln(ratio_j)
           + beta * exp( - (w_j - w0)^2 / (2*sigma^2) )   # Gaussian bump at w0
           + delta_ss * ln(profit_j)
           - lambda_ss * ln(weight_j)

Defaults:
  w0        = 20.0
  sigma     = 8.0
  beta      = 0.80
  delta_ss  = 0.10
  lambda_ss = 0.10

New: "capacity-spot" (general; auto-centers on knapsack capacity scale)
-----------------------------------------------------------------------
Let capacities = {c_i}. We compute:
  w0    = median(c_i)
  sigma = 0.25 * mean(c_i)     # bandwidth derived from instance scale

score(j) = ln(ratio_j)
           + beta_cs * exp( - (w_j - w0)^2 / (2*sigma^2) )
           + delta_cs * ln(profit_j)
           - lambda_cs * ln(weight_j)

Defaults:
  beta_cs   = 0.80
  delta_cs  = 0.10
  lambda_cs = 0.10

Notes:
- Uses safe logs: ln(max(eps, x)) with eps=1e-12 to avoid -inf/NaN.
- If weight == 0 and profit > 0 -> ratio = +inf, ln(ratio) is clamped to avoid exploding scores.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional

import math
from statistics import median, mean

from src.planning.state import SelectionState
from .features import build_feature_table

# ----------------------------
# Config for "log-linear" mode
# ----------------------------
_LOG_LINEAR_GAMMA: float = 0.70
_LOG_LINEAR_LAMBDA: float = 0.30
_LOG_LINEAR_DELTA: float = 0.15

# ----------------------------
# Config for "sweet-spot" mode (manual target)
# ----------------------------
_SWEET_W0: float = 20.0      # target weight to reward
_SWEET_SIGMA: float = 8.0    # bandwidth
_SWEET_BETA: float = 0.80    # bump strength
_SWEET_DELTA: float = 0.10   # small ln(profit) bias
_SWEET_LAMBDA: float = 0.10  # small ln(weight) penalty

# ----------------------------
# Config for "capacity-spot" mode (auto target from capacities)
# ----------------------------
_CAPSPOT_BETA: float = 0.80
_CAPSPOT_DELTA: float = 0.10
_CAPSPOT_LAMBDA: float = 0.10
_CAPSPOT_SIGMA_FRAC: float = 0.25  # sigma = SIGMA_FRAC * mean(capacities)

# Numerics
_EPS: float = 1e-12
# Optional clamp for extremely large ratios in log:
_MAX_LOG_RATIO: float = 50.0  # exp(50) ~ 3.0e21; protects from inf if weight ~ 0

# Allowed keys for user order
_ALLOWED_KEYS = {
    "profit", "ratio", "weight", "-weight",
    "log-linear", "sweet-spot", "capacity-spot",
    "id"
}


def _normalize_order(order: Sequence[str] | None) -> list[str]:
    """
    Normalize the user-provided order:
      - Default to ["ratio", "profit", "-weight"] if None/empty
      - Keep first occurrence only (deduplicate while preserving order)
      - Validate keys against the allowed set
      - Ensure 'id' is present as the final key
    """
    if not order:
        norm = ["ratio", "profit", "-weight"]
    else:
        seen: set[str] = set()
        norm: list[str] = []
        for raw in order:
            key = str(raw).strip()
            if key not in _ALLOWED_KEYS:
                raise ValueError(
                    f"Unknown order key '{key}'. "
                    f"Allowed: {sorted(_ALLOWED_KEYS)}"
                )
            if key not in seen:
                seen.add(key)
                norm.append(key)

    if "id" not in norm:
        norm.append("id")
    return norm


def _safe_log(x: float) -> float:
    return math.log(max(_EPS, float(x)))


def _log_linear_score(features: Dict[str, float]) -> float:
    """
    score = gamma * ln(ratio) - lambda * ln(weight) + delta * ln(profit)
    """
    p = float(features.get("profit", 0.0))
    w = float(features.get("weight", 0.0))
    r = float(features.get("ratio", 0.0))

    ln_p = _safe_log(p)
    ln_w = _safe_log(w)
    ln_r = _safe_log(r)
    if ln_r > _MAX_LOG_RATIO:
        ln_r = _MAX_LOG_RATIO

    return (
        _LOG_LINEAR_GAMMA * ln_r
        - _LOG_LINEAR_LAMBDA * ln_w
        + _LOG_LINEAR_DELTA * ln_p
    )


def _sweet_spot_score(features: Dict[str, float]) -> float:
    """
    score = ln(ratio)
            + beta * exp( - (w - w0)^2 / (2*sigma^2) )
            + delta_ss * ln(profit)
            - lambda_ss * ln(weight)
    """
    p = float(features.get("profit", 0.0))
    w = float(features.get("weight", 0.0))
    r = float(features.get("ratio", 0.0))

    ln_p = _safe_log(p)
    ln_w = _safe_log(w)

    ln_r = _safe_log(r)
    if ln_r > _MAX_LOG_RATIO:
        ln_r = _MAX_LOG_RATIO

    if _SWEET_SIGMA <= 0:
        bump = 0.0
    else:
        diff = w - _SWEET_W0
        bump = math.exp(-(diff * diff) / (2.0 * _SWEET_SIGMA * _SWEET_SIGMA))

    return ln_r + (_SWEET_BETA * bump) + (_SWEET_DELTA * ln_p) - (_SWEET_LAMBDA * ln_w)


def _capacity_spot_params(state: SelectionState) -> Tuple[float, float]:
    """
    Compute (w0, sigma) from knapsack capacities:
      w0    = median(capacities)
      sigma = CAPSPOT_SIGMA_FRAC * mean(capacities)
    """
    caps = [float(k.capacity) for k in state.knapsacks]
    if not caps:
        # Fallback to avoid division by zero; arbitrary neutral params
        return 1.0, 1.0
    w0 = float(median(caps))
    sig = float(_CAPSPOT_SIGMA_FRAC * max(_EPS, mean(caps)))
    if sig <= 0.0:
        sig = 1.0
    return w0, sig


def _capacity_spot_score(
    features: Dict[str, float],
    *,
    w0: float,
    sigma: float,
) -> float:
    """
    score = ln(ratio)
            + beta_cs * exp( - (w - w0)^2 / (2*sigma^2) )
            + delta_cs * ln(profit)
            - lambda_cs * ln(weight)
    """
    p = float(features.get("profit", 0.0))
    w = float(features.get("weight", 0.0))
    r = float(features.get("ratio", 0.0))

    ln_p = _safe_log(p)
    ln_w = _safe_log(w)

    ln_r = _safe_log(r)
    if ln_r > _MAX_LOG_RATIO:
        ln_r = _MAX_LOG_RATIO

    diff = w - w0
    bump = math.exp(-(diff * diff) / (2.0 * sigma * sigma)) if sigma > 0 else 0.0

    return ln_r + (_CAPSPOT_BETA * bump) + (_CAPSPOT_DELTA * ln_p) - (_CAPSPOT_LAMBDA * ln_w)


def _sort_key_for_item(
    item_id: str,
    features: Dict[str, float],
    order_keys: list[str],
    *,
    capspot_params: Optional[Tuple[float, float]] = None,
) -> Tuple:
    """
    Build a Python sort key tuple following our direction rules.

    Implementation trick:
      - Python sorts ascending; for "descending" we negate numeric values.
      - 'id' is kept as string (ascending).
    """
    key: list = []
    for k in order_keys:
        if k == "id":
            key.append(item_id)  # ascending
            continue

        if k == "profit":
            key.append(-features.get("profit", float("-inf")))
            continue
        if k == "ratio":
            key.append(-features.get("ratio", float("-inf")))
            continue

        if k == "weight":
            key.append(-features.get("weight", float("-inf")))
            continue
        if k == "-weight":
            key.append(features.get("weight", float("inf")))
            continue

        if k == "log-linear":
            key.append(-_log_linear_score(features))
            continue

        if k == "sweet-spot":
            key.append(-_sweet_spot_score(features))
            continue

        if k == "capacity-spot":
            if capspot_params is None:
                # neutral if params missing
                key.append(0.0)
            else:
                w0, sig = capspot_params
                key.append(-_capacity_spot_score(features, w0=w0, sigma=sig))
            continue

        raise AssertionError(f"Unhandled order key: {k}")

    return tuple(key)


def select_next(
    state: SelectionState,
    order: Sequence[str] | None = None,
) -> list[str]:
    """
    Return an ordered list of item IDs according to `order`.

    Examples:
      ["ratio", "profit", "-weight"]
      ["log-linear"]
      ["sweet-spot"]
      ["capacity-spot"]             # NEW: lift items near capacity scale
      ["capacity-spot", "ratio"]    # bump near capacity, then value density

    Notes
    -----
    - No mutation occurs here.
    - Sorting is deterministic; 'id' is always the final tiebreaker (ascending).
    """
    order_keys = _normalize_order(order)
    feat_table = build_feature_table(state.items)  # profit, weight, ratio

    # Pre-compute capacity-spot params once per call (if needed)
    capspot_params: Optional[Tuple[float, float]] = None
    if "capacity-spot" in order_keys:
        capspot_params = _capacity_spot_params(state)

    sortable: List[Tuple[str, Tuple]] = []
    for it in state.items:
        row = feat_table[it.id]
        sortable.append(
            (it.id, _sort_key_for_item(it.id, row, order_keys, capspot_params=capspot_params))
        )

    sortable.sort(key=lambda t: t[1])
    return [iid for iid, _ in sortable]
