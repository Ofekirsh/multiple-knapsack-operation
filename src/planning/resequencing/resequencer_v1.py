# -*- coding: utf-8 -*-
"""
Resequencer v1: anchors per knapsack + simple remainder ranking.

Signature required by run_greedy():
    resequencer(state, policy, solution, placement_rows) -> List[str]

Behavior:
  1) Compute knapsack "pain order" = sort knapsacks by remaining capacity DESC (more remaining = more "pain").
  2) For each knapsack in that order, choose a 2-item anchor pair from currently available items:
        - must fit capacity
        - maximize (total_profit) and then minimize leftover (capacity - pair_weight)
     Greedy/approx: try each candidate 'a' (good density), pick best partner 'b' among remaining that fits.
     Once chosen, remove a,b from the available pool.
  3) Emit anchors block: for each chosen pair, output heavier item first, then its partner.
  4) Rank the rest:
        score = (profit / weight) + UNPLACED_BOOST * 1[item was unplaced last iter]
        tie-break: lighter weight first, then higher profit
  5) Return concatenated list: anchors + ranked remainder.

Notes:
  - Deterministic given input (no RNG here).
  - Safe with any place-next heuristic (best_fit, max_remaining, ...).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

from src.planning import SelectionState, Policy, Solution
from src.business_objects.items import Item


# Tunables (lightweight; can be promoted to config later)
_UNPLACED_BOOST: float = 0.25      # mild nudge for items skipped last iteration
_MAX_ANCHORS_PER_KNAP: int = 1     # one pair per knapsack for now (keeps it simple)
_TRY_CANDIDATES_FRACTION: float = 1.0  # 1.0 = try all items as first-in-pair; <1 = speed-up


def _items_by_id(state: SelectionState) -> Dict[str, Item]:
    return {it.id: it for it in state.items}


def _placed_set(sol: Solution) -> set[str]:
    return {iid for iid, kid in sol.assignments.items() if kid is not None}


def _pain_order_knapsacks(state: SelectionState, sol: Solution) -> List[Tuple[str, float, float]]:
    """
    Return list of (knapsack_id, capacity, remaining) sorted by remaining DESC.
    """
    caps = {k.id: float(k.capacity) for k in state.knapsacks}
    rems = sol.remaining
    triples = [(kid, caps[kid], float(rems.get(kid, 0.0))) for kid in caps.keys()]
    triples.sort(key=lambda t: (-t[2], t[0]))  # more remaining first; tie by id
    return triples  # [(id, cap, rem), ...]


def _density(it: Item) -> float:
    w = float(it.weight)
    if w <= 0.0:
        return float("inf") if float(it.profit) > 0.0 else 0.0
    return float(it.profit) / w


def _choose_anchor_pair_for_knap(
    knap_cap: float,
    available: List[Item],
) -> Optional[Tuple[Item, Item]]:
    """
    Greedy approximate best pair:
      - iterate candidates a by density (desc), then profit (desc), then lighter first
      - for each a, scan partners b to fit and pick the one that maximizes total profit,
        breaking ties by smaller leftover.
      - return first winning pair found with best objective under this policy.
    Complexity: O(M^2) worst-case; fine for small/medium M. Can be pruned via TRY_CANDIDATES_FRACTION.
    """
    if len(available) < 2:
        return None

    # Sort available pool for stable behavior
    pool = sorted(
        available,
        key=lambda it: (-_density(it), -float(it.profit), float(it.weight), it.id),
    )

    # Optional candidate truncation for speed
    if 0.0 < _TRY_CANDIDATES_FRACTION < 1.0:
        cut = max(1, int(len(pool) * _TRY_CANDIDATES_FRACTION))
        pool = pool[:cut]

    best_pair: Optional[Tuple[Item, Item]] = None
    best_profit = float("-inf")
    best_leftover = float("inf")

    for i, a in enumerate(pool):
        wa = float(a.weight)
        if wa >= knap_cap:  # cannot fit any partner
            continue
        # Partner search among remaining (excluding a)
        for b in pool[i + 1:]:
            wb = float(b.weight)
            wsum = wa + wb
            if wsum > knap_cap:
                continue
            profit_sum = float(a.profit) + float(b.profit)
            leftover = knap_cap - wsum
            # Primary: higher total profit; Secondary: smaller leftover
            if (profit_sum > best_profit) or (profit_sum == best_profit and leftover < best_leftover):
                best_profit = profit_sum
                best_leftover = leftover
                best_pair = (a, b)

    return best_pair


def _emit_pair_heavy_first(a: Item, b: Item) -> List[str]:
    return [a.id, b.id] if float(a.weight) >= float(b.weight) else [b.id, a.id]


def resequencer_v1(
    state: SelectionState,
    policy: Policy,                       # unused here but kept for future tuning
    solution: Solution,
    placement_rows: List[Dict[str, Any]], # unused here; future: mine near-miss patterns
) -> List[str]:
    items_map = _items_by_id(state)
    placed_prev = _placed_set(solution)

    # Initial available pool is ALL items; anchors can reuse placed or unplaced items
    available_ids: List[str] = sorted(items_map.keys())
    available: List[Item] = [items_map[iid] for iid in available_ids]

    # 1) Pain order of knapsacks (more remaining first)
    knap_order = _pain_order_knapsacks(state, solution)

    # 2) Choose up to one anchor pair per knapsack (sequentially; resolves conflicts naturally)
    anchor_blocks: List[str] = []
    used_ids: set[str] = set()

    for kid, cap, _rem in knap_order:
        if len(available) - len(used_ids) < 2:
            break
        # Build current free pool
        free_pool = [it for it in available if it.id not in used_ids]
        pair = _choose_anchor_pair_for_knap(cap, free_pool)
        if pair is None:
            continue
        a, b = pair
        anchor_blocks.extend(_emit_pair_heavy_first(a, b))
        used_ids.add(a.id)
        used_ids.add(b.id)
        if len(anchor_blocks) >= 2 * _MAX_ANCHORS_PER_KNAP * len(knap_order):
            # global cap (rarely hits; defensive)
            break

    # 3) Rank remaining items
    remainder_ids = [iid for iid in available_ids if iid not in used_ids]
    def _score(iid: str) -> Tuple[float, float, float, str]:
        it = items_map[iid]
        dens = _density(it)
        boost = _UNPLACED_BOOST if iid not in placed_prev else 0.0
        # Sort key: primary score desc → (-score), then lighter first, then profit desc, then id
        score = dens + boost
        return (-score, float(it.weight), -float(it.profit), iid)

    remainder_ids.sort(key=_score)

    # 4) Final sequence: anchors block first, then remainder
    # Deduplicate in case any overlap
    seen: set[str] = set()
    ordered: List[str] = []
    for iid in anchor_blocks + remainder_ids:
        if iid not in seen:
            seen.add(iid)
            ordered.append(iid)

    return ordered
