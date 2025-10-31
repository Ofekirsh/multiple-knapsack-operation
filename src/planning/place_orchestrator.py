# -*- coding: utf-8 -*-
"""
Phase 2 Orchestrator: Placement (Place Next)

Single-step API that:
  1) Chooses the best knapsack for an item via heuristics (no mutation there).
  2) Commits the placement by mutating the chosen RuntimeKnapsack.remaining.
  3) Emits a PlacementDecision and optionally logs it via Tracker.

This module does not aggregate a full Solution; use a solver (e.g., greedy) to
iterate a whole queue and build the final Solution object.
"""

from __future__ import annotations
from typing import Optional, Dict

from src.business_objects.items import Item
from src.planning import RuntimeState, Policy, PlacementDecision
from src.heuristics.place_next.strategies import choose_best_knapsack
from .tracker import Tracker


def place_next(
    item: Item,
    state: RuntimeState,
    policy: Policy,
    tracker: Optional[Tracker] = None,
) -> PlacementDecision:
    """
    Decide & commit placement for a single item.

    Parameters
    ----------
    item : Item
        The item to place (reference from state.items).
    state : RuntimeState
        Mutable runtime state holding the current knapsacks.
    policy : Policy
        Uses policy.place_strategy and policy.eps.
    tracker : Tracker | None
        If provided, appends a CSV row with the decision and post-placement remainders.

    Returns
    -------
    PlacementDecision
        item_id, chosen knapsack_id (or None), and a short reason label.
    """
    # 1) Ask heuristics which knapsack is best right now (no mutation there)
    chosen = choose_best_knapsack(
        item=item,
        knapsacks=state.knapsacks,
        strategy=policy.place_strategy,
        eps=policy.eps,
    )

    if chosen is None:
        decision = PlacementDecision(
            item_id=item.id,
            knapsack_id=None,
            reason="no_feasible_knapsack",
        )
        if tracker is not None:
            # snapshot current remaining capacities
            remaining = {k.id: float(k.remaining) for k in state.knapsacks}
            tracker.append_placement_decision(
                item=item,
                decision=decision,
                remaining_after=remaining,
            )
        return decision

    # 2) Commit the placement (mutate runtime)
    ok = chosen.place(item, eps=policy.eps)
    # Under normal use, 'ok' is True because heuristics checked feasibility.
    if not ok:
        decision = PlacementDecision(
            item_id=item.id,
            knapsack_id=None,
            reason="commit_failed_unexpected",
        )
        if tracker is not None:
            remaining = {k.id: float(k.remaining) for k in state.knapsacks}
            tracker.append_placement_decision(
                item=item,
                decision=decision,
                remaining_after=remaining,
            )
        return decision

    # 3) Emit decision + optional logging
    decision = PlacementDecision(
        item_id=item.id,
        knapsack_id=chosen.id,
        reason=policy.place_strategy,
    )
    if tracker is not None:
        remaining = {k.id: float(k.remaining) for k in state.knapsacks}
        tracker.append_placement_decision(
            item=item,
            decision=decision,
            remaining_after=remaining,
        )
    return decision


# Convenience alias mirroring naming used elsewhere
run_placement_step = place_next
