# -*- coding: utf-8 -*-
"""
Run-time state containers for the MKP planning pipeline.

This module defines:
  - RuntimeKnapsack: mutable knapsack used during a run
  - SelectionState:  immutable input snapshot (items + knapsack specs)
  - RuntimeState:    mutable working state created from SelectionState

Notes
-----
- Business (timeless) entities live in `business_objects/`:
  * business_objects.items.Item
  * business_objects.knapsacks.KnapsackSpec
- Planning/runtime entities (below) are specific to executing a solve.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from src.business_objects.items import Item
from src.business_objects.knapsacks import KnapsackSpec


# ----------------------------
# Exceptions (planning layer)
# ----------------------------

class StateValidationError(ValueError):
    """Raised when in-memory planning state violates domain constraints."""


# ----------------------------
# Runtime (mutable) container
# ----------------------------

@dataclass
class RuntimeKnapsack:
    """
    Mutable knapsack used during planning.

    Attributes
    ----------
    id : str
        Identifier (must match its KnapsackSpec).
    capacity : float
        Capacity limit (copied from spec).
    remaining : float
        Remaining capacity; updated as items are placed.
    """
    id: str
    capacity: float
    remaining: float = field(init=False)

    def __post_init__(self) -> None:  # type: ignore[override]
        # Default to full capacity if constructed directly.
        self.remaining = self.capacity

    @classmethod
    def from_spec(cls, spec: KnapsackSpec) -> "RuntimeKnapsack":
        rk = cls(id=spec.id, capacity=spec.capacity)
        rk.remaining = spec.capacity
        return rk

    def reset(self) -> None:
        """Restore remaining capacity to full capacity."""
        self.remaining = self.capacity

    def can_fit(self, item: Item, eps: float = 1e-12) -> bool:
        """Check if item weight fits (within tolerance)."""
        return item.weight <= self.remaining + eps

    def place(self, item: Item, eps: float = 1e-12) -> bool:
        """
        Attempt to place the item. Returns True if committed, False otherwise.
        No overfill is allowed (beyond eps tolerance).
        """
        if self.can_fit(item, eps=eps):
            self.remaining -= item.weight
            return True
        return False


# ----------------------------
# Immutable input snapshot
# ----------------------------

@dataclass(frozen=True)
class SelectionState:
    """
    Immutable problem input for a planning run.

    Attributes
    ----------
    items : list[Item]
        All available items (each can be placed at most once).
    knapsacks : list[KnapsackSpec]
        Knapsack templates (id + capacity).
    """
    items: List[Item]
    knapsacks: List[KnapsackSpec]

    def __post_init__(self) -> None:  # type: ignore[override]
        # Validate unique IDs and basic cardinality (optional but helpful).
        seen_items: set[str] = set()
        for it in self.items:
            if not it.id:
                raise StateValidationError("Item.id must be non-empty.")
            if it.id in seen_items:
                raise StateValidationError(f"Duplicate Item.id: {it.id}")
            seen_items.add(it.id)

        seen_knaps: set[str] = set()
        for ks in self.knapsacks:
            if not ks.id:
                raise StateValidationError("KnapsackSpec.id must be non-empty.")
            if ks.id in seen_knaps:
                raise StateValidationError(f"Duplicate KnapsackSpec.id: {ks.id}")
            seen_knaps.add(ks.id)

    def to_runtime(self) -> "RuntimeState":
        """
        Create a fresh, mutable RuntimeState to execute planning/placement.
        Items are immutable, so we share references safely.
        """
        return RuntimeState(
            items=self.items,
            knapsacks=[RuntimeKnapsack.from_spec(ks) for ks in self.knapsacks],
        )


# ----------------------------
# Mutable working state
# ----------------------------

@dataclass
class RuntimeState:
    """
    Mutable state used during the planning run.

    Attributes
    ----------
    items : list[Item]
        Treat as read-only.
    knapsacks : list[RuntimeKnapsack]
        Mutable containers with remaining capacity.
    """
    items: List[Item]
    knapsacks: List[RuntimeKnapsack]

    def reset_knapsacks(self) -> None:
        """Restore all runtime knapsacks to full capacity."""
        for k in self.knapsacks:
            k.reset()

    def by_knapsack_id(self) -> Dict[str, RuntimeKnapsack]:
        """Convenience lookup table by knapsack id."""
        return {k.id: k for k in self.knapsacks}
