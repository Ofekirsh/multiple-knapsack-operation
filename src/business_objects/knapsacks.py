# -*- coding: utf-8 -*-
"""
Knapsack models for MKP.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .errors import StateValidationError
from .items import Item


@dataclass(frozen=True)
class KnapsackSpec:
    """
    Immutable knapsack template.

    Attributes
    ----------
    id : str
        Unique identifier.
    capacity : float
        Nonnegative capacity limit.
    """
    id: str
    capacity: float

    def __post_init__(self) -> None:  # type: ignore[override]
        if not self.id:
            raise StateValidationError("KnapsackSpec.id must be non-empty.")
        if self.capacity < 0:
            raise StateValidationError(f"KnapsackSpec[{self.id}] capacity must be >= 0.")


@dataclass
class Knapsack:
    """
    Runtime knapsack with mutable 'remaining' capacity.
    Created from KnapsackSpec for a specific run.
    """
    id: str
    capacity: float
    remaining: float = field(init=False)

    def __post_init__(self) -> None:  # type: ignore[override]
        # Initialize remaining to capacity for safety if constructed directly.
        self.remaining = self.capacity

    @classmethod
    def from_spec(cls, spec: KnapsackSpec) -> "Knapsack":
        k = cls(id=spec.id, capacity=spec.capacity)
        k.remaining = spec.capacity
        return k

    def reset(self) -> None:
        self.remaining = self.capacity

    def can_fit(self, item: Item, eps: float = 1e-12) -> bool:
        return item.weight <= self.remaining + eps

    def place(self, item: Item, eps: float = 1e-12) -> bool:
        """
        Try to place the item; returns True if committed, False otherwise.
        Does not allow overfill beyond eps tolerance.
        """
        if self.can_fit(item, eps=eps):
            self.remaining -= item.weight
            return True
        return False
