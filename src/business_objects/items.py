# -*- coding: utf-8 -*-
"""
Item model for MKP.
"""

from __future__ import annotations
from dataclasses import dataclass
from .errors import StateValidationError


@dataclass(frozen=True)
class Item:
    """
    An item that can be assigned to at most one knapsack.

    Attributes
    ----------
    id : str
        Unique identifier.
    profit : float
        Nonnegative objective contribution if selected.
    weight : float
        Nonnegative weight (capacity consumption).
    """
    id: str
    profit: float
    weight: float

    def __post_init__(self) -> None:  # type: ignore[override]
        if not self.id:
            raise StateValidationError("Item.id must be non-empty.")
        if self.profit < 0:
            raise StateValidationError(f"Item[{self.id}] profit must be >= 0.")
        if self.weight < 0:
            raise StateValidationError(f"Item[{self.id}] weight must be >= 0.")
