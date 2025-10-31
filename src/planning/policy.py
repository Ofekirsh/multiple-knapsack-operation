# -*- coding: utf-8 -*-
"""
Policy (configuration knobs) for the MKP planning pipeline.

Phase 1 (Select Next) now uses a direct feature-order list:
  - select_order: tuple[str, ...] | None
    The left→right priority of features. Allowed keys:
      * "profit"  (always descending; higher is better)
      * "ratio"   (profit/weight; always descending; higher is better)
      * "weight"  (descending = heavier first)
      * "-weight" (ascending  = lighter first)
      * "id"      (ascending; typically auto-appended by the selector as final tiebreaker)
    If None/empty, the selector defaults to ["ratio", "profit", "-weight"] and then appends "id".

Phase 2 (Place Next):
  - place_strategy: {"best_fit","first_fit","max_remaining"}
  - eps: feasibility tolerance

Run control:
  - seed: optional RNG seed for deterministic tie-breaking (if used elsewhere).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Policy:
    """
    Planning knobs (pure data holder).

    Attributes
    ----------
    # Phase 1 (Select Next)
    select_order : tuple[str, ...] | None
        Feature priority list. Examples:
          ("ratio", "profit", "-weight")
          ("profit", "-weight")
          ("-weight",)
        If None/empty, the selector will use its internal default and append "id".

    # Phase 2 (Place Next)
    place_strategy : str
        Knapsack ranking rule: "best_fit" | "first_fit" | "max_remaining".
    eps : float
        Feasibility tolerance for capacity checks.

    # Run control
    seed : int | None
        Optional RNG seed for deterministic tie-breaking.
    """
    # Phase 1
    select_order: Optional[Tuple[str, ...]] = None

    # Phase 2
    place_strategy: str = "best_fit"
    eps: float = 1e-12

    # Run control
    seed: Optional[int] = 123
