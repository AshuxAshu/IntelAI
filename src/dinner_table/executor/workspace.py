"""Workspace zoning: which arm owns which region of the table.

Zones are horizontal footprints from contracts.geometry, evaluated at any
height above the tabletop. A point inside the shared zone or inside both arm
zones is 'shared' - both arms can reach it and the executor must serialize
access through ZoneClaims.
"""

from __future__ import annotations

import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import ARM_A_ZONE, ARM_B_ZONE, SHARED_ZONE
from dinner_table.reasoning.schema import Step

CLAIMABLE_ZONES = ("A", "B", "shared")


class WorkspaceError(DinnerTableError):
    """Invalid workspace or zone usage."""


def _in_footprint(position: np.ndarray, zone: tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = zone
    return bool(x_min <= position[0] <= x_max and y_min <= position[1] <= y_max)


def zone_of(position: np.ndarray) -> str:
    """Classify a world-frame position as 'A' | 'B' | 'shared' | 'out'.

    Uses only x and y; accepts (2,) or (3,) arrays. 'shared' covers the shared
    zone plus the A/B footprint overlap (both arms reach those points).
    """
    in_a = _in_footprint(position, ARM_A_ZONE)
    in_b = _in_footprint(position, ARM_B_ZONE)
    if _in_footprint(position, SHARED_ZONE) or (in_a and in_b):
        return "shared"
    if in_a:
        return "A"
    if in_b:
        return "B"
    return "out"


def zone_for_step(step: Step) -> str:
    """Static zone of a step's goal region: 'A' | 'B' | 'shared' | 'out'.

    'out' means no exclusive zone is claimed: pick/home/retract have dynamic or
    empty goals, so the executor derives their live zone from perception via
    zone_of when a claim is needed. Drawer, handoff, pour, hold and
    non-placemat place targets conservatively claim the shared zone.
    """
    if step.skill in ("open_drawer", "close_drawer", "handoff", "hold", "pour"):
        return "shared"
    if step.skill == "place":
        if step.target == "placemat_1":
            return "A"
        if step.target == "placemat_2":
            return "B"
        return "shared"
    return "out"


class ZoneClaims:
    """Cross-arm mutual exclusion over the claimable zones ('A', 'B', 'shared')."""

    def __init__(self) -> None:
        self._held: dict[str, str] = {}

    def claim(self, arm: str, zone: str) -> bool:
        """Claim a zone for an arm; False if the other arm already holds it."""
        if arm not in ("A", "B"):
            raise WorkspaceError(f"arm must be 'A' or 'B', got {arm!r}")
        if zone not in CLAIMABLE_ZONES:
            return True
        holder = self._held.get(zone)
        if holder is not None and holder != arm:
            return False
        self._held[zone] = arm
        return True

    def release(self, arm: str) -> None:
        """Release every zone held by the arm."""
        self._held = {zone: holder for zone, holder in self._held.items() if holder != arm}
