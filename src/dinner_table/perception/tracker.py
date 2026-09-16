"""Temporal pose filtering + gripper-state fusion (§10.4 Tracker protocol)."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import numpy as np

from dinner_table.perception.interfaces import ObjectPose3D

HISTORY = 5
HOLD_S = 1.0
HELD_RADIUS_M = 0.06
HELD_APERTURE = 0.6


class ObjectTracker:
    """Median filter over the last 5 poses per object with miss holdover.

    `ee_provider` returns ``{arm: (3,) xyz}`` end-effector positions for
    `fuse_held`; without one, `held_by` is always None.
    """

    def __init__(
        self,
        ee_provider: Callable[[], dict[str, np.ndarray]] | None = None,
        *,
        history: int = HISTORY,
        hold_s: float = HOLD_S,
    ) -> None:
        self._ee_provider = ee_provider
        self._history = history
        self._hold_s = hold_s
        self._positions: dict[str, deque] = {}
        self._last: dict[str, ObjectPose3D] = {}
        self._seen: dict[str, float] = {}

    def _filtered(self, name: str, held_by: str | None) -> ObjectPose3D:
        median = np.median(np.stack(list(self._positions[name])), axis=0)
        pose = ObjectPose3D(
            name=name, position=np.asarray(median, dtype=np.float64), held_by=held_by
        )
        self._last[name] = pose
        return pose

    def update(self, poses: dict[str, ObjectPose3D]) -> dict[str, ObjectPose3D]:
        """Ingest freshly grounded poses; return the temporally filtered set."""
        now = time.monotonic()
        out: dict[str, ObjectPose3D] = {}
        for name, pose in poses.items():
            self._positions.setdefault(name, deque(maxlen=self._history)).append(
                np.asarray(pose.position, dtype=np.float64)
            )
            self._seen[name] = now
            out[name] = self._filtered(name, pose.held_by)
        for name, seen in list(self._seen.items()):
            if name in out:
                continue
            if now - seen <= self._hold_s:
                out[name] = self._last[name]
            else:
                del self._seen[name]
                del self._positions[name]
                del self._last[name]
        return out

    def fuse_held(
        self, poses: dict[str, ObjectPose3D], gripper_apertures: dict[str, float]
    ) -> dict[str, ObjectPose3D]:
        """Set `held_by` where an object is within 6 cm of a closed gripper."""
        ee = self._ee_provider() if self._ee_provider is not None else {}
        fused = {}
        for name, pose in poses.items():
            held = None
            for arm, aperture in gripper_apertures.items():
                if aperture >= HELD_APERTURE or arm not in ee:
                    continue
                if float(np.linalg.norm(pose.position - ee[arm])) <= HELD_RADIUS_M:
                    held = arm
                    break
            fused[name] = ObjectPose3D(name=name, position=pose.position, held_by=held)
        return fused
