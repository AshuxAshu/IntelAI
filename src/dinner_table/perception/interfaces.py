"""Perception contracts. The Detector protocol is implemented twice, both real:
- UltralyticsDetector  (perception/detector.py, Dev A) - torch backend used
  during YOLO training validation and on hosts without OpenVINO.
- OpenvinoDetector     (runtime/engines.py, Dev B) - OpenVINO IR backend used
  in the deployed Intel pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

OBJECT_LABELS = (
    "plate",
    "mug",
    "bottle",
    "spoon_1",
    "spoon_2",
    "fork_1",
    "fork_2",
    "drawer_open",
    "drawer_closed",
)


@dataclass(frozen=True)
class Detection:
    label: str  # one of OBJECT_LABELS
    xyxy: tuple[float, float, float, float]  # pixels in the overhead frame
    confidence: float  # 0..1


@dataclass(frozen=True)
class PerceptionSnapshot:
    """One overhead-camera observation plus detections."""

    overhead_rgb: np.ndarray  # (480, 640, 3) uint8
    overhead_depth: np.ndarray  # (480, 640) float32, meters
    detections: tuple[Detection, ...]
    timestamp: float  # time.monotonic()


@dataclass(frozen=True)
class ObjectPose3D:
    name: str
    position: np.ndarray  # (3,) float64, world frame, meters
    held_by: str | None  # "A" | "B" | None, fused from gripper state


@runtime_checkable
class Detector(Protocol):
    def detect(self, overhead_rgb: np.ndarray) -> list[Detection]:
        """Run detection on one (480, 640, 3) uint8 overhead frame."""

    def ground(self, snapshot: PerceptionSnapshot, target: str) -> ObjectPose3D | None:
        """3D-pose grounding of `target` from the snapshot's detections + depth.
        Returns None when the target is not confidently visible."""


@runtime_checkable
class Tracker(Protocol):
    """Temporal pose filtering + gripper-state fusion (implemented by Dev A's
    ObjectTracker in perception/tracker.py, A13; scripted in executor tests)."""

    def update(self, poses: dict[str, ObjectPose3D]) -> dict[str, ObjectPose3D]:
        """Ingest freshly grounded poses; return the temporally filtered set
        (median filtering + confidence gating per the A13 spec)."""

    def fuse_held(
        self, poses: dict[str, ObjectPose3D], gripper_apertures: dict[str, float]
    ) -> dict[str, ObjectPose3D]:
        """Return poses with held_by fused from gripper apertures (object within
        6 cm of an arm's end-effector and that gripper aperture < 0.6)."""
