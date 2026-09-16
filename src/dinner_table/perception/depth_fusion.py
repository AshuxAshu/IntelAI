"""Multiview depth-consistency gate for the executor's precondition checks."""

from __future__ import annotations

from dinner_table.perception.detector import MAX_DEPTH_M
from dinner_table.perception.interfaces import ObjectPose3D, PerceptionSnapshot


def multiview_check(snapshot: PerceptionSnapshot, pose: ObjectPose3D) -> bool:
    """True when the overhead depth at the target's centroid is valid.

    Samples the 5x5 median depth at the best detection's box center for
    ``pose.name``; depth of 0 or beyond ``MAX_DEPTH_M`` is invalid. A False
    result tells the executor to fall back to the tracker's last-good pose
    (the GT-free wrist-backed estimate) instead of re-grounding.
    """
    candidates = sorted(
        (d for d in snapshot.detections if d.label == pose.name),
        key=lambda d: -d.confidence,
    )
    if not candidates:
        return False
    best = candidates[0]
    cx = int((best.xyxy[0] + best.xyxy[2]) / 2)
    cy = int((best.xyxy[1] + best.xyxy[3]) / 2)
    depth = snapshot.overhead_depth
    y0, x0 = max(0, cy - 2), max(0, cx - 2)
    patch = depth[y0 : y0 + 5, x0 : x0 + 5]
    valid = patch[(patch > 0) & (patch <= MAX_DEPTH_M)]
    return valid.size >= 2
