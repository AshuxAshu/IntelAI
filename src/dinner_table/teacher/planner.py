"""Vertical-corridor motion planner with keep-out and mutual-exclusion zone checks.

The same zone model the runtime executor reuses at deployment: no teacher
trajectory may drive an end-effector site into a keep-out box or into the
other arm's exclusive active zone. Corridor blocking is a skill failure
(``CorridorBlocked``), never a crash.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count

import mujoco
import numpy as np

from dinner_table.contracts.geometry import (
    ARM_A_ZONE,
    ARM_B_ZONE,
    CONTROL_HZ,
    SHARED_ZONE,
    TABLE_TOP_HEIGHT,
)
from dinner_table.teacher.ik import ik_above, solve_ik
from dinner_table.teacher.kinematics import arm_q, set_arm_q, site_pose

# Amendment 1: the real SO-101 cannot hold a top-down approach higher than
# ~10 cm above the table, so both the corridor hovers and the detour lift use
# the measured hover envelope (the reference teacher hovers 2.5-7 cm).
SAFE_LIFT_HEIGHT = TABLE_TOP_HEIGHT + 0.10
HOVER_HEIGHTS = (0.04, 0.06)
DETOUR_SAMPLES = 32

ARM_ZONES = {"A": ARM_A_ZONE, "B": ARM_B_ZONE}
OTHER_ARM = {"A": "B", "B": "A"}


class CorridorBlocked(Exception):
    """Raised when no keep-out-free corridor or detour exists for a motion."""

    def __init__(self, arm: str, reason: str) -> None:
        self.arm = arm
        self.reason = reason
        super().__init__(f"corridor blocked for arm {arm}: {reason}")


@dataclass(frozen=True)
class Box:
    """Axis-aligned keep-out box; unset bounds are unbounded."""

    name: str
    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None
    z_min: float | None = None  # shorthand: unbounded below up to z_min

    def contains(self, pos: np.ndarray) -> bool:
        pos = np.asarray(pos, dtype=np.float64)
        if self.z_min is not None and pos[2] <= self.z_min:
            return True
        if self.x is not None and not (self.x[0] <= pos[0] <= self.x[1]):
            return False
        if self.y is not None and not (self.y[0] <= pos[1] <= self.y[1]):
            return False
        if self.z is not None and not (self.z[0] <= pos[2] <= self.z[1]):
            return False
        return self.x is not None or self.y is not None or self.z is not None


# The bounded cabinet box is checked before the unbounded table box so a
# point inside the cabinet reports cabinet_body, not table_surface. The
# cabinet is a tabletop caddy on the arm-A side (Amendment 1): x -0.39..-0.09,
# y 0.02..0.20, z from the tabletop to just above the roof.
KEEP_OUTS = [
    Box(
        "cabinet_body",
        x=(-0.39, -0.09),
        y=(0.02, 0.20),
        z=(TABLE_TOP_HEIGHT, TABLE_TOP_HEIGHT + 0.11),
    ),
    Box("table_surface", z_min=TABLE_TOP_HEIGHT - 0.005),
]


def in_keepout(pos: np.ndarray) -> str | None:
    """Return the keep-out box name containing pos, or None if clear."""
    for box in KEEP_OUTS:
        if box.contains(pos):
            return box.name
    return None


def in_shared_zone(pos: np.ndarray) -> bool:
    """True if the 2-D position lies inside the shared handover zone."""
    pos = np.asarray(pos, dtype=np.float64)
    return SHARED_ZONE[0] <= pos[0] <= SHARED_ZONE[1] and SHARED_ZONE[2] <= pos[1] <= SHARED_ZONE[3]


def in_other_zone(arm: str, pos: np.ndarray) -> bool:
    """True if pos lies in the *other* arm's zone outside the shared zone."""
    other = OTHER_ARM[arm]
    zone = ARM_ZONES[other]
    pos = np.asarray(pos, dtype=np.float64)
    in_zone = zone[0] <= pos[0] <= zone[1] and zone[2] <= pos[1] <= zone[3]
    return in_zone and not in_shared_zone(pos)


def corridor_violation(arm: str, pos: np.ndarray) -> str | None:
    """Return a violation description for an end-effector position, or None."""
    keepout = in_keepout(pos)
    if keepout is not None:
        return keepout
    if in_other_zone(arm, pos):
        return f"arm_{OTHER_ARM[arm]}_zone"
    return None


def _fk_site(scene, arm: str, q: np.ndarray) -> np.ndarray:
    set_arm_q(scene.data, arm, q)
    mujoco.mj_forward(scene.model, scene.data)
    pos, _ = site_pose(scene.data, f"{arm}.ee")
    return pos


def _waypoint_ok(scene, arm: str, q: np.ndarray) -> bool:
    return corridor_violation(arm, _fk_site(scene, arm, q)) is None


def _segment_ok(scene, arm: str, q_a: np.ndarray, q_b: np.ndarray) -> bool:
    for i in range(DETOUR_SAMPLES + 1):
        alpha = i / DETOUR_SAMPLES
        if not _waypoint_ok(scene, arm, (1.0 - alpha) * q_a + alpha * q_b):
            return False
    return True


def plan_corridor(scene, arm: str, q_start: np.ndarray, q_goal: np.ndarray) -> list[np.ndarray]:
    """Plan a vertical-corridor path: current -> hover-above(0.04) -> goal -> hover-above(0.06).

    Any waypoint whose FK site position enters a keep-out or the other arm's
    active zone is rejected; on rejection the planner falls back to a
    joint-space detour that lifts to SAFE_LIFT_HEIGHT first. ``CorridorBlocked``
    is raised only if even the detour violates.
    """
    model, data = scene.model, scene.data
    q_start = np.asarray(q_start, dtype=np.float64).copy()
    q_goal = np.asarray(q_goal, dtype=np.float64).copy()
    qpos_backup = np.array(data.qpos, dtype=np.float64)
    try:
        goal_pos = _fk_site(scene, arm, q_goal)
        goal_pose = (goal_pos, np.array([0.0, 0.0, -1.0]))

        hovers: list[np.ndarray] = []
        hover_failure: str | None = None
        for height in HOVER_HEIGHTS:
            try:
                hovers.append(ik_above(model, data, arm, goal_pose, height))
            except Exception as exc:  # noqa: BLE001 - any IK failure triggers the detour path
                hover_failure = f"hover IK at +{height:.2f} failed: {exc}"
                break

        if hover_failure is None:
            waypoints = [q_start, hovers[0], q_goal, hovers[1]]
            chosen = [q_start, hovers[0], q_goal, hovers[1]]
            if all(_waypoint_ok(scene, arm, q) for q in chosen):
                return waypoints

        # Joint-space detour: lift to a safe height first, then interpolate to the goal.
        start_pos = _fk_site(scene, arm, q_start)
        lift_xy = np.array([start_pos[0], start_pos[1], SAFE_LIFT_HEIGHT], dtype=np.float64)
        approach_dn = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        q_lift = None
        for lift_target in (
            lift_xy,
            np.array([goal_pos[0], goal_pos[1], SAFE_LIFT_HEIGHT], dtype=np.float64),
        ):
            try:
                candidate = solve_ik(
                    model, data, f"{arm}.ee", lift_target, approach_dn, arm_q(data, arm)
                )
            except Exception:  # noqa: BLE001 - try the next lift anchor
                continue
            if _waypoint_ok(scene, arm, candidate):
                q_lift = candidate
                break
        if q_lift is None:
            raise CorridorBlocked(arm, "no safe lift configuration at the safe lift height")

        if not _waypoint_ok(scene, arm, q_goal):
            raise CorridorBlocked(arm, "goal itself violates a keep-out or the other arm's zone")
        if not _segment_ok(scene, arm, q_lift, q_goal):
            raise CorridorBlocked(arm, "joint-space detour still crosses a keep-out")
        return [q_start, q_lift, q_goal]
    finally:
        data.qpos[:] = qpos_backup
        mujoco.mj_forward(model, data)


_JOINT_VEL_LIMITS = np.array([2.0, 2.0, 2.5, 3.0, 3.0], dtype=np.float64)


def follow(scene, arm: str, q_waypoints, speed: float = 1.0) -> Iterator[np.ndarray]:
    """Interpolate in joint space at CONTROL_HZ under per-joint velocity limits.

    Yields a full 12-dim merged target each tick: the acting arm moves, the
    other arm holds its current joint values (grippers hold their current
    normalized aperture).
    """
    q_now = arm_q(scene.data, arm)
    held = scene.qpos_12()
    vel = _JOINT_VEL_LIMITS * float(speed)
    dt = 1.0 / CONTROL_HZ
    max_step = vel * dt

    for waypoint in q_waypoints:
        q_target = np.asarray(waypoint, dtype=np.float64)
        delta = np.abs(q_target - q_now)
        ticks = max(1, int(np.ceil(np.max(delta / max_step))))
        for t in count(1):
            alpha = t / ticks
            q_interp = (1.0 - alpha) * q_now + alpha * q_target
            action = held.copy()
            if arm == "A":
                action[0:5] = q_interp
            else:
                action[6:11] = q_interp
            yield action
            if t == ticks:
                break
        q_now = q_target
