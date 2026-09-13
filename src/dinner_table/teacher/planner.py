"""Motion corridor planner and joint trajectory interpolation for teacher policy."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import (
    ARM_A_ZONE,
    ARM_B_ZONE,
    CONTROL_HZ,
    HOME_JOINTS,
    TABLE_TOP_HEIGHT,
)
from dinner_table.teacher.ik import IKUnreachable, ik_above
from dinner_table.teacher.kinematics import (
    KinematicsError,
    _arm_joint_names,
    set_arm_q,
    site_pose,
)

if TYPE_CHECKING:
    from dinner_table.scene.builder import Scene

JOINT_VELOCITY_LIMITS: dict[str, float] = {
    "shoulder_pan": 2.0,
    "shoulder_lift": 2.0,
    "elbow_flex": 2.5,
    "wrist_flex": 3.0,
    "wrist_roll": 3.0,
}

# Peak-to-average velocity ratio of the quintic blend s(tau) = 10t^3 - 15t^4 + 6t^5:
# s'(tau) peaks at 1.875 while its average over the segment is 1.0, so segment
# durations must be scaled by this factor for the caps to bound the true peak.
QUINTIC_PEAK_FACTOR = 1.875


class CorridorBlocked(DinnerTableError):
    """Exception raised when a planned corridor violates keep-out constraints."""


@dataclass(frozen=True)
class Box:
    """Axis-aligned 3D bounding box for workspace keep-out regions."""

    name: str
    x_range: tuple[float | None, float | None] | None = None
    y_range: tuple[float | None, float | None] | None = None
    z_range: tuple[float | None, float | None] | None = None

    def contains(self, point: np.ndarray) -> bool:
        """Check if a 3D point is inside this box volume."""
        if self.x_range is not None:
            if self.x_range[0] is not None and point[0] < self.x_range[0]:
                return False
            if self.x_range[1] is not None and point[0] > self.x_range[1]:
                return False
        if self.y_range is not None:
            if self.y_range[0] is not None and point[1] < self.y_range[0]:
                return False
            if self.y_range[1] is not None and point[1] > self.y_range[1]:
                return False
        if self.z_range is not None:
            if self.z_range[0] is not None and point[2] < self.z_range[0]:
                return False
            if self.z_range[1] is not None and point[2] > self.z_range[1]:
                return False
        return True


KEEP_OUTS: list[Box] = [
    Box("table_surface", x_range=None, y_range=None, z_range=(None, TABLE_TOP_HEIGHT - 0.005)),
    # The cabinet shell proper sits below the drawer tray; the volume at and
    # above the tray's rim must stay plannable so utensil picks from the
    # opened drawer are not rejected by the corridor check.
    Box("cabinet_body", x_range=(-0.28, 0.28), y_range=(0.53, 0.71), z_range=(0.0, 0.355)),
]


def _in_other_arm_private_zone(pos: np.ndarray, arm: str) -> bool:
    """Check if position intrudes into the other arm private zone."""
    if arm == "A":
        if pos[0] < -0.15 and ARM_B_ZONE[2] <= pos[1] <= ARM_B_ZONE[3]:
            return True
    else:
        if pos[0] > 0.15 and ARM_A_ZONE[2] <= pos[1] <= ARM_A_ZONE[3]:
            return True
    return False


def _violates_constraints(pos: np.ndarray, arm: str) -> bool:
    """Check whether a 3D site position violates keep-outs or zone exclusivity."""
    for box in KEEP_OUTS:
        if box.contains(pos):
            return True
    return _in_other_arm_private_zone(pos, arm)


def _quintic(tau: float) -> float:
    """Evaluate fifth-order minimum-jerk polynomial parameter."""
    t = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * (t**3) - 15.0 * (t**4) + 6.0 * (t**5)


def plan_corridor(
    scene: Scene,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
) -> list[np.ndarray]:
    """Plan a vertical-corridor waypoint sequence avoiding keep-outs and zone intrusion."""
    site_name = f"{arm}.ee"
    saved_qpos = np.array(scene.data.qpos, dtype=np.float64, copy=True)

    try:
        set_arm_q(scene.data, arm, q_start)
        mujoco.mj_forward(scene.model, scene.data)
        site_pose(scene.data, site_name)

        set_arm_q(scene.data, arm, q_goal)
        mujoco.mj_forward(scene.model, scene.data)
        pos_goal, rot_goal = site_pose(scene.data, site_name)

        candidate_waypoints: list[np.ndarray] = [q_start]
        can_build_standard = True

        try:
            q_hover_10 = ik_above(scene.model, scene.data, arm, (pos_goal, rot_goal), height=0.10)
            q_hover_15 = ik_above(scene.model, scene.data, arm, (pos_goal, rot_goal), height=0.15)
            candidate_waypoints = [q_start, q_hover_10, q_goal, q_hover_15]
        except (IKUnreachable, KinematicsError):
            can_build_standard = False

        if can_build_standard:
            has_violation = False
            for q_wp in candidate_waypoints:
                set_arm_q(scene.data, arm, q_wp)
                mujoco.mj_forward(scene.model, scene.data)
                pos_wp, _ = site_pose(scene.data, site_name)
                if _violates_constraints(pos_wp, arm):
                    has_violation = True
                    break
            if not has_violation:
                return candidate_waypoints

        home_q = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
        lift_alpha = 0.5
        q_lift_start = q_start + lift_alpha * (home_q - q_start)
        q_lift_goal = q_goal + lift_alpha * (home_q - q_goal)
        detour_waypoints = [q_start, q_lift_start, q_lift_goal, q_goal]

        for q_wp in detour_waypoints:
            set_arm_q(scene.data, arm, q_wp)
            mujoco.mj_forward(scene.model, scene.data)
            pos_wp, _ = site_pose(scene.data, site_name)
            if _violates_constraints(pos_wp, arm):
                raise CorridorBlocked(
                    f"Detour waypoint at {pos_wp.tolist()} violates keep-out constraints."
                )

        return detour_waypoints
    finally:
        scene.data.qpos[:] = saved_qpos
        mujoco.mj_forward(scene.model, scene.data)


def follow(
    scene: Scene,
    arm: str,
    q_waypoints: list[np.ndarray],
    speed: float = 1.0,
) -> Iterator[np.ndarray]:
    """Interpolate waypoints at 25 Hz respecting velocity limits and yielding 12-dim targets."""
    if len(q_waypoints) == 0:
        return
    for q_wp in q_waypoints:
        if not np.all(np.isfinite(q_wp)):
            raise CorridorBlocked(f"non-finite waypoint passed to follow: {q_wp.tolist()}")

    eff_speed = max(0.01, float(speed))
    dt = 1.0 / CONTROL_HZ
    names = _arm_joint_names(arm)
    vel_caps = np.array(
        [JOINT_VELOCITY_LIMITS[name.split(".")[1]] for name in names],
        dtype=np.float64,
    )

    full_action = np.array(scene.qpos_12(), dtype=np.float64, copy=True)
    if arm == "A":
        hinge_slice = slice(0, 5)
    else:
        hinge_slice = slice(6, 11)

    for seg_idx in range(len(q_waypoints) - 1):
        q_a = np.array(q_waypoints[seg_idx], dtype=np.float64)
        q_b = np.array(q_waypoints[seg_idx + 1], dtype=np.float64)
        delta_q = np.abs(q_b - q_a)

        time_per_joint = QUINTIC_PEAK_FACTOR * delta_q / (vel_caps * eff_speed)
        seg_duration = max(float(np.max(time_per_joint)), dt)
        num_ticks = max(1, int(np.ceil(seg_duration / dt)))

        for step in range(1, num_ticks + 1):
            tau = float(step) / float(num_ticks)
            blend = _quintic(tau)
            q_current = q_a + blend * (q_b - q_a)

            full_action[hinge_slice] = q_current
            yield full_action.copy()
