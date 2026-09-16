"""Collision-checked pouring arcs using the measured bottle-to-gripper pose.

The rigid grasp is a planning approximation, not a constraint on live physics.
Execution must independently verify contact forces and actual flow geometry.
"""

import logging
from dataclasses import dataclass
from itertools import pairwise

import mujoco
import numpy as np
from scipy.optimize import least_squares

from dinner_table.scene.objects import BOTTLE_MOUTH_Z
from dinner_table.teacher.context import SkillFailed
from dinner_table.teacher.kinematics import arm_q, joint_limits, set_arm_q, site_pose

POUR_ARC_LIFT_M = 0.09


@dataclass
class PourPath:
    joints: np.ndarray


class _PourGeometry:
    """Measured rigid-grasp approximation and scratch-only contact checks."""

    def __init__(self, ctx, arm: str, bottle: str) -> None:
        self.ctx, self.arm, self.bottle = ctx, arm, bottle
        self.model = ctx.model
        self.data = mujoco.MjData(ctx.model)
        self.data.qpos[:] = ctx.data.qpos
        self.data.ctrl[:] = ctx.data.ctrl
        self.data.qpos[ctx._jadr[arm]["gripper"]] = ctx.data.ctrl[ctx._gact[arm]]
        site, rotation = site_pose(ctx.data, f"{arm}.ee")
        base, quat = ctx.object(bottle)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quat)
        self.local_base = rotation.T @ (base - site)
        self.local_rotation = rotation.T @ matrix.reshape(3, 3)
        self.body = ctx.model.body(bottle).id
        self.address = int(ctx.model.jnt_qposadr[ctx.model.body_jntadr[self.body]])

    def pose(self, q):
        set_arm_q(self.data, self.arm, q)
        mujoco.mj_kinematics(self.model, self.data)
        position, rotation = site_pose(self.data, f"{self.arm}.ee")
        obj_rotation = rotation @ self.local_rotation
        base = position + rotation @ self.local_base
        return base + BOTTLE_MOUTH_Z * obj_rotation[:, 2], obj_rotation, base

    def clear(self, q) -> bool:
        """No contact between the weld (arm + vessel) and anything external."""
        _, rotation, base = self.pose(q)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        self.data.qpos[self.address:self.address + 7] = np.r_[base, quat]
        mujoco.mj_forward(self.model, self.data)
        geoms = self.ctx._arm_geoms[self.arm]
        for contact in self.data.contact[:self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = (int(self.model.geom_bodyid[g]) for g in (g1, g2))
            for acting, other in ((g1, b2), (g2, b1)):
                name = self.model.body(other).name
                if acting in geoms and not (name.startswith(f"{self.arm}.") or name == self.bottle):
                    logging.getLogger(__name__).debug(
                        "pour weld blocked: %s vs %s dist=%.6f",
                        self.model.body(int(self.model.geom_bodyid[acting])).name,
                        name, contact.dist)
                    return False
            if self.body in (b1, b2):
                other = b2 if b1 == self.body else b1
                if not self.model.body(other).name.startswith(f"{self.arm}."):
                    logging.getLogger(__name__).debug(
                        "pour vessel blocked vs %s dist=%.6f",
                        self.model.body(other).name, contact.dist)
                    return False
        return True

def plan_pour(ctx, arm: str, bottle: str, mouth_goal: np.ndarray, tilt_deg: float) -> PourPath:
    """Search spill-angle arcs until one passes every contact audit."""
    receiver = "B" if arm == "A" else "A"
    geometry = _PourGeometry(ctx, arm, bottle)
    start_q = arm_q(ctx.data, arm)
    lower, upper = joint_limits(ctx.model, arm)
    lower, upper = lower + 0.015, upper - 0.015
    base, quat = ctx.object(bottle)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    start_up = matrix.reshape(3, 3)[:, 2].copy()
    start_mouth = base + BOTTLE_MOUTH_Z * start_up
    wrist = site_pose(ctx.data, f"{receiver}.ee")[0]
    toward = np.array([wrist[0] - mouth_goal[0], wrist[1] - mouth_goal[1]])
    norm = float(np.linalg.norm(toward))
    toward = toward / norm if norm > 1e-6 else np.array([1.0, 0.0])
    base_azimuth = float(np.arctan2(toward[1], toward[0]))
    tilt = np.radians(tilt_deg)
    for height in (0.0, 0.05, 0.10, 0.15, 0.20):
        goal = np.asarray(mouth_goal, dtype=np.float64) + np.array([0.0, 0.0, height])
        for offset in np.radians((0.0, 40.0, -40.0, 80.0, -80.0, 120.0, -120.0, 180.0)):
            azimuth = base_azimuth + offset
            up_end = np.array([np.sin(tilt) * np.cos(azimuth),
                               np.sin(tilt) * np.sin(azimuth), np.cos(tilt)])
            axis = np.cross([0.0, 0.0, 1.0], up_end)
            axis /= np.linalg.norm(axis)
            points = [start_q]
            blocked = None
            for fraction in np.linspace(0.0, 1.0, 41)[1:]:
                angle = tilt * fraction
                up = (np.array([0.0, 0.0, 1.0]) * np.cos(angle)
                      + np.cross(axis, [0.0, 0.0, 1.0]) * np.sin(angle))
                target = start_mouth * (1.0 - fraction) + goal * fraction
                target[2] += POUR_ARC_LIFT_M * np.sin(np.pi * fraction)

                def residual(q, target=target, up=up, previous=points[-1]):
                    mouth, rotation, _ = geometry.pose(q)
                    return np.r_[10.0 * (mouth - target), rotation[:, 2] - up,
                                 0.001 * (q - previous)]

                # Live tracking can overshoot the inset planning bounds.
                seed = np.clip(points[-1], lower, upper)
                result = least_squares(residual, seed, bounds=(lower, upper),
                                       max_nfev=200)
                mouth, rotation, _ = geometry.pose(result.x)
                if (np.linalg.norm(mouth - target) > 0.002
                        or np.linalg.norm(rotation[:, 2] - up) > 0.05):
                    blocked = f"ik_unreachable at fraction {fraction:.2f}"
                    break
                if not geometry.clear(result.x):
                    blocked = f"weld contact at fraction {fraction:.2f}"
                    break
                points.append(result.x)
            if blocked is not None:
                logging.getLogger(__name__).debug(
                    "pour arc rejected height=%.2f offset=%.0fdeg: %s",
                    height, np.degrees(offset), blocked)
                continue
            dense = np.concatenate([
                np.linspace(a, b, max(3, int(np.max(np.abs(b - a)) / 0.015) + 2))
                for a, b in pairwise(points)
            ])
            try:
                ctx.check_path(arm, dense)
            except SkillFailed as exc:
                blocked = f"arm path blocked: {exc.cause}"
                continue
            if any(not geometry.clear(q) for q in dense):
                blocked = "vessel contact on densified path"
                continue
            return PourPath(dense)
    raise SkillFailed("pour", "plan", "no_clear_arc")


def plan_return(ctx, arm: str, bottle: str) -> PourPath:
    """Search level-mouth retreats while recovering the measured bottle axis."""
    geometry = _PourGeometry(ctx, arm, bottle)
    lower, upper = joint_limits(ctx.model, arm)
    lower, upper = lower + 0.015, upper - 0.015
    start_q = arm_q(ctx.data, arm)
    start_mouth, rotation, _ = geometry.pose(start_q)
    start_up = rotation[:, 2].copy()
    world_up = np.array([0.0, 0.0, 1.0])
    axis = np.cross(start_up, world_up)
    norm = float(np.linalg.norm(axis))
    angle = float(np.arccos(np.clip(start_up[2], -1.0, 1.0)))
    if norm < 1e-6:
        if start_up[2] < 0:
            raise SkillFailed("pour", "return", "inverted_bottle")
        return PourPath(np.vstack([start_q, start_q]))
    axis /= norm
    receiver = "B" if arm == "A" else "A"
    wrist = site_pose(ctx.data, f"{receiver}.ee")[0]
    away = start_mouth[:2] - wrist[:2]
    azimuth = float(np.arctan2(away[1], away[0]))
    for distance in (0.05, 0.10, 0.15):
        for offset in np.radians((0.0, 45.0, -45.0, 90.0, -90.0)):
            retreat = distance * np.array([np.cos(azimuth + offset),
                                           np.sin(azimuth + offset), 0.0])
            points = [start_q]
            for fraction in np.linspace(0.0, 1.0, 41)[1:]:
                up = (start_up * np.cos(angle * fraction)
                      + np.cross(axis, start_up) * np.sin(angle * fraction))
                target = start_mouth + retreat * fraction

                def residual(q, target=target, up=up, previous=points[-1]):
                    mouth, rot, _ = geometry.pose(q)
                    return np.r_[10.0 * (mouth - target), rot[:, 2] - up,
                                 0.001 * (q - previous)]

                # Live tracking can overshoot the inset planning bounds.
                seed = np.clip(points[-1], lower, upper)
                result = least_squares(residual, seed, bounds=(lower, upper),
                                       max_nfev=200)
                mouth, rot, _ = geometry.pose(result.x)
                if (np.linalg.norm(mouth - target) > 0.002
                        or np.linalg.norm(rot[:, 2] - up) > 0.05
                        or not geometry.clear(result.x)):
                    break
                points.append(result.x)
            if len(points) != 41:
                continue
            dense = np.concatenate([
                np.linspace(a, b, max(3, int(np.max(np.abs(b - a)) / 0.015) + 2))
                for a, b in pairwise(points)
            ])
            if any(not geometry.clear(q) for q in dense):
                continue
            try:
                ctx.check_path(arm, dense)
            except SkillFailed:
                continue
            return PourPath(dense)
    raise SkillFailed("pour", "return", "no_clear_recovery")
