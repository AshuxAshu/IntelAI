"""Airborne mid-air handoff between dual SO-101 arms with force verification."""

from __future__ import annotations

from typing import Iterator

import mujoco
import numpy as np

from dinner_table.contracts.geometry import (
    HOME_JOINTS,
    TABLE_TOP_HEIGHT,
)
from dinner_table.teacher.context import JAW_FORCE_MIN, SkillFailed, TeacherContext
from dinner_table.teacher.ik import IKUnreachable
from dinner_table.teacher.kinematics import arm_q, site_pose
from dinner_table.teacher.skills import CARRY_GRIP_TORQUE, Skill

HANDOFF_AIR_XYZ = np.array([0.0, -0.24, TABLE_TOP_HEIGHT + 0.05], dtype=np.float64)
HANDOFF_TILTS = (0.0, 0.2618, 0.5236)


class AirborneHandoff(Skill):
    """Direct mid-air object transfer between arms without table relay."""

    phases = (
        "giver_present",
        "taker_approach",
        "taker_close",
        "dual_hold",
        "giver_release",
        "giver_retreat",
        "verify",
    )

    def __init__(self, object_name: str, from_arm: str, to_arm: str) -> None:
        super().__init__(from_arm, object_name)
        if to_arm not in ("A", "B") or to_arm == from_arm:
            raise ValueError(f"from_arm and to_arm must differ, got {from_arm} -> {to_arm}")
        self.from_arm = from_arm
        self.to_arm = to_arm

    def _tilt_toward_arm(self, tilt: float, arm: str) -> np.ndarray:
        sign = 1.0 if arm == "A" else -1.0
        return np.array([-sign * np.sin(tilt), 0.0, -np.cos(tilt)], dtype=np.float64)

    def _far_end_target(self, ctx: TeacherContext) -> np.ndarray:
        obj_pos, obj_quat = ctx.object(self.object_name)
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, np.asarray(obj_quat, dtype=np.float64))
        rot = rot.reshape(3, 3)

        if self.object_name == "bottle":
            return np.asarray(obj_pos, dtype=np.float64) - rot @ np.array([0.0, 0.0, 0.035], dtype=np.float64)

        axis_long = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        giver_site = site_pose(ctx.data, f"{self.from_arm}.ee")[0]
        extent = 0.025
        cands = [
            np.asarray(obj_pos, dtype=np.float64) + extent * axis_long,
            np.asarray(obj_pos, dtype=np.float64) - extent * axis_long,
        ]
        return max(cands, key=lambda c: float(np.linalg.norm(c - giver_site)))

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.from_arm) != self.object_name:
            raise SkillFailed("airborne_handoff", "giver_present", "giver_not_holding")

        carry_torque = CARRY_GRIP_TORQUE.get(self.object_name, 1.5)

        ctx.allowed[self.from_arm].add(self.object_name)
        ctx.allowed[self.to_arm].add(self.object_name)
        ctx.allowed[self.to_arm].update({
            f"{self.from_arm}.jaw_moving",
            f"{self.from_arm}.jaw_fixed",
            f"{self.from_arm}.wrist_roll",
            f"{self.from_arm}.wrist_flex",
        })
        ctx.allowed[self.from_arm].update({
            f"{self.to_arm}.jaw_moving",
            f"{self.to_arm}.jaw_fixed",
            f"{self.to_arm}.wrist_roll",
            f"{self.to_arm}.wrist_flex",
        })

        self._set_phase("giver_present")
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        q_present = None
        for tilt in HANDOFF_TILTS:
            try:
                approach = self._tilt_toward_arm(tilt, self.from_arm)
                q_present = ctx.plan_ik(self.from_arm, HANDOFF_AIR_XYZ, approach)
                break
            except IKUnreachable:
                continue

        if q_present is None:
            raise SkillFailed("airborne_handoff", "giver_present", "giver_ik_unreachable")

        yield from ctx.play_joint(self.from_arm, q_present, duration=3.0)
        ctx.latch_hold(self.from_arm)

        self._set_phase("taker_approach")
        yield from ctx.open_gripper(self.to_arm, 0.25)
        ctx.latch_hold(self.to_arm)

        target = self._far_end_target(ctx)
        hover = target + np.array([0.0, 0.0, 0.05], dtype=np.float64)

        q_hover = None
        q_target = None
        for tilt in HANDOFF_TILTS:
            try:
                approach = self._tilt_toward_arm(tilt, self.to_arm)
                q_hover = ctx.plan_ik(self.to_arm, hover, approach)
                q_target = ctx.plan_ik(self.to_arm, target, approach, seed=q_hover)
                break
            except IKUnreachable:
                continue

        if q_target is None or q_hover is None:
            raise SkillFailed("airborne_handoff", "taker_approach", "taker_ik_unreachable")

        with ctx.seating(self.from_arm):
            with ctx.grip_saturation(self.from_arm, carry_torque):
                yield from ctx.play_joint(self.to_arm, q_hover, duration=2.5)
                ctx.latch_hold(self.to_arm)
                yield from ctx.play_joint(self.to_arm, q_target, duration=2.0)
                ctx.latch_hold(self.to_arm)

                self._set_phase("taker_close")
                yield from ctx.close_gripper(
                    self.to_arm, carry_torque, object_name=self.object_name, q_hold=q_target
                )
                ctx.latch_hold(self.to_arm)

                fixed, moving = ctx.finger_forces(self.to_arm, self.object_name)
                if min(fixed, moving) < JAW_FORCE_MIN:
                    raise SkillFailed("airborne_handoff", "taker_close", "taker_grip_unverified")

                self._set_phase("dual_hold")
                ctx.begin_carry(self.to_arm, self.object_name, carry_torque)
                for _ in range(12):
                    yield ctx.action(self.to_arm, q_target, ctx._grip_now[self.to_arm])
                ctx.latch_hold(self.to_arm)

        self._set_phase("giver_release")
        ctx.end_carry(self.from_arm)
        yield from ctx.open_gripper(self.from_arm, 0.25)
        ctx.latch_hold(self.from_arm)

        self._set_phase("giver_retreat")
        giver_cur = site_pose(ctx.data, f"{self.from_arm}.ee")[0]
        retreat_pt = giver_cur + np.array([0.0, 0.0, 0.06], dtype=np.float64)
        try:
            q_retreat = ctx.plan_ik(self.from_arm, retreat_pt, down)
            yield from ctx.play_joint(self.from_arm, q_retreat, duration=1.5)
            ctx.latch_hold(self.from_arm)
        except IKUnreachable:
            pass

        yield from ctx.play_joint(
            self.from_arm, np.array(HOME_JOINTS[self.from_arm][:5]), duration=2.5
        )
        ctx.latch_hold(self.from_arm)

        self._set_phase("verify")
        if ctx.carrying.get(self.to_arm) != self.object_name:
            raise SkillFailed("airborne_handoff", "verify", "regrasp_missed")
        if ctx.carrying.get(self.from_arm) is not None:
            raise SkillFailed("airborne_handoff", "verify", "giver_still_holding")

        fixed_final, moving_final = ctx.finger_forces(self.to_arm, self.object_name)
        if min(fixed_final, moving_final) < JAW_FORCE_MIN:
            raise SkillFailed("airborne_handoff", "verify", "final_grip_lost")
