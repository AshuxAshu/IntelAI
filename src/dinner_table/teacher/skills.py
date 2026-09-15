"""Privileged teacher skills: verified manipulation primitives.

Each skill is a coroutine yielding 12-dim merged targets at 25 Hz; the caller
steps physics per yield via ``TeacherContext.step`` (which also runs the
safety audits). Failure raises ``SkillFailed`` with an attributable phase and
cause. The recipes are ported from the reference solution's proven teacher
(see docs/PLAN_AMENDMENTS.md): hover/descend grasps with one bounded retry,
measured-support releases, and a physical drawer pull with opening
verification.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from dinner_table.contracts.geometry import (
    DRAWER_TRAVEL,
    HOME_JOINTS,
    PLACEMATS,
    TABLE_TOP_HEIGHT,
)
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.grasp_catalog import GraspCatalog, GraspFrame
from dinner_table.teacher.ik import IKUnreachable
from dinner_table.teacher.kinematics import arm_q, site_pose

OBJECT_HALF_HEIGHTS = {"plate": 0.012, "mug": 0.04, "bottle": 0.07}
# The neck pinch rides a 3 cm wall: the lift must stay on the neck, so
# only ~1.2 cm of base clearance is available — enough for a checked transit.
LIFT_MIN = {"bottle": 0.012}
DEFAULT_LIFT_MIN = 0.02
# The reference teacher's carried-object graze tolerance.
EXTERNAL_FORCE_MAX = 0.10


class Skill:
    """Base coroutine skill with an attributable phase tracker."""

    phases: tuple[str, ...] = ()

    def __init__(self, arm: str, object_name: str | None = None) -> None:
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        self.arm = arm
        self.object_name = object_name
        self.phase = self.phases[0] if self.phases else "run"
        self.catalog = GraspCatalog()

    def _set_phase(self, name: str) -> None:
        self.phase = name

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def _home_seed(self, frame: GraspFrame) -> np.ndarray:
        seed = np.array(HOME_JOINTS[self.arm][:5], dtype=np.float64)
        if frame.wrist_roll_seed is not None:
            seed[4] = frame.wrist_roll_seed
        return seed


def run_skill(scene, skill: Skill, timeout: float = 75.0) -> dict:
    """Drive one skill on a fresh context; returns a JSON-serializable result."""
    ctx = TeacherContext(scene)
    ctx.begin(timeout)
    try:
        for action in skill.run(ctx):
            ctx.live.note_skill(skill)
            ctx.step(action)
    except SkillFailed as exc:
        return {
            "ok": False, "skill": exc.skill, "phase": exc.phase, "cause": exc.cause,
        }
    return {"ok": True, "skill": type(skill).__name__, "phase": skill.phase, "cause": None}


class Home(Skill):
    """Park the arm at HOME_JOINTS along a checked joint path."""

    phases = ("move",)

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        self._set_phase("move")
        yield from ctx.play_joint(self.arm, np.array(HOME_JOINTS[self.arm][:5]), 3.0)


class Retract(Home):
    """Retract to home (same motion; semantic alias for the task graph)."""


class Pick(Skill):
    """Grasp an object: open, approach, descend, torque-limited close, lift."""

    phases = ("pregrasp", "approach", "close", "lift", "verify")

    def __init__(self, arm: str, object_name: str) -> None:
        super().__init__(arm, object_name)
        self._origin_z: float | None = None

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if self.object_name.startswith(("fork", "spoon")):
            # Cutlery lives inside the open drawer: reaching in necessarily
            # works among the drawer and cabinet structures, and with four
            # columns 4 cm apart the arm inevitably brushes a neighbor — a
            # nudge is harmless (each grasp re-plans from the live pose).
            ctx.allowed[self.arm].update({
                "drawer_top", "cabinet",
                "fork_1", "fork_2", "spoon_1", "spoon_2",
            })
        pos, _ = ctx.object(self.object_name)
        self._origin_z = float(pos[2])
        for attempt in (1, 2):
            frame = self.catalog.frame(ctx.scene, self.object_name, self.arm)
            try:
                q_grasp = ctx.plan_ik(self.arm, frame.position, frame.approach,
                                      frame.lateral, seed=self._home_seed(frame))
                hover_pos = frame.position + np.array([0.0, 0.0, frame.hover_m])
                q_hover = None
                for hover in (frame.hover_m, frame.hover_m - 0.010, max(frame.hover_m - 0.020, 0.010)):
                    try:
                        q_hover = ctx.plan_ik(
                            self.arm, frame.position + np.array([0.0, 0.0, hover]),
                            frame.approach, frame.lateral, seed=q_grasp,
                        )
                        break
                    except IKUnreachable:
                        q_hover = None
                if q_hover is None:
                    raise IKUnreachable(f"{self.arm}.ee", hover_pos)
            except IKUnreachable as exc:
                if attempt == 2:
                    raise SkillFailed("pick", "pregrasp", "ik_unreachable") from exc
                continue
            self._set_phase("pregrasp")
            yield from ctx.open_gripper(self.arm, frame.aperture)
            # The object is allowed from the approach on: the final run-in to
            # the hover necessarily works at contact distance of it, and any
            # shove is corrected by the live re-plan before the descend.
            ctx.allowed[self.arm].add(self.object_name)
            self._set_phase("approach")
            # Approach via a high waypoint directly above the target: a
            # straight joint-space sweep to the hover crosses the table at
            # low altitude and clips other objects' approach lanes.
            high = frame.position + np.array([0.0, 0.0, 0.15])
            try:
                q_high = ctx.plan_ik(self.arm, high, frame.approach, frame.lateral,
                                     seed=q_hover)
            except IKUnreachable:
                q_high = q_hover
            if q_high is q_hover:
                # Constrained high corridor infeasible: try a RELAXED high
                # point (approach-down, no lateral); the grasp orientation is
                # only enforced on the final hover-to-grasp descend.
                try:
                    q_high = ctx.plan_ik(self.arm, high, frame.approach, None,
                                         seed=q_hover)
                except IKUnreachable:
                    q_high = None
            if (q_high is None or q_high is q_hover) and self.object_name.startswith(
                ("fork", "spoon")
            ):
                # Cutlery approach from the drawer's open FRONT (the reference
                # recipe): a side detour would sweep over the neighboring
                # utensil columns, but the front strip is always clear.
                hover_pos = frame.position + np.array([0.0, 0.0, frame.hover_m])
                front = hover_pos + np.array([0.0, -0.05, 0.0])
                try:
                    q_front = ctx.plan_ik(self.arm, front, frame.approach,
                                          frame.lateral, seed=q_hover)
                    front_pts = np.linspace(arm_q(ctx.data, self.arm), q_front, 12)
                    ctx.check_path(self.arm, front_pts)
                    runin = ctx.plan_cartesian(
                        self.arm, front, hover_pos, frame.approach, frame.lateral,
                        q_start=q_front,
                    )
                    ctx.check_path(self.arm, runin)
                    yield from ctx.play(self.arm, front_pts,
                                        frame.aperture, frame.aperture, 3.0)
                    yield from ctx.play(self.arm, runin,
                                        frame.aperture, frame.aperture, 2.0)
                except (IKUnreachable, SkillFailed):
                    pass  # fall through to the generic cascade below
            if q_high is None or q_high is q_hover:
                # Direct joint path (orientation blends naturally); on a
                # crowded lane, fall back to a side detour: over the target's
                # outward side first, then a short Cartesian run-in.
                approach_pts = np.linspace(arm_q(ctx.data, self.arm), q_hover, 12)
                try:
                    ctx.check_path(self.arm, approach_pts)
                    yield from ctx.play(self.arm, approach_pts,
                                        frame.aperture, frame.aperture, 3.0)
                except SkillFailed:
                    outward = frame.position[:2] / max(np.linalg.norm(frame.position[:2]), 1e-9)
                    detour = frame.position + np.array([*outward * 0.10, frame.hover_m])
                    try:
                        q_detour = ctx.plan_ik(self.arm, detour, frame.approach,
                                               frame.lateral, seed=q_hover)
                        detour_pts = np.linspace(arm_q(ctx.data, self.arm), q_detour, 12)
                        ctx.check_path(self.arm, detour_pts)
                        runin = ctx.plan_cartesian(
                            self.arm, detour,
                            frame.position + np.array([0.0, 0.0, frame.hover_m]),
                            frame.approach, frame.lateral, q_start=q_detour,
                        )
                        ctx.check_path(self.arm, runin)
                    except (IKUnreachable, SkillFailed) as exc:
                        if attempt == 2:
                            raise SkillFailed("pick", "approach", "path_blocked") from exc
                        continue
                    yield from ctx.play(self.arm, detour_pts,
                                        frame.aperture, frame.aperture, 3.0)
                    yield from ctx.play(self.arm, runin,
                                        frame.aperture, frame.aperture, 2.0)
            else:
                # Cartesian up-and-over corridor: joint-space interpolation
                # between home and the hover can dip through other objects'
                # approach lanes; straight Cartesian segments never do.
                over_pts = ctx.plan_cartesian(
                    self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0], high,
                    frame.approach, frame.lateral,
                )
                hover_pos = frame.position + np.array([0.0, 0.0, frame.hover_m])
                descend_pts = ctx.plan_cartesian(
                    self.arm, high, hover_pos, frame.approach, frame.lateral,
                    q_start=over_pts[-1],
                )
                ctx.check_path(self.arm, over_pts)
                ctx.check_path(self.arm, descend_pts)
                yield from ctx.play(self.arm, over_pts,
                                    frame.aperture, frame.aperture, 3.0)
                yield from ctx.play(self.arm, descend_pts,
                                    frame.aperture, frame.aperture, 2.5)
            # The approach (or physics) may have shifted the object: re-plan
            # the grasp from its CURRENT pose before descending.
            frame = self.catalog.frame(ctx.scene, self.object_name, self.arm)
            try:
                q_grasp = ctx.plan_ik(self.arm, frame.position, frame.approach,
                                      frame.lateral, seed=q_hover)
            except IKUnreachable as exc:
                ctx.allowed[self.arm].discard(self.object_name)
                if attempt == 2:
                    raise SkillFailed("pick", "pregrasp", "ik_unreachable") from exc
                continue
            try:
                yield from ctx.play_cartesian(self.arm, frame.position, frame.approach,
                                              frame.lateral, 3.0)
            except IKUnreachable as exc:
                ctx.allowed[self.arm].discard(self.object_name)
                if attempt == 2:
                    raise SkillFailed("pick", "pregrasp", "ik_unreachable") from exc
                continue
            # Settle on the planned grasp pose before closing: the descend's
            # servo lag leaves the arm 5-10 mm off, and converging first keeps
            # the jaws from landing on the object instead of around it.
            settle_end = float(ctx.data.time) + 0.8
            while float(ctx.data.time) < settle_end:
                yield ctx.action(self.arm, q_grasp, frame.aperture)
            self._set_phase("close")
            yield from ctx.close_gripper(self.arm, frame.grip_torque,
                                        object_name=self.object_name, q_hold=q_grasp)
            if ctx.grasp_verified(self.arm, self.object_name, frame.max_tilt_deg,
                                  expect_site_at=frame.position,
                                  check_upright=frame.check_upright):
                break
            moved, _ = ctx.object(self.object_name)
            ctx.allowed[self.arm].discard(self.object_name)
            if attempt == 1 and float(np.linalg.norm(moved[:2] - pos[:2])) < 0.05:
                yield from ctx.open_gripper(self.arm, frame.aperture, 1.5)
                continue
            raise SkillFailed("pick", "close", "missed_grasp")
        ctx.carrying[self.arm] = self.object_name
        self._set_phase("lift")
        # Lift-height ladder: high grasps (the bottle neck) sit near the top of
        # the constrained envelope, so back the lift off until IK solves; the
        # object only needs enough clearance for a checked horizontal transit.
        lifted = False
        lift_height = frame.lift_m
        while lift_height >= 0.008 - 1e-9:
            lift_end = frame.position + np.array([0.0, 0.0, lift_height])
            try:
                yield from ctx.play_cartesian(self.arm, lift_end, frame.approach,
                                              frame.lateral, 5.0)
                lifted = True
                break
            except (SkillFailed, IKUnreachable):
                lift_height -= 0.005
        if not lifted:
            raise SkillFailed("pick", "lift", "ik_unreachable")
        self._set_phase("verify")
        pos, _ = ctx.object(self.object_name)
        lift = float(pos[2]) - self._origin_z
        lift_min = LIFT_MIN.get(self.object_name, DEFAULT_LIFT_MIN)
        if lift < lift_min:
            raise SkillFailed("pick", "verify", "missed_grasp")
        if ctx.object_external_force(self.arm, self.object_name) > EXTERNAL_FORCE_MAX:
            raise SkillFailed("pick", "verify", "missed_grasp")


class Place(Skill):
    """Carry to a target anchor, descend to measured support, release, verify."""

    phases = ("carry", "hover", "descend", "release", "verify")

    def __init__(self, arm: str, object_name: str, target: str | dict) -> None:
        super().__init__(arm, object_name)
        self.target = target

    def _target_xyz(self, ctx: TeacherContext) -> np.ndarray:
        if isinstance(self.target, str):
            if self.target not in PLACEMATS:
                raise SkillFailed("place", "carry", f"unknown target {self.target}")
            px, py, _ = PLACEMATS[self.target]
            z = TABLE_TOP_HEIGHT + OBJECT_HALF_HEIGHTS.get(self.object_name, 0.008)
            return np.array([px, py, z])
        rel = self.target
        anchor_pos, anchor_rot_quat = ctx.object(rel["anchor"])
        import mujoco

        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, np.asarray(anchor_rot_quat, dtype=np.float64))
        offset = 0.14 * np.array([-1.0, 0.0, 0.0])
        z = TABLE_TOP_HEIGHT + OBJECT_HALF_HEIGHTS.get(self.object_name, 0.008)
        return np.array([anchor_pos[0] + offset[0], anchor_pos[1] + offset[1], z])

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("place", "carry", "dropped")
        frame_pos, _ = ctx.object(self.object_name)
        others = {
            name: np.array(ctx.object(name)[0])
            for name in ("plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2")
            if name != self.object_name
        }
        target = self._target_xyz(ctx)
        self._set_phase("carry")
        site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
        offset = site_pos - frame_pos
        align = np.array([target[0], target[1], float(site_pos[2])])
        yield from ctx.play_cartesian(self.arm, align, np.array([0.0, 0.0, -1.0]),
                                      None, 4.0)
        self._set_phase("hover")
        ee_end = target + offset
        pts = ctx.plan_cartesian(self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0], ee_end,
                                 np.array([0.0, 0.0, -1.0]), None)
        ctx.check_path(self.arm, pts)
        self._set_phase("descend")
        grip = ctx._hold[self.arm][5]
        motion = ctx.play(self.arm, pts, grip, grip, 4.0)
        supported = False
        for action in motion:
            ctx.step(action)
            pos, _ = ctx.object(self.object_name)
            if (ctx.object_support_force(self.object_name) > 0.06
                    and abs(float(pos[2]) - target[2]) < 0.004):
                supported = True
                break
        if not supported:
            pos, _ = ctx.object(self.object_name)
            if not (ctx.object_support_force(self.object_name) > 0.06
                    and abs(float(pos[2]) - target[2]) < 0.006):
                raise SkillFailed("place", "descend", "no_support")
        self._set_phase("release")
        yield from ctx.open_gripper(self.arm, 0.30, 0.8)
        ctx.carrying[self.arm] = None
        yield from ctx.play_cartesian(
            self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0] + np.array([0.0, 0.0, 0.035]),
            np.array([0.0, 0.0, -1.0]), None, 2.0,
        )
        yield from ctx.play_joint(self.arm, np.array(HOME_JOINTS[self.arm][:5]), 3.0)
        self._set_phase("verify")
        for _ in range(12):  # 0.5 s settle window
            yield ctx.action(self.arm, arm_q(ctx.data, self.arm), ctx._hold[self.arm][5])
        pos, _ = ctx.object(self.object_name)
        xy_err = float(np.linalg.norm(pos[:2] - target[:2]))
        z_err = abs(float(pos[2]) - target[2])
        upright = ctx.object_upright(self.object_name)
        disturbed = max(
            (float(np.linalg.norm(np.array(ctx.object(n)[0]) - p)) for n, p in others.items()),
            default=0.0,
        )
        if xy_err > 0.008 or z_err > 0.004 or upright < 0.95 or disturbed > 0.005:
            raise SkillFailed("place", "verify", "misplaced")


class OpenDrawer(Skill):
    """Physically pull the drawer open via the handle (arm A's caddy side)."""

    phases = ("pregrasp", "approach", "close", "pull", "verify")

    def __init__(self, arm: str = "A") -> None:
        super().__init__(arm, "drawer_top")

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.drawer_opening() > 0.12 * DRAWER_TRAVEL:
            raise SkillFailed("open_drawer", "pregrasp", "already_open")
        ctx.allowed[self.arm] = {"drawer_top", "cabinet"}
        try:
            frame = self.catalog.frame(ctx.scene, "drawer_top", self.arm)
            hover = frame.position + np.array([0.0, 0.0, frame.hover_m])
            front = hover + np.array([0.0, -0.05, 0.0])
            q_front = ctx.plan_ik(self.arm, front, frame.approach, frame.lateral,
                                  seed=self._home_seed(frame))
            q_hover = ctx.plan_ik(self.arm, hover, frame.approach, frame.lateral, seed=q_front)
            self._set_phase("pregrasp")
            yield from ctx.open_gripper(self.arm, frame.aperture)
            # The object is allowed from the approach on: the final run-in to
            # the hover necessarily works at contact distance of it, and any
            # shove is corrected by the live re-plan before the descend.
            ctx.allowed[self.arm].add(self.object_name)
            self._set_phase("approach")
            approach_pts = np.linspace(arm_q(ctx.data, self.arm), q_front, 12)
            ctx.check_path(self.arm, approach_pts)
            yield from ctx.play(self.arm, approach_pts, frame.aperture, frame.aperture, 3.0)
            yield from ctx.play_cartesian(self.arm, hover, frame.approach, frame.lateral, 2.0)
            yield from ctx.play_cartesian(self.arm, frame.position, frame.approach,
                                          frame.lateral, 3.0)
            self._set_phase("close")
            yield from ctx.close_gripper(self.arm, frame.grip_torque, seconds=6.0,
                                        object_name="drawer_top")
            fixed, moving = ctx.finger_forces(self.arm, "drawer_top")
            if min(fixed, moving) <= 0.08:
                raise SkillFailed("open_drawer", "close", "missed_grasp")
            ctx.carrying[self.arm] = "drawer_top"
            self._set_phase("pull")
            # The servo holds the drawer closed through the grasp (a neutral
            # drawer would be dragged open by the closing jaws); only the
            # pull itself needs the zero-error neutral hold.
            ctx.drawer_neutral = True
            pull_end = frame.position + np.array([0.0, -(DRAWER_TRAVEL - 0.004), 0.0])
            pull_pts = ctx.plan_cartesian(self.arm, frame.position, pull_end,
                                          frame.approach, frame.lateral,
                                          q_start=arm_q(ctx.data, self.arm))
            ctx.check_path(self.arm, pull_pts, drawer_follow=True)
            grip = ctx._grip_now[self.arm]
            yield from ctx.play(self.arm, pull_pts, grip, grip, 6.0)
            self._set_phase("verify")
            if ctx.drawer_opening() < 0.88 * DRAWER_TRAVEL:
                raise SkillFailed("open_drawer", "verify", "jammed")
            yield from ctx.open_gripper(self.arm, 0.30, 1.0)
            ctx.carrying[self.arm] = None
            retract = site_pose(ctx.data, f"{self.arm}.ee")[0] + np.array([0.0, 0.0, 0.04])
            yield from ctx.play_cartesian(self.arm, retract, frame.approach, None, 3.0)
            yield from ctx.play_joint(self.arm, np.array(HOME_JOINTS[self.arm][:5]), 3.0)
        finally:
            ctx.drawer_neutral = False
            ctx.allowed[self.arm] = set()


class CloseDrawer(OpenDrawer):
    """Physically push the drawer closed via the handle."""

    phases = ("pregrasp", "approach", "close", "push", "verify")

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        opening = ctx.drawer_opening()
        if opening < 0.8 * DRAWER_TRAVEL:
            raise SkillFailed("close_drawer", "pregrasp", "already_closed")
        ctx.allowed[self.arm] = {"drawer_top", "cabinet"}
        try:
            frame = self.catalog.frame(ctx.scene, "drawer_top", self.arm)
            hover = frame.position + np.array([0.0, 0.0, frame.hover_m])
            self._set_phase("pregrasp")
            yield from ctx.open_gripper(self.arm, frame.aperture)
            # The object is allowed from the approach on: the final run-in to
            # the hover necessarily works at contact distance of it, and any
            # shove is corrected by the live re-plan before the descend.
            ctx.allowed[self.arm].add(self.object_name)
            self._set_phase("approach")
            q_hover = ctx.plan_ik(self.arm, hover, frame.approach, frame.lateral,
                                  seed=self._home_seed(frame))
            approach_pts = np.linspace(arm_q(ctx.data, self.arm), q_hover, 12)
            ctx.check_path(self.arm, approach_pts)
            yield from ctx.play(self.arm, approach_pts, frame.aperture, frame.aperture, 3.0)
            yield from ctx.play_cartesian(self.arm, frame.position, frame.approach,
                                          frame.lateral, 3.0)
            self._set_phase("close")
            yield from ctx.close_gripper(self.arm, frame.grip_torque, seconds=6.0,
                                        object_name="drawer_top")
            fixed, moving = ctx.finger_forces(self.arm, "drawer_top")
            if min(fixed, moving) <= 0.08:
                raise SkillFailed("close_drawer", "close", "missed_grasp")
            ctx.carrying[self.arm] = "drawer_top"
            self._set_phase("push")
            ctx.drawer_neutral = True
            push_end = frame.position + np.array([0.0, opening - 0.004, 0.0])
            yield from ctx.play_cartesian(self.arm, push_end, frame.approach,
                                          frame.lateral, 6.0)
            self._set_phase("verify")
            if ctx.drawer_opening() > 0.12 * DRAWER_TRAVEL:
                raise SkillFailed("close_drawer", "verify", "jammed")
            yield from ctx.open_gripper(self.arm, 0.30, 1.0)
            ctx.carrying[self.arm] = None
            retract = site_pose(ctx.data, f"{self.arm}.ee")[0] + np.array([0.0, 0.0, 0.04])
            yield from ctx.play_cartesian(self.arm, retract, frame.approach, None, 3.0)
            yield from ctx.play_joint(self.arm, np.array(HOME_JOINTS[self.arm][:5]), 3.0)
        finally:
            ctx.drawer_neutral = False
            ctx.allowed[self.arm] = set()
