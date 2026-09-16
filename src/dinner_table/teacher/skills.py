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

import mujoco
import numpy as np

from dinner_table.contracts.geometry import (
    DRAWER_TRAVEL,
    HOME_JOINTS,
    TABLE_TOP_HEIGHT,
)
from dinner_table.policies.conditioning import goal_for_skill
from dinner_table.reasoning.schema import RelativeTarget
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.grasp_catalog import GraspCatalog, GraspFrame
from dinner_table.teacher.ik import IKUnreachable
from dinner_table.teacher.kinematics import arm_q, site_pose

# Every catalog object's body origin rests at its base: a placed object's
# target height is the supporting surface, not surface + half-height.
PLACE_Z = {"table": TABLE_TOP_HEIGHT, "drawer": 0.385}
# The neck pinch rides a 3 cm wall: the lift must stay on the neck, so
# only ~1.2 cm of base clearance is available — enough for a checked transit.
# Cutlery: the eastern drawer columns' IK ceiling caps a pure-vertical lift
# at ~13-19 mm; the transit clearance is established by Place's carry rise
# instead (the drawer is still open at pick time, so nothing is transited).
LIFT_MIN = {
    "bottle": 0.012,
    "fork_1": 0.012, "fork_2": 0.012, "spoon_1": 0.012, "spoon_2": 0.012,
}
DEFAULT_LIFT_MIN = 0.02
# Base radii for the flatten-shift correction: a hollow vessel delivered
# tilted (the side-wall pinch hangs it a few degrees off vertical — the
# grasp-point lever is inherent) lands on its rim edge and flattens on
# release, shifting its base center by radius*sin(tilt) along the lean.
BASE_RADIUS = {"plate": 0.09, "mug": 0.04, "bottle": 0.03}
# The reference teacher's carried-object graze tolerance.
EXTERNAL_FORCE_MAX = 0.10
# The plate's rim tube is smooth: a 0.7 N m saturated press lets the
# swinging rim slide out of the jaws mid-transit (measured). The reference
# carries with the full force-clamped servo (2.94 N m) and the rigid rim
# takes it — the close stays at the tuned 0.7 (a full close ejects the
# plate against the table at grasp time). Cutlery rides the protruding jaw
# tip spheres (narrow boxes never reach the jaw faces): only the full
# clamp drives the tips in deep enough to resist the box hinging on the
# point contacts (measured: 0.5 N m carry loses the moving jaw mid-transit).
CARRY_GRIP_TORQUE = {
    "plate": 2.94,
    "fork_1": 2.94, "fork_2": 2.94, "spoon_1": 2.94, "spoon_2": 2.94,
}


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
        # Keep the saturated close pressing through the lift: once the close
        # generator exits, a plain servo at the resting aperture exerts no
        # steady-state squeeze — the contact decays over ~1 s and a smooth
        # object on the jaw tips (a cutlery box) slides out mid-lift
        # (measured: jaws 6 -> 2.3 N, then dropped). The standalone pick
        # never saw this because the episode ended at the verify before the
        # carry audit could fire.
        with ctx.grip_saturation(self.arm,
                                 CARRY_GRIP_TORQUE.get(self.object_name,
                                                       frame.grip_torque)):
            # Lift-height ladder: high grasps (the bottle neck) sit near the
            # top of the constrained envelope, so back the lift off until IK
            # solves; the object only needs enough clearance for a checked
            # horizontal transit. Cutlery lifts straight up first (inside the
            # caddy, under the roof), then EXITS south over the open front
            # wall to a high outside point — fully clear of the drawer before
            # anything else moves (the reference's clearance stage; the
            # in-caddy envelope is roof-capped ~0.43 while south of the
            # drawer the arm reaches 0.46+).
            lifted = False
            lift_height = frame.lift_m
            while lift_height >= 0.008 - 1e-9:
                lift_end = frame.position + np.array([0.0, 0.0, lift_height])
                try:
                    start = site_pose(ctx.data, f"{self.arm}.ee")[0]
                    pts = ctx.plan_cartesian(self.arm, start, lift_end,
                                             frame.approach, frame.lateral)
                    ctx.check_path(self.arm, pts)
                    grip = ctx._grip_now[self.arm]
                    yield from ctx.play(self.arm, pts, grip, grip, 5.0)
                    # Converge on the commanded lift before measuring it: the
                    # playback clock ends with the servo still short of the
                    # last waypoint, and reading the lagged pose fails deep
                    # reaches spuriously (the drawer columns lift only ~70%
                    # at the tick).
                    yield from ctx.hold(self.arm, pts[-1], 0.6)
                    lifted = True
                    break
                except (SkillFailed, IKUnreachable):
                    lift_height -= 0.005
            if not lifted:
                raise SkillFailed("pick", "lift", "ik_unreachable")
            if self.object_name.startswith(("fork", "spoon")):
                exited = False
                site_now, _ = site_pose(ctx.data, f"{self.arm}.ee")
                for out_y, out_z in ((0.13, 0.44), (0.13, 0.43), (0.10, 0.44),
                                     (0.10, 0.43), (0.10, 0.42)):
                    exit_pt = np.array([site_now[0], site_now[1] - out_y, out_z])
                    try:
                        start = site_pose(ctx.data, f"{self.arm}.ee")[0]
                        pts = ctx.plan_cartesian(self.arm, start, exit_pt,
                                                 frame.approach, None)
                        ctx.check_path(self.arm, pts)
                        grip = ctx._grip_now[self.arm]
                        yield from ctx.play(self.arm, pts, grip, grip, 4.0)
                        yield from ctx.hold(self.arm, pts[-1], 0.4)
                        exited = True
                        break
                    except (SkillFailed, IKUnreachable):
                        continue
                if not exited:
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
        """Resolve the placement anchor through the deployed goal rule.

        Named settings and anchor-relative targets go through
        ``conditioning.goal_for_skill`` so the teacher and the runtime resolve
        them identically; a dict carrying ``point`` is an explicit world
        position (the handoff's relay anchor).
        """
        target = self.target
        if isinstance(target, dict):
            if "point" in target:
                return np.asarray(target["point"], dtype=np.float64)
            target = RelativeTarget(relation=target["relation"], anchor=target["anchor"])
        anchor_position = None
        if isinstance(target, RelativeTarget):
            anchor_position = ctx.object(target.anchor)[0]
        try:
            goal = goal_for_skill("place", self.object_name, None, target, anchor_position)
        except (ValueError, KeyError) as exc:
            raise SkillFailed("place", "carry", f"unknown target {self.target}") from exc
        return np.array([goal[0], goal[1], PLACE_Z["table"]])

    def _flatten_shift(self, obj_quat: np.ndarray) -> np.ndarray:
        """Predicted horizontal shift when a leaning vessel rocks flat.

        A tilted cylinder lands on its rim edge; flattening rotates the base
        center toward the lean by radius*sin(tilt). Zero for objects without
        a cataloged base radius (cutlery rolls negligibly).
        """
        radius = BASE_RADIUS.get(self.object_name)
        if radius is None:
            return np.zeros(3)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, np.asarray(obj_quat, dtype=np.float64))
        lean = np.array(mat.reshape(3, 3)[:2, 2], dtype=np.float64)  # up-axis xy
        norm = float(np.linalg.norm(lean))
        if norm < 1e-6:
            return np.zeros(3)
        tilt = float(np.arctan2(norm, abs(float(mat.reshape(3, 3)[2, 2]))))
        return radius * np.sin(tilt) * np.array([lean[0], lean[1], 0.0]) / norm

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("place", "carry", "dropped")
        frame_pos, _ = ctx.object(self.object_name)
        others = {
            name: np.array(ctx.object(name)[0])
            for name in ("plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2")
            if name != self.object_name
            # Drawer-resident cutlery legitimately moves with the drawer
            # (the servo close after retrieval slides it); it is only
            # checked when the placed object is NOT cutlery.
            and not (self.object_name.startswith(("fork", "spoon"))
                     and name.startswith(("fork", "spoon")))
        }
        target = self._target_xyz(ctx)
        # Carry in the grasp orientation: the cataloged lateral pins the wrist
        # roll through every carried motion — an unconstrained roll twists the
        # grasp until a jaw unloads (measured: 0/4 for every object).
        frame = self.catalog.frame(ctx.scene, self.object_name, self.arm)
        self._set_phase("carry")
        site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
        offset = site_pos - frame_pos
        # Hold an active squeeze through every carried motion (the reference
        # carries with the servo commanded closed): a plain servo at the
        # resting aperture lets a swinging rim pinch or a grazed utensil tip
        # unload a jaw mid-transit (measured: plate and fork, 0/4 each).
        with ctx.grip_saturation(self.arm,
                                 CARRY_GRIP_TORQUE.get(self.object_name,
                                                       frame.grip_torque)):
            # Rise to the carry line before transiting: a fork lifted only to
            # its pick height rides AT the drawer walls' tops, and the servo
            # drawer-close then shoves it north in the grip (measured). The
            # line is ABSOLUTELY capped ~5 mm inside the constrained-IK
            # ceiling: a warm-chained rise plan can solve past the ceiling,
            # but the executed corner pose (wrist_flex at its limit) cannot
            # be re-solved by any later plan — the align then fails at every
            # height (measured).
            carry_cap = TABLE_TOP_HEIGHT + 0.068
            for rise in (0.020, 0.015, 0.010, 0.005):
                target_z = min(float(site_pos[2]) + rise, carry_cap)
                if target_z <= float(site_pos[2]) + 1e-6:
                    break
                try:
                    yield from ctx.play_cartesian(
                        self.arm, np.array([site_pos[0], site_pos[1], target_z]),
                        frame.approach, frame.lateral, 1.5,
                    )
                    break
                except (IKUnreachable, SkillFailed):
                    continue
            site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
            if self.object_name.startswith(("fork", "spoon")):
                # The open drawer's front wall crosses the cutlery placement
                # band beside the plate; slide the drawer closed by servo
                # before transiting (arm A is holding the utensil, so the
                # physical CloseDrawer grasp is not an option). The carried
                # utensil rides above the wall tops at the carry line.
                yield from ctx.servo_drawer(self.arm)
            # Align the GRASP POINT over the target (target + grasp offset):
            # the site must not go to the placemat center itself — a rim pinch
            # then hangs a plate one rim-radius off and the diagonal descent
            # swings the grasp loose (measured). The transit runs at the
            # reference's cutlery speed (~3 cm/s): a 4 s sprint quadruples
            # the carried pendulum's swing energy and rolls a clamped
            # utensil's handle until it snaps out of the jaws (measured).
            # Where the carry line rides at the arm's IK ceiling (the deep
            # columns), step the align height down until the plan solves —
            # the drawer is closed by now, so the transit is clear at any
            # height above the table. Cutlery transits with the lateral
            # relaxed (approach-down only): the deep west columns' corner
            # does not solve with the wrist pinned, and a friction-held box
            # between the jaw tips does not care about wrist orientation.
            is_cutlery = self.object_name.startswith(("fork", "spoon"))
            transit_lateral = None if is_cutlery else frame.lateral
            align_xy = (target[0] + offset[0], target[1] + offset[1])
            align_z = float(site_pos[2])
            aligned = False
            while align_z >= TABLE_TOP_HEIGHT + 0.05 - 1e-9:
                try:
                    yield from ctx.play_cartesian(
                        self.arm, np.array([align_xy[0], align_xy[1], align_z]),
                        frame.approach, transit_lateral, 7.0,
                    )
                    aligned = True
                    break
                except IKUnreachable:
                    align_z -= 0.005
            if not aligned:
                raise SkillFailed("place", "carry", "ik_unreachable")
            self._set_phase("hover")
            # Sample the grasp offset across the carried object's swing
            # rather than waiting the swing out: the mean over ~1 s equals
            # the equilibrium hang (the pendulum is symmetric), and a long
            # dead hold only grinds the plate's rim tube between the
            # clamped jaws until it escapes (measured: 3 s hold, 2/4 seeds).
            hold_q = arm_q(ctx.data, self.arm)
            samples: list[np.ndarray] = []
            sample_end = float(ctx.data.time) + 1.0
            while float(ctx.data.time) < sample_end:
                yield from ctx.hold(self.arm, hold_q, 0.1)
                s_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
                o_pos, _ = ctx.object(self.object_name)
                samples.append(s_pos - o_pos)
            # A leaning vessel is delivered short of the target by its
            # predicted flatten shift so it settles ON the target when the
            # release lets it rock flat. The last 2 mm is a deliberate
            # press: an object delivered exactly to its rest height only
            # grazes the surface with ~0 N support while the grip still
            # carries its weight (measured on the fork). The offset is the
            # MEDIAN across the swing — a mean is inflated by the pendulum
            # extremes (measured: a swinging spoon produced an 8 cm offset
            # and an unreachable descent target).
            obj_quat = ctx.object(self.object_name)[1]
            ee_end = (target + np.median(samples, axis=0)
                      - self._flatten_shift(obj_quat)
                      - np.array([0.0, 0.0, 0.002]))
            try:
                pts = ctx.plan_cartesian(self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0],
                                         ee_end, frame.approach, frame.lateral)
            except IKUnreachable as exc:
                raise SkillFailed("place", "hover", "ik_unreachable") from exc
            ctx.check_path(self.arm, pts)
            self._set_phase("descend")
            grip = ctx._grip_now[self.arm]
            motion = ctx.play(self.arm, pts, grip, grip, 4.0)

            def seated() -> bool:
                # Near-flat on the surface with real support: a first
                # rim-edge touch leaves hollow vessels tilted a few degrees
                # and a couple mm high — the flatten happens on release
                # (predicted and pre-compensated above).
                pos, _ = ctx.object(self.object_name)
                return (ctx.object_support_force(self.object_name) > 0.06
                        and abs(float(pos[2]) - target[2]) < 0.003)

            supported = False
            for action in motion:
                ctx.step(action)
                if seated():
                    supported = True
                    break
            if not supported:
                # The playback clock ends with the servo short of the last
                # waypoint; hold the drop target until the object seats (or
                # genuinely never touches).
                hold_end = float(ctx.data.time) + 1.5
                while float(ctx.data.time) < hold_end:
                    yield ctx.action(self.arm, pts[-1], grip)
                    if seated():
                        supported = True
                        break
            if not supported:
                pos, _ = ctx.object(self.object_name)
                if not (ctx.object_support_force(self.object_name) > 0.06
                        and abs(float(pos[2]) - target[2]) < 0.003):
                    raise SkillFailed("place", "descend", "no_support")
        self._set_phase("release")
        # End the carry BEFORE opening: the grasp audit must not read the
        # intentional release as a fumbled grasp, and the exit's scratch
        # audit must treat the placed object as world, not cargo. Open
        # slowly (the reference's 3 s release): a fast open throws the
        # just-supported object ~6 mm as the pinch preload relaxes
        # (measured on the mug).
        ctx.carrying[self.arm] = None
        yield from ctx.open_gripper(self.arm, 0.30, 3.0)
        # Rise well clear of the placed object before the home sweep: the
        # joint arc dips a few mm early on and the jaw tips catch the rim of
        # a just-placed mug (measured: hooked at site z 0.444 vs rim 0.435).
        for rise in (0.06, 0.05, 0.04, 0.03):
            try:
                yield from ctx.play_cartesian(
                    self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0]
                    + np.array([0.0, 0.0, rise]),
                    frame.approach, frame.lateral, 2.0,
                )
                break
            except (IKUnreachable, SkillFailed):
                continue
        # Home along a checked path; if the arc would clip the placed object
        # (or anything else), retreat toward the table edge and re-plan.
        q_home = np.array(HOME_JOINTS[self.arm][:5])

        def home_points() -> np.ndarray:
            q_start = arm_q(ctx.data, self.arm)
            n = max(8, int(np.max(np.abs(q_home - q_start)) / 0.035) + 2)
            return np.linspace(q_start, q_home, n)

        try:
            home_pts = home_points()
            ctx.check_path(self.arm, home_pts)
        except SkillFailed:
            back = site_pose(ctx.data, f"{self.arm}.ee")[0] + np.array([0.0, -0.10, 0.0])
            yield from ctx.play_cartesian(self.arm, back, frame.approach,
                                          frame.lateral, 2.5)
            home_pts = home_points()
            ctx.check_path(self.arm, home_pts)
        grip_now = ctx._grip_now[self.arm]
        yield from ctx.play(self.arm, home_pts, grip_now, grip_now, 3.0)
        self._set_phase("verify")
        for _ in range(12):  # 0.5 s settle window
            yield ctx.action(self.arm, arm_q(ctx.data, self.arm), ctx._grip_now[self.arm])
        pos, _ = ctx.object(self.object_name)
        xy_err = float(np.linalg.norm(pos[:2] - target[:2]))
        z_err = abs(float(pos[2]) - target[2])
        upright = ctx.object_upright(self.object_name)
        disturbed = max(
            (float(np.linalg.norm(np.array(ctx.object(n)[0]) - p)) for n, p in others.items()),
            default=0.0,
        )
        if xy_err > 0.006 or z_err > 0.003 or disturbed > 0.004:
            raise SkillFailed("place", "verify", "misplaced")
        if frame.check_upright and upright < 0.98:
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
            # Full-servo close (reference port): the STS-3215 servo is
            # force-clamped at 2.94 N m, so holding the fully-closed target
            # keeps a firm squeeze on the handle through the pull. The
            # torque-saturated close used for gram-scale objects would decay
            # the moment the drawer's drag loads the jaws (measured).
            q_now = arm_q(ctx.data, self.arm)
            yield from ctx.play(self.arm, np.vstack([q_now, q_now]),
                                ctx._grip_now[self.arm], 0.0, 4.0)
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
            yield from ctx.play(self.arm, pull_pts, 0.0, 0.0, 6.0)
            self._set_phase("verify")
            if ctx.drawer_opening() < 0.88 * DRAWER_TRAVEL:
                raise SkillFailed("open_drawer", "verify", "jammed")
            # End the carry BEFORE opening: the grasp audit must not read the
            # intentional release as a fumbled grasp.
            ctx.carrying[self.arm] = None
            # Hold the drawer open via its servo before releasing: while the
            # neutral hold is active the drawer is effectively free and the
            # withdrawing arm's drag slides it shut (measured).
            ctx.hold_drawer_open()
            yield from ctx.open_gripper(self.arm, 0.30, 1.0)
            # Withdraw along a checked joint path home: a constrained
            # Cartesian retract at the pulled handle position can sit past
            # the approach-down IK envelope (measured after the site-depth
            # alignment); the joint arc clears it and is contact-audited.
            try:
                home_pts = np.linspace(
                    arm_q(ctx.data, self.arm),
                    np.array(HOME_JOINTS[self.arm][:5]), 12,
                )
                ctx.check_path(self.arm, home_pts)
            except SkillFailed as exc:
                raise SkillFailed("drawer", "verify", "path_blocked") from exc
            grip_now = ctx._grip_now[self.arm]
            yield from ctx.play(self.arm, home_pts, grip_now, grip_now, 3.0)
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
        # The OPEN handle sits deep past the arm's approach-down grasp
        # envelope (too close to the base; measured). Servo-draw the drawer
        # to mid-travel first so the handle returns to the proven grasp
        # band, then finish the last stretch with the physical handle push.
        GRASP_BAND_OPENING = 0.06
        if opening > GRASP_BAND_OPENING:
            yield from ctx.servo_drawer(self.arm, target_opening=GRASP_BAND_OPENING)
            opening = ctx.drawer_opening()
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
            # Approach via the front waypoint (OpenDrawer's proven chain): at
            # the OPEN handle's deep-south position the hover does not solve
            # from a cold home seed, but the closer front point does and the
            # warm-chained hover follows.
            front = hover + np.array([0.0, -0.05, 0.0])
            q_front = ctx.plan_ik(self.arm, front, frame.approach, frame.lateral,
                                  seed=self._home_seed(frame))
            q_hover = ctx.plan_ik(self.arm, hover, frame.approach, frame.lateral,
                                  seed=q_front)
            approach_pts = np.linspace(arm_q(ctx.data, self.arm), q_front, 12)
            ctx.check_path(self.arm, approach_pts)
            yield from ctx.play(self.arm, approach_pts, frame.aperture, frame.aperture, 3.0)
            yield from ctx.play_cartesian(self.arm, hover, frame.approach,
                                          frame.lateral, 2.0)
            yield from ctx.play_cartesian(self.arm, frame.position, frame.approach,
                                          frame.lateral, 3.0)
            self._set_phase("close")
            # Full-servo close, as in OpenDrawer: the force-clamped servo holds
            # the squeeze on the handle through the push.
            q_now = arm_q(ctx.data, self.arm)
            yield from ctx.play(self.arm, np.vstack([q_now, q_now]),
                                ctx._grip_now[self.arm], 0.0, 4.0)
            fixed, moving = ctx.finger_forces(self.arm, "drawer_top")
            if min(fixed, moving) <= 0.08:
                raise SkillFailed("close_drawer", "close", "missed_grasp")
            ctx.carrying[self.arm] = "drawer_top"
            self._set_phase("push")
            ctx.drawer_neutral = True
            push_end = frame.position + np.array([0.0, opening - 0.004, 0.0])
            # _grip_now is 0.0 from the close: play_cartesian holds the
            # fully-closed servo target through the push.
            yield from ctx.play_cartesian(self.arm, push_end, frame.approach,
                                          frame.lateral, 6.0)
            self._set_phase("verify")
            if ctx.drawer_opening() > 0.12 * DRAWER_TRAVEL:
                raise SkillFailed("close_drawer", "verify", "jammed")
            # End the carry BEFORE opening (grasp-audit exemption; see OpenDrawer).
            ctx.carrying[self.arm] = None
            yield from ctx.open_gripper(self.arm, 0.30, 1.0)
            # Withdraw along a checked joint path home: a constrained
            # Cartesian retract at the pulled handle position can sit past
            # the approach-down IK envelope (measured after the site-depth
            # alignment); the joint arc clears it and is contact-audited.
            try:
                home_pts = np.linspace(
                    arm_q(ctx.data, self.arm),
                    np.array(HOME_JOINTS[self.arm][:5]), 12,
                )
                ctx.check_path(self.arm, home_pts)
            except SkillFailed as exc:
                raise SkillFailed("drawer", "verify", "path_blocked") from exc
            grip_now = ctx._grip_now[self.arm]
            yield from ctx.play(self.arm, home_pts, grip_now, grip_now, 3.0)
        finally:
            ctx.drawer_neutral = False
            ctx.allowed[self.arm] = set()
