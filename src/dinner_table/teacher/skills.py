"""Privileged teacher skills: verified manipulation primitives.

Each skill is a coroutine yielding 12-dim merged targets at 25 Hz; the caller
steps physics per yield via ``TeacherContext.step`` (which also runs the
safety audits). Failure raises ``SkillFailed`` with an attributable phase and
cause. The recipes are hover/descend grasps with one bounded retry,
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
    PLACEMATS,
    TABLE_TOP_HEIGHT,
)
from dinner_table.executor.workspace import ZoneClaims, zone_of
from dinner_table.policies.conditioning import goal_for_skill
from dinner_table.reasoning.schema import RelativeTarget
from dinner_table.scene.objects import (
    BOTTLE_MOUTH_Z,
    MUG_INNER_R,
    MUG_RIM_Z,
)
from dinner_table.scene.water import MAX_WATER_HALF_HEIGHT, set_fill_fraction
from dinner_table.teacher.context import JAW_FORCE_MIN, SkillFailed, TeacherContext
from dinner_table.teacher.grasp_catalog import GraspCatalog, GraspCatalogError, GraspFrame
from dinner_table.teacher.ik import IKUnreachable
from dinner_table.teacher.kinematics import arm_q, site_pose

# Every catalog object's body origin rests at its base: a placed object's
# target height is the supporting surface, not surface + half-height.
PLACE_Z = {"table": TABLE_TOP_HEIGHT, "drawer": 0.385}
LIFT_MIN = {"bottle": 0.05}
DEFAULT_LIFT_MIN = 0.02
# Closed-loop seating: a carried vessel hangs off its grasp point and rocks
# flat against the table as it touches down, so where it lands cannot be
# predicted from the hanging pose. The descend measures the residual and
# re-seats on it instead (see Place.run).
PLACE_RESEAT_TOL_M = 0.003   # residual at which re-seating stops paying
PLACE_RESEAT_ATTEMPTS = 2    # bounded: the residual collapses in one or two
PLACE_RESEAT_LIFT_M = 0.005  # just enough to unload; a big lift lets it re-tilt
# A tall vessel set down on its base rim rocks for seconds before it is still,
# and a placement judged mid-rock reads high with the lean to match (measured
# on the bottle: 11 mm off and 0.92 upright at 1 s, 0.4 mm and 1.00 at 6 s).
# Verification therefore waits for the plan's stillness threshold first.
PLACE_STILL_SPEED = 0.003  # m/s
PLACE_SETTLE_MAX_S = 8.0
PLACE_RELEASED_FORCE_MAX = 0.01  # N per jaw once the object is truly let go
# Opening the jaws retracts only the MOVING one; the fixed jaw stays where it
# was, a hair outboard of what was gripped. Re-solving IK for the retreat
# rotates the wrist a fraction of a degree, which is enough to swing the fixed
# jaw's tip sphere back into a just-released object and lever it over
# (measured on the relayed bottle: 0.47 N at the tip, then a topple). Backing
# the tool point off along its own +X — the direction the fixed jaw sits in —
# clears the jaw before any lift.
RELEASE_BACKOFF_M = 0.010
# Carried-object graze tolerance: forces above this mean the object is being
# dragged against something rather than carried clear of it.
EXTERNAL_FORCE_MAX = 0.10
# The plate's rim tube is smooth: a 0.7 N m saturated press lets the
# swinging rim slide out of the jaws mid-transit (measured). The rim needs
# the full force-clamped servo (2.94 N m) and the rigid rim
# takes it — the close stays at the tuned 0.7 (a full close ejects the
# plate against the table at grasp time). Cutlery rides the protruding jaw
# tip spheres (narrow boxes never reach the jaw faces): only the full
# clamp drives the tips in deep enough to resist the box hinging on the
# point contacts (measured: 0.5 N m carry loses the moving jaw mid-transit).
CARRY_GRIP_TORQUE = {
    "plate": 2.94,
    "fork_1": 2.94, "fork_2": 2.94, "spoon_1": 2.94, "spoon_2": 2.94,
    # The hollow vessels carry at their cataloged close torque; listing them
    # keeps the table total so the bimanual skills, which have no grasp frame
    # in hand while an object is already held, can read it directly.
    "mug": 0.7, "bottle": 0.5,
}
# Table-supported relay anchor: the bottle's side grasp reaches a far annulus
# on each arm, so the dual-reach lens for it sits mid-table rather than in the
# front-center SHARED_ZONE the top-down grasps share (measured).
# The anchor may shift toward the receiving arm.
RELAY_ANCHOR = (0.0, -0.02)
RELAY_ARM_BIAS = 0.04
# Pour schedule. POUR_TILT_DEG is the commanded arc tilt, but grasp compliance
# rotates the bottle back inside the jaws under load (measured: actual tilt
# 46.5 deg against 72 commanded, and a further 46.5 -> 21 deg slide when more
# tilt is commanded, at double mass). Commanded extra tilt therefore worsens
# the slip; the return must re-measure the physical bottle pose instead.
POUR_TILT_DEG = 72.0
POUR_FLOW_TILT_DEG = 55.0
POUR_GRIP_TORQUE = 2.5  # N m, below the gripper actuator's 2.94 N m limit.
# Measured boundary: even at the actuator's physical 2.94 N m cap the
# double-mass bottle over the table reaches only ~46.5 deg actual tilt
# (55 deg required to flow) with progressive neck slip, and commanding more
# tilt slides it further (46.5 -> 21 deg). The heavy/table pour is a physical
# limit of the modeled neck pinch, not a planner or threshold issue.
POUR_RATE_PER_S = 0.55  # mug fill fraction gained per second of flow


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
                                      frame.lateral, seed=self._home_seed(frame),
                                      axis_index=frame.axis_index)
                hover_pos = frame.position + np.array([0.0, 0.0, frame.hover_m])
                q_hover = None
                hovers = (frame.hover_m, frame.hover_m - 0.010,
                          max(frame.hover_m - 0.020, 0.010))
                for hover in hovers:
                    try:
                        q_hover = ctx.plan_ik(
                            self.arm, frame.position + np.array([0.0, 0.0, hover]),
                            frame.approach, frame.lateral, seed=q_grasp,
                            axis_index=frame.axis_index)
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
                                     seed=q_hover, axis_index=frame.axis_index)
            except IKUnreachable:
                q_high = q_hover
            if q_high is q_hover:
                # Constrained high corridor infeasible: try a RELAXED high
                # point (approach-down, no lateral); the grasp orientation is
                # only enforced on the final hover-to-grasp descend.
                try:
                    q_high = ctx.plan_ik(self.arm, high, frame.approach, None,
                                         seed=q_hover, axis_index=frame.axis_index)
                except IKUnreachable:
                    q_high = None
            if (q_high is None or q_high is q_hover) and self.object_name.startswith(
                ("fork", "spoon")
            ):
                # Cutlery approach from the drawer's open FRONT:
                # a side detour would sweep over the neighboring
                # utensil columns, but the front strip is always clear.
                hover_pos = frame.position + np.array([0.0, 0.0, frame.hover_m])
                front = hover_pos + np.array([0.0, -0.05, 0.0])
                try:
                    q_front = ctx.plan_ik(self.arm, front, frame.approach,
                                          frame.lateral, seed=q_hover, axis_index=frame.axis_index)
                    front_pts = np.linspace(arm_q(ctx.data, self.arm), q_front, 12)
                    ctx.check_path(self.arm, front_pts)
                    runin = ctx.plan_cartesian(
                        self.arm, front, hover_pos, frame.approach, frame.lateral,
                        q_start=q_front, axis_index=frame.axis_index)
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
                                               frame.lateral, seed=q_hover,
                                               axis_index=frame.axis_index)
                        detour_pts = np.linspace(arm_q(ctx.data, self.arm), q_detour, 12)
                        ctx.check_path(self.arm, detour_pts)
                        runin = ctx.plan_cartesian(
                            self.arm, detour,
                            frame.position + np.array([0.0, 0.0, frame.hover_m]),
                            frame.approach, frame.lateral, q_start=q_detour,
                            axis_index=frame.axis_index)
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
                    frame.approach, frame.lateral, axis_index=frame.axis_index)
                hover_pos = frame.position + np.array([0.0, 0.0, frame.hover_m])
                descend_pts = ctx.plan_cartesian(
                    self.arm, high, hover_pos, frame.approach, frame.lateral,
                    q_start=over_pts[-1], axis_index=frame.axis_index)
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
                                      frame.lateral, seed=q_hover, axis_index=frame.axis_index)
            except IKUnreachable as exc:
                ctx.allowed[self.arm].discard(self.object_name)
                if attempt == 2:
                    raise SkillFailed("pick", "pregrasp", "ik_unreachable") from exc
                continue
            try:
                yield from ctx.play_cartesian(self.arm, frame.position, frame.approach,
                                              frame.lateral, 3.0, axis_index=frame.axis_index)
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
        ctx.begin_carry(self.arm, self.object_name,
                        CARRY_GRIP_TORQUE.get(self.object_name, frame.grip_torque))
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
            # anything else moves (a full clearance stage; the
            # in-caddy envelope is roof-capped ~0.43 while south of the
            # drawer the arm reaches 0.46+).
            lifted = False
            lift_height = frame.lift_m
            while lift_height >= 0.008 - 1e-9:
                lift_end = frame.position + np.array([0.0, 0.0, lift_height])
                try:
                    start = site_pose(ctx.data, f"{self.arm}.ee")[0]
                    pts = ctx.plan_cartesian(self.arm, start, lift_end,
                                             frame.approach, frame.lateral,
                                             axis_index=frame.axis_index)
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
                                                 frame.approach, None, axis_index=frame.axis_index)
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
        if frame.check_upright and ctx.object_upright(self.object_name) < float(
            np.cos(np.deg2rad(frame.max_tilt_deg))
        ):
            raise SkillFailed("pick", "verify", "missed_grasp")
        pos, _ = ctx.object(self.object_name)
        lift = float(pos[2]) - self._origin_z
        lift_min = LIFT_MIN.get(self.object_name, DEFAULT_LIFT_MIN)
        if lift < lift_min:
            raise SkillFailed("pick", "verify", "missed_grasp")
        if ctx.object_external_force(self.arm, self.object_name) > EXTERNAL_FORCE_MAX:
            raise SkillFailed("pick", "verify", "missed_grasp")


def plan_grasp_and_hover(ctx: TeacherContext, arm: str, frame: GraspFrame,
                        seed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Solve the grasp pose and a hover above it, or raise ``IKUnreachable``.

    This is the exact feasibility test ``Pick`` applies before it commits to an
    approach, exposed so a planner can ask "could this arm pick the object up
    from there?" and get the same answer the skill will (see
    ``bimanual.relay_anchor``, which must not park an object outside the
    receiving arm's envelope).
    """
    if seed is None:
        seed = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
        if frame.wrist_roll_seed is not None:
            seed[4] = frame.wrist_roll_seed
    q_grasp = ctx.plan_ik(arm, frame.position, frame.approach, frame.lateral, seed=seed)
    hover_pos = frame.position + np.array([0.0, 0.0, frame.hover_m])
    for hover in (frame.hover_m, frame.hover_m - 0.010, max(frame.hover_m - 0.020, 0.010)):
        try:
            q_hover = ctx.plan_ik(
                arm, frame.position + np.array([0.0, 0.0, hover]),
                frame.approach, frame.lateral, seed=q_grasp,
            )
        except IKUnreachable:
            continue
        return q_grasp, q_hover
    raise IKUnreachable(f"{arm}.ee", hover_pos)


def _joint_segment(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Joint-space waypoints from one arm pose to another, ~0.035 rad apart."""
    q_from = np.asarray(q_from, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)
    n = max(8, int(np.max(np.abs(q_to - q_from)) / 0.035) + 2)
    return np.linspace(q_from, q_to, n)


def _flatten_rotation(obj_quat: np.ndarray) -> np.ndarray:
    """Rotation (3,3) taking an object's current attitude to its flat one.

    A carried vessel hangs off its grasp point by a few degrees and rocks flat
    as the table takes its weight. While it rocks, the clamped grasp point is
    the pivot, so the object's own origin swings by exactly this rotation
    applied to the origin-to-grasp-point vector — which is what lets the
    descend aim the tool point at where the object will END UP rather than
    where it currently hangs. The flat attitude is the current one with the
    lean taken out: same yaw, no tilt.
    """
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(obj_quat, dtype=np.float64))
    rot = mat.reshape(3, 3)
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    cos, sin = np.cos(yaw), np.sin(yaw)
    flat = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return flat @ rot.T


class Place(Skill):
    """Carry to a target anchor, descend to measured support, release, verify."""

    phases = ("carry", "hover", "descend", "release", "verify")

    def __init__(self, arm: str, object_name: str,
                 target: str | RelativeTarget | tuple | np.ndarray) -> None:
        super().__init__(arm, object_name)
        self.target = target

    def _target_xyz(self, ctx: TeacherContext) -> np.ndarray:
        if isinstance(self.target, str):
            if self.target not in PLACEMATS:
                raise SkillFailed("place", "carry", f"unknown target {self.target}")
            px, py, _ = PLACEMATS[self.target]
            return np.array([px, py, PLACE_Z["table"]])
        if isinstance(self.target, RelativeTarget):
            # Anchor-relative target resolved through the same function the
            # runtime conditions on, so the teacher's demonstrations and the
            # deployed policy agree on where "left of the plate" is.
            anchor_pos, _ = ctx.object(self.target.anchor)
            goal = goal_for_skill("place", self.object_name, None, self.target,
                                  anchor_position=np.asarray(anchor_pos, dtype=np.float64))
            return np.array([goal[0], goal[1], PLACE_Z["table"]])
        # Explicit world-frame anchor (the relay spot); z is always the table.
        explicit = np.asarray(self.target, dtype=np.float64).ravel()
        if explicit.shape[0] < 2:
            raise SkillFailed("place", "carry", f"unknown target {self.target}")
        return np.array([explicit[0], explicit[1], PLACE_Z["table"]])

    def _carry_frame(self, ctx: TeacherContext) -> GraspFrame:
        """The grasp frame to carry in; a tipped vessel has no valid frame."""
        try:
            return self.catalog.frame(ctx.scene, self.object_name, self.arm)
        except GraspCatalogError as exc:
            raise SkillFailed("place", "carry", "dropped") from exc

    def feasible_align_height(self, ctx: TeacherContext) -> float | None:
        """Highest align height whose carry transit dry-plans (no motion).

        Mirrors run()'s carry block (rise, then the descending align-height
        loop) using scratch planning only; keep the two in sync. Callers use
        it to choose between candidate targets before committing to motion.
        """
        frame = self._carry_frame(ctx)
        site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
        frame_pos, _ = ctx.object(self.object_name)
        offset = site_pos - frame_pos
        target = self._target_xyz(ctx)
        carry_cap = TABLE_TOP_HEIGHT + 0.068
        risen = np.array(site_pos, dtype=np.float64)
        for rise in (0.020, 0.015, 0.010, 0.005):
            target_z = min(float(site_pos[2]) + rise, carry_cap)
            if target_z <= float(site_pos[2]) + 1e-6:
                break
            try:
                ctx.plan_cartesian(
                    self.arm,
                    risen,
                    np.array([site_pos[0], site_pos[1], target_z]),
                    frame.approach,
                    frame.lateral,
                    axis_index=frame.axis_index,
                )
                risen = np.array([site_pos[0], site_pos[1], target_z])
                break
            except (IKUnreachable, SkillFailed):
                continue
        is_cutlery = self.object_name.startswith(("fork", "spoon"))
        transit_lateral = None if is_cutlery else frame.lateral
        align_xy = (target[0] + offset[0], target[1] + offset[1])
        z = float(risen[2])
        while z >= TABLE_TOP_HEIGHT + 0.05 - 1e-9:
            goal = np.array([align_xy[0], align_xy[1], z])
            try:
                pts = ctx.plan_cartesian(
                    self.arm,
                    risen,
                    goal,
                    frame.approach,
                    transit_lateral,
                    axis_index=frame.axis_index,
                )
                if self.object_name == "bottle":
                    pts = _joint_segment(arm_q(ctx.data, self.arm), pts[-1])
                ctx.check_path(self.arm, pts)
                return z
            except (IKUnreachable, SkillFailed):
                z -= 0.005
        return None

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
        frame = self._carry_frame(ctx)
        self._set_phase("carry")
        site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
        offset = site_pos - frame_pos
        # Hold an active squeeze through every carried motion (the servo
        # is commanded closed through the carry): a plain servo at the
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
                        frame.approach, frame.lateral, 1.5, axis_index=frame.axis_index)
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
            # tuned cutlery speed (~3 cm/s): a 4 s sprint quadruples
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
                    goal = np.array([align_xy[0], align_xy[1], align_z])
                    if self.object_name == "bottle":
                        pts = ctx.plan_cartesian(
                            self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0], goal,
                            frame.approach, transit_lateral, axis_index=frame.axis_index)
                        pts = _joint_segment(arm_q(ctx.data, self.arm), pts[-1])
                        ctx.check_path(self.arm, pts)
                        grip = ctx._grip_now[self.arm]
                        yield from ctx.play(self.arm, pts, grip, grip, 7.0)
                    else:
                        yield from ctx.play_cartesian(
                            self.arm, goal, frame.approach, transit_lateral,
                            7.0, axis_index=frame.axis_index)
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
            # Deliver the GRASP POINT so the object lands on the target: the
            # last 2 mm is a deliberate press, since an object delivered
            # exactly to its rest height only grazes the surface with ~0 N
            # support while the grip still carries its weight (measured on
            # the fork). The offset is the MEDIAN across the swing — a mean
            # is inflated by the pendulum extremes (measured: a swinging
            # spoon produced an 8 cm offset and an unreachable descent
            # target).
            #
            # A carried vessel hangs off the grasp point by several degrees
            # (the lever is inherent to a rim or wall pinch), so the hanging
            # offset sampled here is NOT the offset the object settles at:
            # it rocks flat against the table DURING the descend, while
            # still gripped, which swings its base center by up to 4 cm.
            # Predicting that swing open-loop is what the earlier
            # flatten-shift correction tried; measured against the physics
            # it over-corrected by its own magnitude (predicted 22-41 mm of
            # post-release settle where the real settle is under 1 mm — the
            # object is already flat by the time it carries load). The
            # descend is therefore closed-loop instead: seat, measure where
            # the object actually landed, and re-seat on the residual.
            hang_offset = np.median(samples, axis=0)
            settled_offset = _flatten_rotation(ctx.object(self.object_name)[1]) @ hang_offset
            # Horizontal from the settled (post-rock) offset, vertical from
            # the hanging one less a 2 mm press: the descend stops on measured
            # support, so an over-deep z target only means the last
            # millimetre is a press rather than a graze.
            ee_end = np.array([
                target[0] + settled_offset[0],
                target[1] + settled_offset[1],
                target[2] + hang_offset[2] - 0.002,
            ])
            try:
                pts = ctx.plan_cartesian(self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0],
                                         ee_end, frame.approach, frame.lateral,
                                         axis_index=frame.axis_index)
            except IKUnreachable as exc:
                raise SkillFailed("place", "hover", "ik_unreachable") from exc
            ctx.check_path(self.arm, pts)
            self._set_phase("descend")
            grip = ctx._grip_now[self.arm]

            def seated() -> bool:
                # Near-flat on the surface with real support: a first
                # rim-edge touch leaves hollow vessels tilted a few degrees
                # and a couple mm high, and they rock flat as the descend
                # keeps pressing.
                pos, _ = ctx.object(self.object_name)
                return (ctx.object_support_force(self.object_name) > 0.06
                        and abs(float(pos[2]) - target[2]) < 0.003)

            supported = False
            # One delivery plus bounded re-seats. Each pass lifts the (still
            # gripped) object clear, shifts the tool point by the measured
            # residual, and sets it down again; because the object is flat
            # from the first touchdown on, the offset is stable and the
            # residual collapses in one or two passes. The whole block runs
            # inside the seating window: between touchdown and release the
            # object's weight is on the table, so the jaws read slack and the
            # carry audit would otherwise call a correct placement a fumble.
            with ctx.seating(self.arm):
                for attempt in range(PLACE_RESEAT_ATTEMPTS + 1):
                    try:
                        pts = ctx.plan_cartesian(
                            self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0],
                            ee_end, frame.approach, frame.lateral,
                            axis_index=frame.axis_index,
                        )
                    except IKUnreachable as exc:
                        if attempt == 0:
                            raise SkillFailed("place", "hover", "ik_unreachable") from exc
                        break  # keep the seating already achieved
                    ctx.check_path(self.arm, pts)
                    self._set_phase("descend")
                    supported = False
                    for action in ctx.play(self.arm, pts, grip, grip, 4.0):
                        # Yield rather than step directly: the driver owns the
                        # physics tick, and a self-stepped action never reaches
                        # the episode log the data engine replays from.
                        yield action
                        if seated():
                            supported = True
                            break
                    if not supported:
                        # The playback clock ends with the servo short of the
                        # last waypoint; hold the drop target until the object
                        # seats (or genuinely never touches).
                        hold_end = float(ctx.data.time) + 1.5
                        while float(ctx.data.time) < hold_end:
                            yield ctx.action(self.arm, pts[-1], grip)
                            if seated():
                                supported = True
                                break
                    if not supported:
                        if not seated():
                            raise SkillFailed("place", "descend", "no_support")
                        supported = True
                    if self.object_name.startswith(("fork", "spoon")):
                        for correction in range(4):
                            pos, _ = ctx.object(self.object_name)
                            residual = target[:2] - pos[:2]
                            if float(np.linalg.norm(residual)) <= PLACE_RESEAT_TOL_M:
                                break
                            correction_goal = site_pose(ctx.data, f"{self.arm}.ee")[0].copy()
                            correction_goal[:2] += np.clip(residual, -0.004, 0.004)
                            yield from ctx.play_cartesian(
                                self.arm, correction_goal, frame.approach, frame.lateral,
                                1.0, axis_index=frame.axis_index)
                    pos, _ = ctx.object(self.object_name)
                    residual = target[:2] - pos[:2]
                    if float(np.linalg.norm(residual)) <= PLACE_RESEAT_TOL_M:
                        break
                    if attempt == PLACE_RESEAT_ATTEMPTS:
                        break
                    # Lift clear, then aim the tool point at the residual-shifted
                    # drop: the object is flat now, so site-minus-object is the
                    # offset it will keep.
                    site_now, _ = site_pose(ctx.data, f"{self.arm}.ee")
                    ee_end = np.array([
                        site_now[0] + residual[0], site_now[1] + residual[1], ee_end[2],
                    ])
                    try:
                        yield from ctx.play_cartesian(
                            self.arm, site_now + np.array([0.0, 0.0, PLACE_RESEAT_LIFT_M]),
                            frame.approach, frame.lateral, 1.2, axis_index=frame.axis_index,
                        )
                    except (IKUnreachable, SkillFailed):
                        break  # cannot lift to re-seat: keep what we have
        self._set_phase("release")
        # End the carry BEFORE opening: the grasp audit must not read the
        # intentional release as a fumbled grasp, and the exit's scratch
        # audit must treat the placed object as world, not cargo. Open
        # slowly (a 3 s release): a fast open throws the
        # just-supported object ~6 mm as the pinch preload relaxes
        # (measured on the mug).
        ctx.end_carry(self.arm)
        release_aperture = 0.15 if self.object_name.startswith(("fork", "spoon")) else 0.30
        yield from ctx.open_gripper(self.arm, release_aperture, 3.0)
        # Slide the fixed jaw off the object before lifting at all.
        site_now, rot_now = site_pose(ctx.data, f"{self.arm}.ee")
        try:
            yield from ctx.play_cartesian(
                self.arm, site_now + RELEASE_BACKOFF_M * rot_now[:, 0],
                frame.approach, frame.lateral, 1.0,
            )
        except (IKUnreachable, SkillFailed):
            pass  # no room to back off: the plain rise below is the fallback
        # Rise well clear of the placed object before the home sweep: the
        # joint arc dips a few mm early on and the jaw tips catch the rim of
        # a just-placed mug (measured: hooked at site z 0.444 vs rim 0.435).
        # Pinned first (the orientation that seats the grasp is also the one
        # whose rise clears it); the relaxed rungs below only run when every
        # pinned rise fails — relaxing unconditionally regressed 17/20 to
        # 13/20 (measured: slower branches, timeouts, knocked bottles).
        for rise in (0.06, 0.05, 0.04, 0.03):
            try:
                yield from ctx.play_cartesian(
                    self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0]
                    + np.array([0.0, 0.0, rise]),
                    frame.approach, frame.lateral, 2.0, axis_index=frame.axis_index)
                break
            except (IKUnreachable, SkillFailed):
                continue
        else:
            # No pinned rise solved (low south stations): retry small rises
            # with the wrist free — the grip is open, nothing left to protect.
            for rise in (0.03, 0.02, 0.01):
                try:
                    yield from ctx.play_cartesian(
                        self.arm, site_pose(ctx.data, f"{self.arm}.ee")[0]
                        + np.array([0.0, 0.0, rise]),
                        frame.approach, None, 2.0, axis_index=frame.axis_index)
                    break
                except (IKUnreachable, SkillFailed):
                    continue
        # The object is furniture again: `Pick` whitelisted it so the approach
        # could work at contact distance, and leaving it whitelisted would let
        # the home sweep knock a placed object over unnoticed.
        ctx.allowed[self.arm].discard(self.object_name)
        # Home along a checked path; if the arc would clip the placed object
        # (or anything else), retreat toward the table edge and re-plan.
        q_home = np.array(HOME_JOINTS[self.arm][:5])

        def home_points() -> np.ndarray:
            return _joint_segment(arm_q(ctx.data, self.arm), q_home)

        grip_now = ctx._grip_now[self.arm]
        try:
            home_pts = home_points()
            ctx.check_path(self.arm, home_pts)
        except SkillFailed:
            # The straight home arc would clip the placed object; retreat and
            # re-plan. -y first (the tuned table-edge escape), then the other
            # cardinals, then diagonal micro-lifts: south of the comfort
            # stations -y lands in the near-field dead zone while pure +z
            # hits the orientation ceiling (measured), but up-and-sideways
            # keeps the wrist solvable. Every rung is plan- and contact-
            # checked; the ladder only engages when the straight arc fails.
            side = 1.0 if self.arm == "B" else -1.0
            pinned = (
                [0.0, -0.10, 0.0],
                [0.0, 0.10, 0.0],
                [0.10 * side, 0.0, 0.0],
                [-0.10 * side, 0.0, 0.0],
                [-0.05 * side, 0.0, 0.01],
                [-0.05 * side, 0.0, 0.02],
                [0.05 * side, 0.0, 0.01],
                [0.05 * side, 0.0, 0.02],
            )
            # Second-chance rungs, tried only after every pinned rung fails:
            # the same offsets with the wrist free, then up-and-sideways
            # escapes plus plain verticals for the low south stations where
            # every level rung lands in the near-field dead zone. Pinned
            # first is load-bearing — relaxing unconditionally regressed
            # 17/20 to 13/20 (measured).
            relaxed = (
                *pinned,
                [0.05 * side, -0.05, 0.03],
                [-0.05 * side, -0.05, 0.03],
                [0.05 * side, 0.05, 0.03],
                [-0.05 * side, 0.05, 0.03],
                [0.0, 0.0, 0.05],
                [0.0, 0.0, 0.08],
            )
            ladder = [(r, frame.lateral) for r in pinned] + [(r, None) for r in relaxed]
            recovered = False
            for retreat, lateral in ladder:
                back = (site_pose(ctx.data, f"{self.arm}.ee")[0]
                        + np.array(retreat))
                try:
                    yield from ctx.play_cartesian(self.arm, back, frame.approach,
                                                  lateral, 2.5,
                                                  axis_index=frame.axis_index)
                    recovered = True
                    break
                except (IKUnreachable, SkillFailed):
                    continue
            if not recovered:
                raise SkillFailed("place", "retract", "ik_unreachable")
            home_pts = home_points()
            ctx.check_path(self.arm, home_pts)
        grip_now = ctx._grip_now[self.arm]
        yield from ctx.play(self.arm, home_pts, grip_now, grip_now, 3.0)
        self._set_phase("verify")
        q_still = arm_q(ctx.data, self.arm)
        settle_end = float(ctx.data.time) + PLACE_SETTLE_MAX_S
        while float(ctx.data.time) < settle_end:
            yield ctx.action(self.arm, q_still, ctx._grip_now[self.arm])
            if ctx.object_speed(self.object_name) < PLACE_STILL_SPEED:
                break
        pos, _ = ctx.object(self.object_name)
        xy_err = float(np.linalg.norm(pos[:2] - target[:2]))
        z_err = abs(float(pos[2]) - target[2])
        upright = ctx.object_upright(self.object_name)
        disturbed = max(
            (float(np.linalg.norm(np.array(ctx.object(n)[0]) - p)) for n, p in others.items()),
            default=0.0,
        )
        if ctx.object_support_force(self.object_name) <= 0.06:
            raise SkillFailed("place", "verify", "no_support")
        if ctx.object_speed(self.object_name) >= PLACE_STILL_SPEED:
            raise SkillFailed("place", "verify", "misplaced")
        if max(ctx.finger_forces(self.arm, self.object_name)) > PLACE_RELEASED_FORCE_MAX:
            raise SkillFailed("place", "verify", "misplaced")
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
            # Full-servo close: the STS-3215 servo is
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
            ctx.begin_carry(self.arm, "drawer_top", 2.94)
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
            ctx.end_carry(self.arm)
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
            ctx.begin_carry(self.arm, "drawer_top", 2.94)
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
            ctx.end_carry(self.arm)
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


class Hold(Skill):
    """Keep a grasped object still and verified while the other arm works."""

    phases = ("hold",)

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("hold", "hold", "fumble")
        self._set_phase("hold")
        torque = CARRY_GRIP_TORQUE.get(self.object_name, 0.7)
        held = arm_q(ctx.data, self.arm)
        slipped_since: float | None = None
        with ctx.grip_saturation(self.arm, torque):
            while True:
                fixed, moving = ctx.finger_forces(self.arm, self.object_name)
                if min(fixed, moving) <= JAW_FORCE_MIN:
                    now = float(ctx.data.time)
                    if slipped_since is None:
                        slipped_since = now
                    if now - slipped_since > 0.40:
                        raise SkillFailed("hold", "hold", "fumble")
                else:
                    slipped_since = None
                yield ctx.action(self.arm, held, ctx._grip_now[self.arm])


class Handoff(Skill):
    """Table-supported relay: place on the shared spot, park, other arm re-picks."""

    phases = ("relay_place", "from_retract", "regrasp", "verify")

    def __init__(self, object_name: str, from_arm: str, to_arm: str) -> None:
        super().__init__(from_arm, object_name)
        if to_arm not in ("A", "B") or to_arm == from_arm:
            raise ValueError(f"handoff arms must differ, got {from_arm!r} -> {to_arm!r}")
        self.to_arm = to_arm

    def anchor(self) -> np.ndarray:
        """Relay spot, nudged toward the receiving arm's side of the lens."""
        bias = RELAY_ARM_BIAS
        if self.to_arm == "A":
            bias = -RELAY_ARM_BIAS
        return np.array([RELAY_ANCHOR[0] + bias, RELAY_ANCHOR[1], PLACE_Z["table"]])

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("handoff", "relay_place", "dropped")
        target = self.anchor()
        self._set_phase("relay_place")
        place = Place(self.arm, self.object_name, target)
        try:
            yield from place.run(ctx)
        except SkillFailed as exc:
            raise SkillFailed("handoff", "relay_place", "relay_misplaced") from exc
        # Place already drives the from-arm home along a contact-audited path;
        # assert the park rather than assume it, because the receiving arm's
        # approach corridor crosses the relay spot.
        self._set_phase("from_retract")
        parked = site_pose(ctx.data, f"{self.arm}.ee")[0]
        if zone_of(parked) == zone_of(target) and zone_of(target) != "out":
            yield from ctx.play_joint(self.arm, np.array(HOME_JOINTS[self.arm][:5]), 2.5)
            parked = site_pose(ctx.data, f"{self.arm}.ee")[0]
        ctx.latch_hold(self.arm)
        self._set_phase("regrasp")
        pick = Pick(self.to_arm, self.object_name)
        try:
            yield from pick.run(ctx)
        except SkillFailed as exc:
            raise SkillFailed("handoff", "regrasp", "regrasp_missed") from exc
        self._set_phase("verify")
        if ctx.carrying.get(self.to_arm) != self.object_name:
            raise SkillFailed("handoff", "verify", "regrasp_missed")
        if ctx.carrying.get(self.arm) is not None:
            raise SkillFailed("handoff", "verify", "collision")
        fixed, moving = ctx.finger_forces(self.to_arm, self.object_name)
        if min(fixed, moving) <= JAW_FORCE_MIN:
            raise SkillFailed("handoff", "verify", "regrasp_missed")


class Pour(Skill):
    """Tilt a held bottle over a mug until the target fill is reached."""

    phases = ("prepare", "plan", "align", "tilt", "return", "verify")

    def __init__(self, arm: str, object_name: str = "bottle", target: str = "mug",
                 amount: float = 0.6) -> None:
        super().__init__(arm, object_name)
        self.target = target
        self.amount = float(amount)
        if not 0.0 < self.amount <= 1.0:
            raise ValueError(f"pour amount must be in (0, 1], got {amount}")

    def _mouth(self, ctx: TeacherContext) -> tuple[np.ndarray, np.ndarray]:
        """World position of the bottle's pour lip and its up axis."""
        pos, quat = ctx.object(self.object_name)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
        rot = mat.reshape(3, 3)
        return pos + rot[:, 2] * BOTTLE_MOUTH_Z, np.array(rot[:, 2], dtype=np.float64)

    def _interior(self, ctx: TeacherContext) -> tuple[np.ndarray, float]:
        """World center of the mug's interior mouth plane and its rim height."""
        pos, quat = ctx.object(self.target)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
        rot = mat.reshape(3, 3)
        rim = pos + rot[:, 2] * MUG_RIM_Z
        return rim, float(rim[2])

    def _flowing(self, ctx: TeacherContext) -> bool:
        """True when the lip is over the mug interior and tilted past the spill angle."""
        mouth, up = self._mouth(ctx)
        rim, rim_z = self._interior(ctx)
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
        if tilt < POUR_FLOW_TILT_DEG:
            return False
        if float(np.linalg.norm(mouth[:2] - rim[:2])) > MUG_INNER_R:
            return False
        return mouth[2] > rim_z - 0.005

    def _transfer(self, ctx: TeacherContext, seconds: float) -> None:
        """Move liquid from the bottle proxy into the mug proxy."""
        gained = min(POUR_RATE_PER_S * seconds, 1.0 - ctx.fill_fraction(self.target))
        if gained <= 0.0:
            return
        model = ctx.model
        mug_fill = ctx.fill_fraction(self.target) + gained
        set_fill_fraction(model, self.target, mug_fill)
        # Volume is conserved against each proxy's own full-height scale.
        drained = gained * MAX_WATER_HALF_HEIGHT[self.target] / MAX_WATER_HALF_HEIGHT["bottle"]
        set_fill_fraction(model, "bottle", ctx.fill_fraction("bottle") - drained)

    def prepare(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        """Move the receiver into the common pouring workspace using real grasps."""
        receiver = "B" if self.arm == "A" else "A"
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("pour", "prepare", "dropped")
        from dinner_table.scene.objects import MUG_WALL_R, PLATE_RIM_R

        mug_pos, _ = ctx.object(self.target)
        plate_pos, _ = ctx.object("plate")
        clearance = PLATE_RIM_R + MUG_WALL_R + 0.035
        base = np.array([0.0, plate_pos[1] - np.sqrt(
            max(0.0, clearance ** 2 - plate_pos[0] ** 2))])
        # Idempotent against every station this skill can choose: prepare runs
        # twice under ParallelGroup (group setup, then run), so the staged
        # check must accept the comfort grid as well as the base — otherwise
        # the second pass re-stages a settled mug.
        xs = self._COMFORT_XS if receiver == "B" else tuple(-x for x in self._COMFORT_XS)
        valid = [base, *(np.array([x, y]) for x in xs for y in self._COMFORT_YS)]
        if min(float(np.linalg.norm(mug_pos[:2] - s)) for s in valid) < 0.015:
            return  # staged — held or resting at the station, nothing to move
        if ctx.carrying[receiver] not in (None, self.target):
            raise SkillFailed("pour", "prepare", "receiver_busy")
        self._set_phase("prepare")
        if ctx.carrying[receiver] == self.target:
            # Receiver already holds the mug: park the bottle clear of the
            # staging corridor, reseat the mug, then re-take both grasps.
            station = self._choose_station(ctx, receiver, base, clearance, plate_pos)
            yield from Place(self.arm, self.object_name, (0.04, -0.02)).run(ctx)
            ctx.latch_hold()
            yield from Place(receiver, self.target, station).run(ctx)
            ctx.latch_hold()
            yield from Pick(receiver, self.target).run(ctx)
            ctx.latch_hold()
            yield from Pick(self.arm, self.object_name).run(ctx)
            ctx.latch_hold()
        else:
            yield from Place(self.arm, self.object_name, (0.04, -0.02)).run(ctx)
            ctx.latch_hold()
            yield from Pick(receiver, self.target).run(ctx)
            ctx.latch_hold()
            station = self._choose_station(ctx, receiver, base, clearance, plate_pos)
            yield from Place(receiver, self.target, station).run(ctx)
            ctx.latch_hold()
            yield from Pick(self.arm, self.object_name).run(ctx)
            ctx.latch_hold()

    # Receiver comfort grid (arm-B coordinates; mirrored for arm A): the
    # mid-workspace columns north of the near-field singularity zone, where
    # the mug-carry transit demonstrably plans (measured grids: the feasible
    # band sits near y = -0.20 while the plate-relative base drops to
    # y = -0.30 for south plates). Candidates are filtered by plate
    # clearance and dry-planned before use — the grid proposes, IK disposes.
    _COMFORT_XS = (-0.04, 0.02, 0.05, 0.08, 0.11)
    _COMFORT_YS = (-0.22, -0.20, -0.18)

    def _choose_station(self, ctx: TeacherContext, receiver: str, base: np.ndarray,
                        clearance: float, plate_pos: np.ndarray) -> np.ndarray:
        """First plate-clear station whose staging transit dry-plans.

        The straight-line carry transit crosses orientation dead zones from
        some pick configurations even when the goal solves (measured: seed 1
        vs seed 0 with 3 mm-apart stations); a verified shift re-routes the
        segment. Falls back to the base station when nothing dry-plans, so
        execution raises the honest error.
        """
        try:
            if Place(receiver, self.target, base).feasible_align_height(ctx) is not None:
                return base
        except SkillFailed:
            pass
        xs = self._COMFORT_XS if receiver == "B" else tuple(-x for x in self._COMFORT_XS)
        cells = sorted(
            ((x, y) for x in xs for y in self._COMFORT_YS),
            key=lambda c: (c[0] - base[0]) ** 2 + (c[1] - base[1]) ** 2,
        )
        for x, y in cells:
            alt = np.array([x, y])
            if float(np.linalg.norm(alt - plate_pos[:2])) < clearance:
                continue
            try:
                if Place(receiver, self.target, alt).feasible_align_height(ctx) is not None:
                    return alt
            except SkillFailed:
                continue
        return base

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        from dinner_table.teacher.pour_planner import plan_pour, plan_return

        yield from self.prepare(ctx)

        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("pour", "plan", "dropped")
        rim, _ = self._interior(ctx)
        mouth_goal = rim + np.array([0.0, 0.0, 0.052])
        self._set_phase("plan")
        path = plan_pour(ctx, self.arm, self.object_name, mouth_goal, POUR_TILT_DEG)
        # The neck pinch carries the bottle's full weight below the jaw line;
        # the ordinary carry clamp lets it lever out mid-arc (measured: 27 mm
        # slip then jaw unload), so pouring squeezes harder, still bounded.
        with ctx.grip_saturation(self.arm, POUR_GRIP_TORQUE):
            grip = ctx._grip_now[self.arm]
            self._set_phase("align")
            yield from ctx.play(self.arm, path.joints, grip, grip, 8.0)
            self._set_phase("tilt")
            end = float(ctx.data.time) + 3.0
            while float(ctx.data.time) < end:
                before = float(ctx.data.time)
                yield from ctx.hold(self.arm, path.joints[-1], 0.08)
                if self._flowing(ctx) and ctx.fill_fraction(self.target) < self.amount:
                    self._transfer(ctx, float(ctx.data.time) - before)
                if ctx.fill_fraction(self.target) >= self.amount:
                    break
            self._set_phase("return")
            recovery = plan_return(ctx, self.arm, self.object_name)
            yield from ctx.play(self.arm, recovery.joints, grip, grip, 4.0)
            if ctx.object_upright(self.object_name) < 0.98:
                raise SkillFailed("pour", "return", "not_upright")
            if min(ctx.finger_forces(self.arm, self.object_name)) <= JAW_FORCE_MIN:
                raise SkillFailed("pour", "return", "lost_grasp")
        self._set_phase("verify")
        if ctx.fill_fraction(self.target) < 0.8 * self.amount:
            raise SkillFailed("pour", "verify", "spilled")
        if ctx.object_upright(self.object_name) < 0.98:
            raise SkillFailed("pour", "verify", "not_upright")


class ParallelGroup:
    """Advance two arms' skills tick by tick under mutual zone exclusion.

    The primary skill drives the group: when it finishes, the partner's
    coroutine is closed. Each arm contributes only its own half of the merged
    12-dim target, so neither generator's stale "other arm holds" snapshot can
    drag the arm the other generator is driving.
    """

    def __init__(self, primary: Skill, partner: Skill) -> None:
        if primary.arm == partner.arm:
            raise ValueError("a parallel group must use different arms")
        self.primary = primary
        self.partner = partner
        self.claims = ZoneClaims()

    def _claim(self, ctx: TeacherContext) -> None:
        """Claim each arm's working zone under the primary arm (coordinated group).

        A coordinated pair shares its zones by design, so both are claimed by
        one arm: an uncoordinated third motion is still excluded while the two
        members never block each other.
        """
        for skill in (self.primary, self.partner):
            zone = zone_of(site_pose(ctx.data, f"{skill.arm}.ee")[0])
            if not self.claims.claim(self.primary.arm, zone):
                raise SkillFailed("parallel", "claim", "zone_conflict")

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if isinstance(self.primary, Pour) and isinstance(self.partner, Hold):
            yield from self.primary.prepare(ctx)
        self._claim(ctx)
        lead = self.primary.run(ctx)
        follow = self.partner.run(ctx)
        lead_slice = slice(0, 6) if self.primary.arm == "A" else slice(6, 12)
        follow_slice = slice(6, 12) if self.primary.arm == "A" else slice(0, 6)
        try:
            for lead_action in lead:
                try:
                    follow_action = next(follow)
                except StopIteration:
                    raise SkillFailed("parallel", "run", "partner_ended") from None
                merged = np.asarray(lead_action, dtype=np.float64).copy()
                merged[follow_slice] = np.asarray(follow_action, dtype=np.float64)[follow_slice]
                merged[lead_slice] = np.asarray(lead_action, dtype=np.float64)[lead_slice]
                yield merged
        finally:
            follow.close()
            self.claims.release(self.primary.arm)
            ctx.latch_hold()
