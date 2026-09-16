"""Bimanual teacher skills: stiffness hold, table-supported relay, pour.

The handoff is the reference solution's proven relay (see
docs/PLAN_AMENDMENTS.md): the donor arm places the object on a shared-zone
anchor with the full verified-release machinery and parks, then the receiving
arm regrasps it with its own cataloged grasp frame. No airborne
gripper-to-gripper transfer is claimed. The pour is our own skill — the
reference has none — and drives the visual-proxy water state from the
bottle mouth's pose over the mug interior.

The skills here live beside ``skills.py`` rather than inside it because that
module is already at the size limit; the classes are the ones the plan lists
under the bimanual commit.
"""

from __future__ import annotations

from collections.abc import Iterator

import mujoco
import numpy as np

from dinner_table.contracts.geometry import (
    ARM_MOUNTS,
    HOME_JOINTS,
    SHARED_ZONE,
    TABLE_TOP_HEIGHT,
)
from dinner_table.scene.objects import BOTTLE_WALL_R, MUG_WALL_R
from dinner_table.scene.water import set_fill_fraction
from dinner_table.teacher.context import JAW_FORCE_MIN, TICK, SkillFailed, TeacherContext
from dinner_table.teacher.grasp_catalog import GraspCatalog, GraspCatalogError
from dinner_table.teacher.ik import IKUnreachable
from dinner_table.teacher.kinematics import arm_q, site_pose
from dinner_table.teacher.skills import (
    CARRY_GRIP_TORQUE,
    Home,
    Pick,
    Place,
    Skill,
    plan_grasp_and_hover,
)

# Relay anchors inside the measured dual-reach lens, as (x offset, y). The
# plate's spawn jitter reaches into the same lens, so the site is chosen at run
# time: the first candidate that clears every other object. The x offset is
# applied TOWARD THE RECEIVING ARM (the plan's +/-0.04 m allowance): the centre
# line is reachable by both, but a site pushed to the donor's side is outside
# the receiver's envelope and the regrasp fails IK outright (measured: arm A
# regrasping at x = +0.04 fails 3/4 seeds, at x = 0.0 it succeeds).
# Candidate sites as (x offset toward the receiving arm, y). The offsets stay
# inside the plan's +/-0.04 m allowance; the plate spawns straight across this
# band (measured: x -0.05..-0.11, y -0.14..-0.24 over 10 seeds), so a single
# fixed anchor is regularly occupied and the list has to be walked.
RELAY_ARM_BIAS = 0.04
RELAY_ANCHORS = tuple(
    (offset, y)
    for y in (-0.24, -0.21, -0.27, -0.185, -0.29)
    for offset in (0.0, RELAY_ARM_BIAS, 2 * RELAY_ARM_BIAS, -RELAY_ARM_BIAS)
)
# Footprint radii for the relay's clear-site test. A flat centre-to-centre
# clearance has to be sized for the plate (the widest thing on the table) and
# then rejects sites that are in fact wide open next to a mug or a fork
# (measured: 3 of 6 seeds found no site at a flat 0.13 m).
FOOTPRINT_R = {
    # Outer radii, not grasp radii: the plate's disc is 0.066 with a 10 mm rim
    # wall whose centreline is the 0.061 grasp offset.
    "plate": 0.071,
    "mug": MUG_WALL_R,
    "bottle": BOTTLE_WALL_R,
}
DEFAULT_FOOTPRINT_R = 0.02  # cutlery: a 12.6 cm capsule lying flat, half-width
RELAY_MARGIN_M = 0.02  # room for the donor's release swing between footprints
# A dropped jaw force is tolerated for this long before a hold is a fumble:
# the same transient window the carry audit uses (objects re-seat after a
# pendulum swing without the grasp ever being lost).
GRIP_TRANSIENT_S = 0.40

POUR_TILT_RAD = np.deg2rad(78.0)
POUR_TILT_STEPS = 8  # tilt increments; each is a re-solved tool pose
POUR_RAMP_S = 1.6  # total ramp time across the increments
POUR_DWELL_S = 3.2
POUR_STREAM_TILT_RAD = np.deg2rad(55.0)
POUR_RATE_PER_S = 0.42  # mug fill fraction gained per second of established stream
# Measured off the compiled model, not the build recipe: the mug's rim ring
# tops out 78 mm above its base and the bottle's mouth 88 mm above its own.
BOTTLE_MOUTH_Z = 0.088
MUG_RIM_Z = 0.078
MUG_INTERIOR_R = 0.021  # mug bore radius; the mouth must sit inside it
# How far above the rim the mouth is held. A mug standing on the table takes
# the generous clearance; one held up by the other arm puts its rim near the
# top of the SO-101's envelope, where only the tight end still solves.
POUR_CLEARANCE_LADDER = (0.035, 0.025, 0.018, 0.012, 0.008)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix (3,3) of `angle` radians about a unit `axis` (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    cos, sin = np.cos(angle), np.sin(angle)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return cos * np.eye(3) + sin * skew + (1.0 - cos) * np.outer(axis, axis)


def _tilt_rad(ctx: TeacherContext, object_name: str) -> float:
    """Angle (rad) between the object's own up axis and world up."""
    upright = float(np.clip(ctx.object_upright(object_name), -1.0, 1.0))
    return float(np.arccos(upright))


def _mouth_position(ctx: TeacherContext) -> np.ndarray:
    """World position (3,) of the bottle's mouth, from its current pose."""
    pos, quat = ctx.object("bottle")
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return pos + mat.reshape(3, 3) @ np.array([0.0, 0.0, BOTTLE_MOUTH_Z])


def in_shared_zone(position: np.ndarray) -> bool:
    """True if a world position's xy lies inside the shared handover lens."""
    return (
        SHARED_ZONE[0] <= position[0] <= SHARED_ZONE[1]
        and SHARED_ZONE[2] <= position[1] <= SHARED_ZONE[3]
    )


def relay_anchor(ctx: TeacherContext, object_name: str, from_arm: str,
                 to_arm: str) -> np.ndarray:
    """First relay site (3,) that is clear AND graspable by BOTH arms.

    Clearance alone is not enough: the dual-reach lens was measured for a tool
    point at table+0.05, while a relayed bottle is grasped at its neck 8 cm up,
    and a site the donor can set down on is regularly outside the receiver's
    envelope at that height (measured: arm A fails IK at x=+0.04 on 3 of 4
    seeds). Each candidate is therefore checked against the real solver, using
    the grasp frame the object WOULD have once it is standing there.

    ``to_arm`` receives the object, so any x offset leans its way.
    """
    toward = -1.0 if to_arm == "A" else 1.0
    upright = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    own_r = FOOTPRINT_R.get(object_name, DEFAULT_FOOTPRINT_R)
    others = [
        (np.array(ctx.object(name)[0]), FOOTPRINT_R.get(name, DEFAULT_FOOTPRINT_R))
        for name in ("plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2")
        if name != object_name
    ]
    catalog = GraspCatalog()
    for offset, y in RELAY_ANCHORS:
        site = np.array([toward * offset, y, TABLE_TOP_HEIGHT], dtype=np.float64)
        if any(
            float(np.linalg.norm(other[:2] - site[:2])) < own_r + other_r + RELAY_MARGIN_M
            for other, other_r in others
        ):
            continue
        try:
            reachable = True
            for arm in (to_arm, from_arm):
                frame = catalog.frame_at(object_name, site, upright, arm)
                plan_grasp_and_hover(ctx, arm, frame)
        except (IKUnreachable, GraspCatalogError):
            reachable = False
        if reachable:
            return site
    raise SkillFailed("handoff", "relay_place", "no_clear_relay_site")


class Hold(Skill):
    """Keep a grasped object still while the other arm works.

    Runs until the driver closes the generator (the paired skill finished);
    the grasp is verified every tick, and losing it is a fumble.
    """

    phases = ("hold",)

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("hold", "hold", "fumble")
        self._set_phase("hold")
        frame = self.catalog.frame(ctx.scene, self.object_name, self.arm)
        q_hold = arm_q(ctx.data, self.arm)
        torque = CARRY_GRIP_TORQUE.get(self.object_name, frame.grip_torque)
        bad_since: float | None = None
        with ctx.grip_saturation(self.arm, torque):
            while True:
                fixed, moving = ctx.finger_forces(self.arm, self.object_name)
                if min(fixed, moving) <= JAW_FORCE_MIN:
                    now = float(ctx.data.time)
                    if bad_since is None:
                        bad_since = now
                    if now - bad_since > GRIP_TRANSIENT_S:
                        raise SkillFailed("hold", "hold", "fumble")
                else:
                    bad_since = None
                yield ctx.action(self.arm, q_hold, ctx._grip_now[self.arm])


class Handoff(Skill):
    """Table-supported relay: the donor places on a shared anchor, the
    receiver regrasps it there."""

    phases = ("relay_place", "from_retract", "regrasp", "verify")

    def __init__(self, object_name: str, from_arm: str, to_arm: str) -> None:
        super().__init__(from_arm, object_name)
        if to_arm not in ("A", "B") or to_arm == from_arm:
            raise ValueError(f"to_arm must be the other arm, got {to_arm!r}")
        self.to_arm = to_arm

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("handoff", "relay_place", "dropped")
        anchor = relay_anchor(ctx, self.object_name, self.arm, self.to_arm)
        self._set_phase("relay_place")
        try:
            yield from Place(self.arm, self.object_name, {"point": anchor}).run(ctx)
        except SkillFailed as exc:
            raise SkillFailed("handoff", "relay_place", "relay_misplaced") from exc
        self._set_phase("from_retract")
        # The donor must be out of the lens before the receiver enters it;
        # Place ends at home, so this only re-parks a donor left elsewhere.
        if in_shared_zone(site_pose(ctx.data, f"{self.arm}.ee")[0]):
            try:
                yield from Home(self.arm).run(ctx)
            except SkillFailed as exc:
                raise SkillFailed("handoff", "from_retract", "collision") from exc
        self._set_phase("regrasp")
        # The receiving arm is about to act, so the donor becomes the held arm:
        # re-latch it at its parked pose before any of the receiver's actions
        # command it back to wherever it stood when the step began.
        ctx.latch_hold(self.arm)
        try:
            yield from Pick(self.to_arm, self.object_name).run(ctx)
        except SkillFailed as exc:
            # Keep the receiving pick's own cause in the message: "the relay
            # site was out of the receiver's envelope" and "the receiver closed
            # on nothing" need different fixes.
            raise SkillFailed("handoff", "regrasp",
                              f"regrasp_missed ({exc.phase}:{exc.cause})") from exc
        self._set_phase("verify")
        if ctx.carrying.get(self.to_arm) != self.object_name:
            raise SkillFailed("handoff", "verify", "regrasp_missed")
        if ctx.carrying.get(self.arm) is not None:
            raise SkillFailed("handoff", "verify", "relay_misplaced")
        donor_q = arm_q(ctx.data, self.arm)
        if float(np.max(np.abs(donor_q - np.array(HOME_JOINTS[self.arm][:5])))) > 0.15:
            raise SkillFailed("handoff", "verify", "collision")


class Pour(Skill):
    """Tilt a held bottle over the mug until the requested fill is reached.

    The tilt is an IK ORIENTATION target, not a wrist_flex joint override: the
    wrist joint's axis is offset from the grasp point, so a bare joint command
    swings the whole bottle through a ~10 cm arc (into the table, or across the
    mug), while asking IK for a tilted tool frame pivots the bottle roughly in
    place. Because the bottle is rigidly gripped, the mouth's offset in the
    tool frame is constant, so the tool pose that puts the MOUTH over the bore
    at any tilt angle is exact arithmetic rather than a search.
    """

    phases = ("lift", "align", "tilt", "return", "verify")

    def __init__(self, arm: str, object_name: str = "bottle", target: str = "mug",
                 amount: float = 0.6) -> None:
        super().__init__(arm, object_name)
        if target != "mug":
            raise ValueError(f"pour target must be the mug, got {target!r}")
        if not 0.0 < amount <= 1.0:
            raise ValueError(f"pour amount must be in (0, 1], got {amount}")
        self.target = target
        self.amount = float(amount)

    # ---- geometry ---------------------------------------------------------

    def _mouth_in_tool(self, ctx: TeacherContext) -> np.ndarray:
        """The bottle mouth expressed in the tool frame (constant while held)."""
        site_pos, site_rot = site_pose(ctx.data, f"{self.arm}.ee")
        return site_rot.T @ (_mouth_position(ctx) - site_pos)

    def _mouth_target(self, ctx: TeacherContext, clearance: float) -> np.ndarray:
        """Where the mouth must be: over the bore, `clearance` above the rim."""
        mug_pos, _ = ctx.object(self.target)
        return np.array([mug_pos[0], mug_pos[1], float(mug_pos[2]) + MUG_RIM_Z + clearance])

    def _tilt_sign(self, ctx: TeacherContext) -> float:
        """Tilt the bottle AWAY from the arm's own mount, never back over it."""
        site_pos, site_rot = site_pose(ctx.data, f"{self.arm}.ee")
        mount = np.asarray(ARM_MOUNTS[self.arm], dtype=np.float64)
        outward = site_pos[:2] - mount[:2]
        if float(np.linalg.norm(outward)) < 1e-9:
            return 1.0
        # Tilting by +phi about the tool X swings the mouth along -(tool Y).
        return 1.0 if float(np.dot(-site_rot[:3, 1][:2], outward)) >= 0.0 else -1.0

    def _pour_pose(self, ctx: TeacherContext, phi: float, mouth_in_tool: np.ndarray,
                   clearance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tool position/approach/lateral that holds the mouth on target at `phi`."""
        _, site_rot = site_pose(ctx.data, f"{self.arm}.ee")
        axis = site_rot[:, 0]  # tilt about the tool X: the jaw-spread axis
        rot = _axis_rotation(axis, phi) @ site_rot
        position = self._mouth_target(ctx, clearance) - rot @ mouth_in_tool
        return position, rot[:, 2], rot[:, 0]

    def _stream_open(self, ctx: TeacherContext) -> bool:
        """True when the mouth is over the mug bore and tilted past the lip."""
        mouth = _mouth_position(ctx)
        mug_pos, _ = ctx.object(self.target)
        if float(np.linalg.norm(mouth[:2] - mug_pos[:2])) > MUG_INTERIOR_R:
            return False
        if mouth[2] < mug_pos[2] + MUG_RIM_Z:
            return False
        return _tilt_rad(ctx, self.object_name) >= POUR_STREAM_TILT_RAD

    def _transfer(self, ctx: TeacherContext) -> None:
        """Move one tick's worth of water from the bottle into the mug."""
        source = ctx.scene.fill_fraction(self.object_name)
        if source <= 0.0:
            return
        step = min(POUR_RATE_PER_S * TICK, source)
        target_fill = min(1.0, ctx.scene.fill_fraction(self.target) + step)
        set_fill_fraction(ctx.model, self.object_name, source - step)
        set_fill_fraction(ctx.model, self.target, target_fill)

    # ---- motion -----------------------------------------------------------

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("pour", "lift", "dropped")
        frame = self.catalog.frame(ctx.scene, self.object_name, self.arm)
        start_fill = ctx.scene.fill_fraction(self.target)
        mouth_in_tool = self._mouth_in_tool(ctx)

        # Clearance ladder: a mug held in the air puts its rim near the top of
        # the arm's envelope, so a generous pour height simply has no IK
        # solution. Back it off rather than failing the pour; the stream test
        # only needs the mouth above the rim.
        clearances = [c for c in POUR_CLEARANCE_LADDER]
        self._set_phase("lift")
        # Raise the UPRIGHT bottle clear of the rim before travelling, so the
        # body never sweeps through the mug on the way in.
        site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
        lift_to = self._mouth_target(ctx, clearances[0]) - mouth_in_tool
        for drop in (0.0, 0.01, 0.02, 0.03):
            try:
                yield from ctx.play_cartesian(
                    self.arm,
                    np.array([site_pos[0], site_pos[1],
                              float(lift_to[2]) - drop]),
                    frame.approach, frame.lateral, 2.5,
                )
                break
            except (IKUnreachable, SkillFailed):
                continue

        self._set_phase("align")
        sign = self._tilt_sign(ctx)
        # Pick the first clearance whose FULL tilt pose solves, then approach
        # it through a ramp of intermediate angles.
        chosen = None
        for clearance in clearances:
            try:
                pos, approach, lateral = self._pour_pose(
                    ctx, sign * POUR_TILT_RAD, mouth_in_tool, clearance,
                )
                ctx.plan_ik(self.arm, pos, approach, lateral)
            except IKUnreachable:
                continue
            chosen = clearance
            break
        if chosen is None:
            raise SkillFailed("pour", "align", "ik_unreachable")
        try:
            pos, approach, lateral = self._pour_pose(ctx, 0.0, mouth_in_tool, chosen)
            yield from ctx.play_cartesian(self.arm, pos, approach, lateral, 3.0)
        except (IKUnreachable, SkillFailed) as exc:
            raise SkillFailed("pour", "align", "ik_unreachable") from exc

        self._set_phase("tilt")
        reached = 0.0
        for step in range(1, POUR_TILT_STEPS + 1):
            phi = sign * POUR_TILT_RAD * step / POUR_TILT_STEPS
            try:
                pos, approach, lateral = self._pour_pose(ctx, phi, mouth_in_tool, chosen)
                yield from self._move_pouring(ctx, pos, approach, lateral,
                                              POUR_RAMP_S / POUR_TILT_STEPS)
            except (IKUnreachable, SkillFailed):
                break
            reached = phi
            if ctx.scene.fill_fraction(self.target) >= self.amount:
                break
        # Dwell at the deepest tilt reached until the mug has what it asked for.
        dwell_end = float(ctx.data.time) + POUR_DWELL_S
        q_pour = arm_q(ctx.data, self.arm)
        while float(ctx.data.time) < dwell_end:
            if self._stream_open(ctx):
                self._transfer(ctx)
            if ctx.scene.fill_fraction(self.target) >= self.amount:
                break
            yield ctx.action(self.arm, q_pour, ctx._grip_now[self.arm])

        self._set_phase("return")
        for step in range(POUR_TILT_STEPS - 1, -1, -1):
            phi = reached * step / max(POUR_TILT_STEPS, 1)
            try:
                pos, approach, lateral = self._pour_pose(ctx, phi, mouth_in_tool, chosen)
                yield from self._move_pouring(ctx, pos, approach, lateral,
                                              POUR_RAMP_S / POUR_TILT_STEPS)
            except (IKUnreachable, SkillFailed):
                continue

        self._set_phase("verify")
        gained = ctx.scene.fill_fraction(self.target) - start_fill
        if gained < 0.8 * self.amount:
            raise SkillFailed("pour", "verify", "spilled")

    def _move_pouring(self, ctx: TeacherContext, position: np.ndarray,
                      approach: np.ndarray, lateral: np.ndarray, seconds: float):
        """Play one tilt increment, transferring water whenever the stream is open."""
        for action in ctx.play_cartesian(self.arm, position, approach, lateral, seconds):
            if self._stream_open(ctx):
                self._transfer(ctx)
            yield action


class ParallelGroup:
    """Advance one skill per arm tick-by-tick, merging their arm targets.

    Each member yields a full 12-dim action in which only its own arm's slice
    is meaningful; the group splices the two halves. ``Hold`` members run
    until every other member has finished, then are closed.
    """

    def __init__(self, skills) -> None:
        members = list(skills)
        arms = [skill.arm for skill in members]
        if len(set(arms)) != len(arms):
            raise ValueError("a parallel group must use one skill per arm")
        if not any(not isinstance(skill, Hold) for skill in members):
            raise ValueError("a parallel group needs at least one finite skill")
        self.skills = members

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        runners = {skill.arm: skill.run(ctx) for skill in self.skills}
        finite = {skill.arm for skill in self.skills if not isinstance(skill, Hold)}
        last = {arm: None for arm in runners}
        done: set[str] = set()
        try:
            while not finite.issubset(done):
                for arm, runner in runners.items():
                    if arm in done:
                        continue
                    try:
                        last[arm] = next(runner)
                    except StopIteration:
                        done.add(arm)
                merged = np.zeros(12, dtype=np.float64)
                base = ctx.scene.qpos_12()
                merged[:] = base
                for arm, action in last.items():
                    if action is None:
                        continue
                    if arm == "A":
                        merged[0:6] = action[0:6]
                    else:
                        merged[6:12] = action[6:12]
                yield merged
        finally:
            for arm, runner in runners.items():
                if arm not in done:
                    runner.close()
