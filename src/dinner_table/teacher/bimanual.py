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
    HOME_JOINTS,
    SHARED_ZONE,
    TABLE_TOP_HEIGHT,
)
from dinner_table.scene.water import set_fill_fraction
from dinner_table.teacher.context import JAW_FORCE_MIN, TICK, SkillFailed, TeacherContext
from dinner_table.teacher.ik import IKUnreachable
from dinner_table.teacher.kinematics import arm_q, site_pose
from dinner_table.teacher.skills import CARRY_GRIP_TORQUE, Home, Pick, Place, Skill

# Relay anchors inside the measured dual-reach lens, ordered by preference.
# The plate's spawn jitter reaches into the same lens, so the site is chosen
# at run time: the first candidate that clears every other object.
RELAY_ANCHORS = (
    (0.0, -0.24),
    (0.04, -0.24),
    (-0.04, -0.24),
    (0.0, -0.21),
    (0.04, -0.21),
    (-0.04, -0.21),
    (0.0, -0.27),
)
RELAY_CLEARANCE_M = 0.13  # centre-to-centre room for the donor's release swing
# A dropped jaw force is tolerated for this long before a hold is a fumble:
# the same transient window the carry audit uses (objects re-seat after a
# pendulum swing without the grasp ever being lost).
GRIP_TRANSIENT_S = 0.40

POUR_TILT_RAD = np.deg2rad(78.0)
POUR_RAMP_S = 1.4
POUR_DWELL_S = 3.2
POUR_STREAM_TILT_RAD = np.deg2rad(55.0)
POUR_RATE_PER_S = 0.42  # mug fill fraction gained per second of established stream
BOTTLE_MOUTH_Z = 0.088  # neck top above the bottle's base (scene/objects.py)
MUG_INTERIOR_R = 0.021  # mug bore radius; the mouth must sit inside it
MUG_RIM_Z = 0.066
POUR_MOUTH_CLEARANCE_M = 0.055  # mouth height above the mug rim while pouring


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


def relay_anchor(ctx: TeacherContext, object_name: str) -> np.ndarray:
    """Pick the first dual-reach relay site (3,) clear of every other object."""
    others = [
        np.array(ctx.object(name)[0])
        for name in ("plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2")
        if name != object_name
    ]
    for x, y in RELAY_ANCHORS:
        site = np.array([x, y, TABLE_TOP_HEIGHT], dtype=np.float64)
        clear = True
        for other in others:
            if float(np.linalg.norm(other[:2] - site[:2])) < RELAY_CLEARANCE_M:
                clear = False
                break
        if clear:
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
        anchor = relay_anchor(ctx, self.object_name)
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
        try:
            yield from Pick(self.to_arm, self.object_name).run(ctx)
        except SkillFailed as exc:
            raise SkillFailed("handoff", "regrasp", "regrasp_missed") from exc
        self._set_phase("verify")
        if ctx.carrying.get(self.to_arm) != self.object_name:
            raise SkillFailed("handoff", "verify", "regrasp_missed")
        if ctx.carrying.get(self.arm) is not None:
            raise SkillFailed("handoff", "verify", "relay_misplaced")
        donor_q = arm_q(ctx.data, self.arm)
        if float(np.max(np.abs(donor_q - np.array(HOME_JOINTS[self.arm][:5])))) > 0.15:
            raise SkillFailed("handoff", "verify", "collision")


class Pour(Skill):
    """Tilt a held bottle over the mug until the requested fill is reached."""

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

    def _pour_height(self, ctx: TeacherContext) -> float:
        mug_pos, _ = ctx.object(self.target)
        return float(mug_pos[2]) + MUG_RIM_Z + POUR_MOUTH_CLEARANCE_M

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

    def _tilt_sign(self, ctx: TeacherContext) -> float:
        """Wrist-flex direction (+1/-1) that pitches the bottle mouth downward.

        The bottle hangs along the gripper's approach axis, so the sign that
        tilts it is the one that rotates the site's own +Z away from world
        down; it depends on which side of the arm the pour happens on.
        """
        _, rot = site_pose(ctx.data, f"{self.arm}.ee")
        mug_pos, _ = ctx.object(self.target)
        site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
        toward = mug_pos - site_pos
        if float(np.dot(np.cross(rot[:, 2], toward), rot[:, 1])) >= 0.0:
            return 1.0
        return -1.0

    def run(self, ctx: TeacherContext) -> Iterator[np.ndarray]:
        if ctx.carrying.get(self.arm) != self.object_name:
            raise SkillFailed("pour", "lift", "dropped")
        frame = self.catalog.frame(ctx.scene, self.object_name, self.arm)
        torque = CARRY_GRIP_TORQUE.get(self.object_name, frame.grip_torque)
        start_fill = ctx.scene.fill_fraction(self.target)
        with ctx.grip_saturation(self.arm, torque):
            self._set_phase("lift")
            site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
            mouth = _mouth_position(ctx)
            lift_to = np.array([
                site_pos[0], site_pos[1],
                site_pos[2] + (self._pour_height(ctx) - float(mouth[2])),
            ])
            yield from self._reach(ctx, lift_to, frame, "lift")
            self._set_phase("align")
            # Drive the MOUTH over the mug bore, not the gripper: the bottle
            # hangs off the grasp point by its own lever.
            site_pos, _ = site_pose(ctx.data, f"{self.arm}.ee")
            mouth = _mouth_position(ctx)
            mug_pos, _ = ctx.object(self.target)
            align_to = site_pos + np.array([
                mug_pos[0] - mouth[0], mug_pos[1] - mouth[1],
                self._pour_height(ctx) - float(mouth[2]),
            ])
            yield from self._reach(ctx, align_to, frame, "align")
            self._set_phase("tilt")
            q_pour = arm_q(ctx.data, self.arm)
            sign = self._tilt_sign(ctx)
            yield from self._ramp(ctx, q_pour, 0.0, sign * POUR_TILT_RAD, POUR_RAMP_S)
            q_tilted = q_pour.copy()
            q_tilted[3] = q_pour[3] + sign * POUR_TILT_RAD
            dwell_end = float(ctx.data.time) + POUR_DWELL_S
            while float(ctx.data.time) < dwell_end:
                if self._stream_open(ctx):
                    self._transfer(ctx)
                if ctx.scene.fill_fraction(self.target) >= self.amount:
                    break
                yield ctx.action(self.arm, q_tilted, ctx._grip_now[self.arm])
            self._set_phase("return")
            yield from self._ramp(ctx, q_pour, sign * POUR_TILT_RAD, 0.0, POUR_RAMP_S)
        self._set_phase("verify")
        gained = ctx.scene.fill_fraction(self.target) - start_fill
        if gained < 0.8 * self.amount:
            raise SkillFailed("pour", "verify", "spilled")

    def _reach(self, ctx: TeacherContext, target: np.ndarray, frame, phase: str):
        """Cartesian move that keeps the grasp orientation, stepping down on
        an unreachable height rather than failing the pour outright."""
        for drop in (0.0, 0.01, 0.02, 0.03):
            try:
                yield from ctx.play_cartesian(
                    self.arm, target - np.array([0.0, 0.0, drop]),
                    frame.approach, frame.lateral, 3.0,
                )
                return
            except (IKUnreachable, SkillFailed):
                continue
        raise SkillFailed("pour", phase, "ik_unreachable")

    def _ramp(self, ctx: TeacherContext, q_base: np.ndarray, start: float,
              end: float, seconds: float):
        """Ramp the wrist-flex offset from start to end, pouring as it goes."""
        t0 = float(ctx.data.time)
        while float(ctx.data.time) < t0 + seconds:
            frac = (float(ctx.data.time) - t0) / seconds
            q = q_base.copy()
            q[3] = q_base[3] + start + (end - start) * frac
            if self._stream_open(ctx):
                self._transfer(ctx)
            yield ctx.action(self.arm, q, ctx._grip_now[self.arm])


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
