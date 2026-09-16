"""A7 physics acceptance for the ACTIVE teacher.skills implementations.

No mocks, xfails, pre-filled receivers, teleported grasps, or softened rate
thresholds. Both relay directions and all four pour conditions use seeds 0-19
from dr_train. A failed prerequisite counts against the rate. Contact failures
are unconditional failures, even when the skill success-rate budget permits a
failed manipulation. Run this slow gate explicitly with ``pytest -m slow``.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pytest

from dinner_table.contracts.geometry import HOME_JOINTS, TABLE_TOP_HEIGHT
from dinner_table.executor.workspace import zone_of
from dinner_table.scene.builder import Scene
from dinner_table.scene.objects import (
    BOTTLE_MOUTH_Z,
    MUG_INNER_R,
    MUG_RIM_Z,
    MUG_WALL_R,
    PLATE_RIM_R,
)
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.ik import IKUnreachable
from dinner_table.teacher.kinematics import arm_q, site_pose
from dinner_table.teacher.skills import Handoff, Hold, ParallelGroup, Pick, Place, Pour

pytestmark = pytest.mark.slow
SEEDS = tuple(range(20))
TARGET_FILL = 0.6


@dataclass
class Episode:
    outcomes: dict[str, str] = field(default_factory=dict)
    collisions: list[str] = field(default_factory=list)
    ticks: int = 0
    fill: float = 0.0
    flow_events: int = 0


class PhysicsAudit:
    """Independently audit BOTH arms at each merged commanded waypoint.

    The scratch pose includes both commanded arms, not one arm against a stale
    snapshot of its partner. Also inspect executed contacts, without the
    production audit's penetration tolerance or allowed-body exemptions.
    """

    def __init__(self, ctx: TeacherContext, result: Episode) -> None:
        self.ctx, self.result = ctx, result
        self.scratch = mujoco.MjData(ctx.model)
        self.arm_geoms = {
            arm: {
                gid for gid in range(ctx.model.ngeom)
                if (mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(ctx.model.geom_bodyid[gid])) or "").startswith(arm + ".")
            }
            for arm in ("A", "B")
        }
        self.joints = []
        for arm in ("A", "B"):
            for suffix in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                           "wrist_flex", "wrist_roll", "gripper"):
                jid = ctx.model.joint(f"{arm}.{suffix}").id
                self.joints.append(int(ctx.model.jnt_qposadr[jid]))
        self.previous_fill = ctx.fill_fraction("mug")

    def contacts(self, data, where: str) -> None:
        for contact in data.contact[:data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if ((g1 in self.arm_geoms["A"] and g2 in self.arm_geoms["B"])
                    or (g2 in self.arm_geoms["A"] and g1 in self.arm_geoms["B"])):
                names = [mujoco.mj_id2name(self.ctx.model, mujoco.mjtObj.mjOBJ_BODY,
                                         int(self.ctx.model.geom_bodyid[g]))
                         for g in (g1, g2)]
                detail = f"{where} tick={self.ctx.ticks()} {names} dist={contact.dist:.6g}"
                if detail not in self.result.collisions:
                    self.result.collisions.append(detail)

    def command(self, action: np.ndarray) -> None:
        action = np.asarray(action)
        assert action.shape == (12,) and np.isfinite(action).all()
        self.scratch.qpos[:] = self.ctx.data.qpos
        self.scratch.qvel[:] = 0.0
        for index, address in enumerate(self.joints):
            value = float(action[index])
            if index in (5, 11):
                value = self.ctx.scene._aperture_to_ctrl("A" if index == 5 else "B", value)
            self.scratch.qpos[address] = value
        mujoco.mj_forward(self.ctx.model, self.scratch)
        self.contacts(self.scratch, "commanded")

    def liquid(self) -> None:
        """A fill increment must have actual tilted-mouth-over-interior geometry."""
        fill = self.ctx.fill_fraction("mug")
        if fill > self.previous_fill + 1e-9:
            model, data = self.ctx.model, self.ctx.data
            bottle = model.body("bottle").id
            mug = model.body("mug").id
            bottle_up = data.xmat[bottle].reshape(3, 3)[:, 2]
            mug_up = data.xmat[mug].reshape(3, 3)[:, 2]
            mouth = data.xpos[bottle] + bottle_up * BOTTLE_MOUTH_Z
            rim = data.xpos[mug] + mug_up * MUG_RIM_Z
            assert np.degrees(np.arccos(np.clip(bottle_up[2], -1, 1))) >= 55.0
            assert np.linalg.norm(mouth[:2] - rim[:2]) <= MUG_INNER_R
            assert mouth[2] > rim[2] - 0.005
            self.result.flow_events += 1
        self.previous_fill = fill
        self.result.fill = fill

    def drive(self, skill) -> None:
        self.ctx.latch_hold()
        self.ctx.extend_deadline(180.0)
        generator = skill.run(self.ctx)
        try:
            while True:
                try:
                    action = next(generator)
                except StopIteration:
                    self.liquid()
                    break
                self.liquid()  # coroutine updates water after the previous physics tick
                self.command(action)
                try:
                    self.ctx.step(action)
                finally:
                    self.contacts(self.ctx.data, "executed")
                    self.result.ticks = self.ctx.ticks()
        finally:
            generator.close()


def _scene(seed: int, bottle_mass_scale: float = 1.0) -> Scene:
    scene = Scene(seed=seed, dr_profile="dr_train")
    # Scale only the bottle, including its inertia, on top of its DR sample.
    # Other objects retain their independently generated scene conditions.
    bid = scene.model.body("bottle").id
    original = float(scene.model.body_mass[bid])
    scene.model.body_mass[bid] *= bottle_mass_scale
    scene.model.body_inertia[bid] *= bottle_mass_scale
    mujoco.mj_setConst(scene.model, scene.data)
    assert float(scene.model.body_mass[bid]) == pytest.approx(original * bottle_mass_scale)
    scene.hold_safe()
    return scene


def _failure(exc: Exception) -> str:
    cause = f"; caused by {exc.__cause__}" if exc.__cause__ else ""
    return f"{type(exc).__name__}: {exc}{cause}"


def _relay_episode(seed: int) -> Episode:
    result = Episode()
    scene = _scene(seed)
    ctx = TeacherContext(scene)
    ctx.begin(180.0)
    audit = PhysicsAudit(ctx, result)
    try:
        try:
            audit.drive(Pick("A", "bottle"))
        except (SkillFailed, IKUnreachable) as exc:
            result.outcomes = {direction: "initial pick: " + _failure(exc)
                               for direction in ("A->B", "B->A")}
            return result
        for sender, receiver in (("A", "B"), ("B", "A")):
            direction = f"{sender}->{receiver}"
            skill = Handoff("bottle", sender, receiver)
            try:
                audit.drive(skill)
                assert ctx.carrying[receiver] == "bottle"
                assert ctx.carrying[sender] is None
                assert min(ctx.finger_forces(receiver, "bottle")) > 0.08
                assert max(ctx.finger_forces(sender, "bottle")) < 0.01
                assert ctx.object("bottle")[0][2] > TABLE_TOP_HEIGHT + 0.011
                assert ctx.object_support_force("bottle") < 0.01
                assert np.max(np.abs(arm_q(ctx.data, sender)
                                     - np.asarray(HOME_JOINTS[sender][:5]))) < 0.05
                assert zone_of(site_pose(ctx.data, f"{sender}.ee")[0]) != "shared"
                result.outcomes[direction] = "ok"
            except (SkillFailed, IKUnreachable, AssertionError) as exc:
                result.outcomes[direction] = _failure(exc)
                if sender == "A":
                    result.outcomes["B->A"] = "prerequisite A->B failed: " + _failure(exc)
                break
        return result
    finally:
        # Scene, context and live publisher own cycles containing large MuJoCo
        # arrays. Collect after every seed rather than accumulating 20 models.
        del audit, ctx, scene
        gc.collect()


def _pour_episode(seed: int, mass_scale: float, held: bool) -> Episode:
    result = Episode()
    scene = _scene(seed, mass_scale)
    ctx = TeacherContext(scene)
    ctx.begin(180.0)
    audit = PhysicsAudit(ctx, result)
    stage = "precondition"
    try:
        assert ctx.fill_fraction("mug") < 0.01, "receiver must start empty"
        initial_bottle_fill = ctx.fill_fraction("bottle")
        if held:
            audit.drive(Pick("B", "mug"))
        audit.drive(Pick("A", "bottle"))
        stage = "pour"
        pour = Pour("A", amount=TARGET_FILL)
        audit.drive(ParallelGroup(pour, Hold("B", "mug")) if held else pour)
        assert ctx.fill_fraction("mug") >= 0.8 * TARGET_FILL
        assert result.flow_events > 0, "no physically valid flow was observed"
        assert ctx.fill_fraction("bottle") < initial_bottle_fill
        assert ctx.carrying["A"] == "bottle"
        assert min(ctx.finger_forces("A", "bottle")) > 0.08
        assert ctx.object_upright("bottle") > 0.98, "controlled return must restore upright"
        if held:
            assert ctx.carrying["B"] == "mug"
            assert min(ctx.finger_forces("B", "mug")) > 0.08
        else:
            assert ctx.object_support_force("mug") > 0.06
            assert ctx.carrying["B"] is None
        result.outcomes["pour"] = "ok"
    except (SkillFailed, IKUnreachable, AssertionError) as exc:
        result.outcomes["pour"] = f"{stage}: {_failure(exc)}"
    finally:
        result.fill = ctx.fill_fraction("mug")
        del audit, ctx, scene
        gc.collect()
    return result


def test_pour_station_feasible() -> None:
    """Seed 1's staging transit must dry-plan and execute at the chosen station."""
    result = Episode()
    scene = _scene(1, 0.5)
    ctx = TeacherContext(scene)
    ctx.begin(180.0)
    audit = PhysicsAudit(ctx, result)
    try:
        audit.drive(Pick("A", "bottle"))
        audit.drive(Place("A", "bottle", (0.04, -0.02)))
        audit.drive(Pick("B", "mug"))
        plate_pos, _ = ctx.object("plate")
        clearance = PLATE_RIM_R + MUG_WALL_R + 0.035
        base = np.array([0.0, plate_pos[1] - np.sqrt(max(0.0, clearance**2 - plate_pos[0] ** 2))])
        station = Pour("A")._choose_station(ctx, "B", base, clearance, plate_pos)
        assert Place("B", "mug", station).feasible_align_height(ctx) is not None
        audit.drive(Place("B", "mug", station))
        assert not result.collisions
    finally:
        del audit, ctx, scene
        gc.collect()


def test_relay_success_rate() -> None:
    """Each relay direction must independently succeed on at least 19/20 seeds."""
    results = [(seed, _relay_episode(seed)) for seed in SEEDS]
    collisions = [(seed, r.collisions) for seed, r in results if r.collisions]
    assert not collisions, f"arm-arm contacts are never in the failure budget: {collisions}"
    for direction in ("A->B", "B->A"):
        failed = [(seed, r.outcomes[direction]) for seed, r in results
                  if r.outcomes[direction] != "ok"]
        assert (len(SEEDS) - len(failed)) / len(SEEDS) >= 0.95, (
            f"relay {direction}: {len(SEEDS) - len(failed)}/{len(SEEDS)}; {failed}"
        )


@pytest.mark.parametrize("held", [False, True], ids=["table", "held"])
@pytest.mark.parametrize("mass_scale", [0.5, 2.0], ids=["mass-half", "mass-double"])
def test_pour_success_rate(mass_scale: float, held: bool) -> None:
    """Each mass/support condition needs >=18/20 genuine fills, not a pooled rate."""
    results = [(seed, _pour_episode(seed, mass_scale, held)) for seed in SEEDS]
    collisions = [(seed, r.collisions) for seed, r in results if r.collisions]
    assert not collisions, f"arm-arm contacts are never in the failure budget: {collisions}"
    failed = [(seed, r.outcomes["pour"], r.fill) for seed, r in results
              if r.outcomes["pour"] != "ok"]
    assert (len(SEEDS) - len(failed)) / len(SEEDS) >= 0.90, (
        f"pour held={held} bottle mass x{mass_scale}: "
        f"{len(SEEDS) - len(failed)}/{len(SEEDS)}; {failed}"
    )


@pytest.mark.parametrize("skill", [Hold("B", "mug"), Pour("A")], ids=["hold", "pour"])
def test_requires_physical_grasp(skill) -> None:
    """A skill must not succeed merely because the object exists in the scene."""
    scene = _scene(0)
    ctx = TeacherContext(scene)
    ctx.begin(5.0)
    generator = skill.run(ctx)
    try:
        with pytest.raises(SkillFailed) as caught:
            next(generator)
        assert caught.value.cause == ("fumble" if isinstance(skill, Hold) else "dropped")
        assert ctx.fill_fraction("mug") < 0.01
    finally:
        generator.close()
        del generator, ctx, scene
        gc.collect()
