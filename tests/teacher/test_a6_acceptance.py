"""A6 gates from IMPLEMENTATION_PLAN.md 2773-2781; never xfail failures.

Default: each condition runs all 20 dr_train seeds. For bounded workers set
A6_SEEDS=0 (or 0,1) and A6_RESULTS=/absolute/fresh/run-directory, selecting a
single condition. Partial workers RECORD outcomes, not acceptance passes.
After every condition/seed has been recorded, run test_a6_aggregate with the
same A6_RESULTS and no A6_SEEDS. Missing/stale records fail the aggregate.
Failure injections are separate tests and must also pass for A6 acceptance.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from dinner_table.contracts.geometry import DRAWER_TRAVEL, PLACEMATS, TABLE_TOP_HEIGHT
from dinner_table.scene.builder import Scene
from dinner_table.scene.objects import BOTTLE_NECK_Z
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.ik import IKUnreachable
from dinner_table.teacher.kinematics import arm_q
from dinner_table.teacher.skills import CloseDrawer, OpenDrawer, Pick, Place

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]
OBJECTS = ("plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2")
CONDITIONS = {
    **{f"pick-{name}": ("pick", name, None, None) for name in OBJECTS},
    "place-plate": ("place", "plate", "placemat_1", None),
    "place-mug": ("place", "mug", "placemat_2", None),
    "place-fork_1": ("place", "fork_1", "placemat_1", None),
    "drawer-0.5": ("drawer", None, None, 0.5),
    "drawer-2.0": ("drawer", None, None, 2.0),
}


def _stamp():
    # A batch cannot silently mix results from different controller revisions.
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src/dinner_table").rglob("*.py"))
    paths += sorted((ROOT / "scenes").rglob("*.xml"))
    paths += [Path(__file__), ROOT / "configs/scene/dr_train.yaml"]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture
def seeds():
    raw = os.environ.get("A6_SEEDS")
    if raw is None:
        return tuple(range(20))
    selected = tuple(int(value) for value in raw.split(","))
    assert selected and len(set(selected)) == len(selected)
    assert all(0 <= seed < 20 for seed in selected)
    assert os.environ.get("A6_RESULTS"), "Partial batches require A6_RESULTS"
    return selected


@pytest.fixture
def profiles(tmp_path):
    result = {None: "dr_train"}
    for scale in (0.5, 2.0):
        profile = yaml.safe_load((ROOT / "configs/scene/dr_train.yaml").read_text())
        profile["drawer_friction_scale"] = [scale, scale]
        path = tmp_path / f"friction-{scale}.yaml"
        path.write_text(yaml.safe_dump(profile))
        result[scale] = str(path)
    return result


def _body_name(model, geom):
    return model.body(int(model.geom_bodyid[geom])).name or "world"


def _contacts(model, data):
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > 0:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        yield contact, max(0.0, float(wrench[0]))


def _jaw_bodies(model, arm):
    # Meshes may be unnamed: classify all collision geoms by jaw BODY.
    result = {}
    for geom in range(model.ngeom):
        name = model.geom(geom).name
        for jaw in ("fixed", "moving"):
            if name.startswith(f"{arm}.{jaw}_jaw"):
                result[int(model.geom_bodyid[geom])] = jaw
    assert set(result.values()) == {"fixed", "moving"}
    return result


def _forces(scene, arm, name):
    model, data = scene.model, scene.data
    body = model.body(name).id
    jaws = _jaw_bodies(model, arm)
    values = {"fixed": 0.0, "moving": 0.0, "external": 0.0, "support": 0.0}
    for contact, force in _contacts(model, data):
        b1 = int(model.geom_bodyid[contact.geom1])
        b2 = int(model.geom_bodyid[contact.geom2])
        if body not in (b1, b2):
            continue
        other = b2
        if b2 == body:
            other = b1
        if other in jaws:
            values[jaws[other]] += force
        else:
            values["external"] += force
            # Placement requires measured TABLE support, not a neighbor/arm.
            if model.body(other).name == "table" and contact.pos[2] <= TABLE_TOP_HEIGHT + 0.003:
                values["support"] += force
    return values


def _pose(scene, name):
    body = scene.model.body(name).id
    return scene.data.xpos[body].copy(), scene.data.xmat[body].reshape(3, 3).copy()


def _opening(scene):
    joint = scene.model.joint("drawer_slide").id
    return float(scene.data.qpos[scene.model.jnt_qposadr[joint]])


def _check_pick(scene, arm, name, z0):
    pos, rot = _pose(scene, name)
    force = _forces(scene, arm, name)
    clearance = 0.02
    tilt = 15.0
    if name == "bottle":
        clearance = 0.05
    if name.startswith(("fork", "spoon")):
        tilt = 30.0
    assert pos[2] - z0 >= clearance, f"lift={pos[2] - z0:.6f} < {clearance}"
    assert min(force["fixed"], force["moving"]) > 0.08, force
    assert force["external"] == 0.0, force
    grasp_point = pos
    if name == "bottle":
        # The neck is the contact region; the free-body origin is at the base.
        grasp_point = pos + rot[:, 2] * BOTTLE_NECK_Z
    assert np.linalg.norm(grasp_point - scene.data.site(f"{arm}.ee").xpos) < 0.06
    assert rot[2, 2] >= np.cos(np.deg2rad(tilt)), f"upright={rot[2, 2]}"


def _check_place(scene, arm, name, target, others):
    pos, rot = _pose(scene, name)
    force = _forces(scene, arm, name)
    goal = np.array(PLACEMATS[target])
    assert np.linalg.norm(pos[:2] - goal[:2]) < 0.006, f"xy={pos[:2] - goal[:2]}"
    assert abs(pos[2] - goal[2]) < 0.003, f"z={pos[2] - goal[2]}"
    assert rot[2, 2] > 0.98, f"upright={rot[2, 2]}"
    assert force["support"] > 0.06, force
    assert max(force["fixed"], force["moving"]) < 0.01, force
    body = scene.model.body(name).id
    dof = scene.model.jnt_dofadr[scene.model.body_jntadr[body]]
    assert np.linalg.norm(scene.data.qvel[dof:dof + 3]) < 0.003
    for other, before in others.items():
        assert np.linalg.norm(_pose(scene, other)[0] - before) <= 0.004, other
    if name.startswith(("fork", "spoon")):
        # Catalog flat orientation is yaw=0, long axis +Y (not spawn jitter).
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        assert abs(yaw) < np.deg2rad(10), f"yaw={np.rad2deg(yaw)}"


class Audit:
    """Independent scratch audit of every executed 25Hz waypoint.

    Uses actual executed qpos/object pose (not requested servo controls), so
    the carried object's measured relative transform is retained. Intentional
    jaw/object contact, pick lift-off and place seating are not transit.
    Drawer is an articulated fixture, not free cargo. Never consult allowed.
    """

    def __init__(self, scene):
        self.scene = scene
        self.scratch = mujoco.MjData(scene.model)
        self.errors = set()
        self.samples = 0
        self.path_errors = set()
        self.airborne = set()

    def sample(self, ctx, skill):
        model, data = self.scene.model, self.scene.data
        self.scratch.qpos[:] = data.qpos
        self.scratch.qvel[:] = data.qvel
        mujoco.mj_forward(model, self.scratch)
        self.samples += 1
        if isinstance(skill, OpenDrawer) and skill.phase in ("pull", "push"):
            force = _forces(self.scene, skill.arm, "drawer_top")
            assert min(force["fixed"], force["moving"]) > 0.08, (
                f"drawer {skill.phase}: both-jaw grasp lost {force}"
            )
        transit = isinstance(skill, Place) and skill.phase in ("carry", "hover")
        if isinstance(skill, Pick) and skill.phase in ("lift", "verify"):
            body = model.body(skill.object_name).id
            jaws = _jaw_bodies(model, skill.arm)
            # Lift-off and recontact must use the same recomputed state.
            touching_world = False
            for contact, _ in _contacts(model, self.scratch):
                b1, b2 = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                if body in (b1, b2) and (b2 if b1 == body else b1) not in jaws:
                    touching_world = True
                    break
            if not touching_world:
                self.airborne.add(skill.object_name)
            transit = skill.object_name in self.airborne
        for contact, _ in _contacts(model, self.scratch):
            b1, b2 = _body_name(model, contact.geom1), _body_name(model, contact.geom2)
            a1, a2 = b1.split(".")[0], b2.split(".")[0]
            if a1 in ("A", "B") and a2 in ("A", "B") and a1 != a2:
                self.errors.add(f"arm-arm {b1}/{b2}")
            if (a1 in ("A", "B") and b2 == "table") or (
                a2 in ("A", "B") and b1 == "table"
            ):
                self.errors.add(f"arm-table {b1}/{b2}")
            if transit and skill.object_name in (b1, b2):
                other = b2
                if b2 == skill.object_name:
                    other = b1
                jaw_ids = _jaw_bodies(model, skill.arm)
                if model.body(other).id not in jaw_ids:
                    self.errors.add(f"cargo-world {b1}/{b2}")
        # Exercise the public scratch path audit on the executed waypoint too.
        # Approach/seating contacts are deliberate, so audit free transit only.
        if (isinstance(skill, Place) and skill.phase in ("carry", "hover")
                and not ctx.path_clear(skill.arm, [arm_q(data, skill.arm)], skill.object_name)):
            self.path_errors.add(f"path_clear rejected executed {skill.phase}")


def _run(ctx, skill, audit):
    ctx.begin(180.0)
    generator = skill.run(ctx)
    try:
        for action in generator:
            # Observe the state BEFORE the first release command.
            if isinstance(skill, Place) and skill.phase == "release":
                force = _forces(ctx.scene, skill.arm, skill.object_name)
                if not getattr(skill, "_a6_release_seen", False):
                    pos, _ = _pose(ctx.scene, skill.object_name)
                    assert force["support"] > 0.06, "release without table support"
                    assert abs(pos[2] - TABLE_TOP_HEIGHT) < 0.003
                    skill._a6_release_seen = True
            ctx.step(action)
            audit.sample(ctx, skill)
    finally:
        generator.close()


def _episode(seed, condition, profile):
    scene = ctx = audit = None
    try:
        kind, name, target, _ = CONDITIONS[condition]
        scene = Scene(seed=seed, dr_profile=profile)
        scene.hold_safe()
        ctx = TeacherContext(scene)
        audit = Audit(scene)
        arm = "A"
        if name == "mug":
            arm = "B"
        if kind == "drawer" or name.startswith(("fork", "spoon")):
            _run(ctx, OpenDrawer("A"), audit)
            assert _opening(scene) >= 0.88 * DRAWER_TRAVEL, "drawer prerequisite"
        if kind == "drawer":
            _run(ctx, CloseDrawer("A"), audit)
            assert _opening(scene) <= 0.12 * DRAWER_TRAVEL
        else:
            # Measure AFTER drawer motion; do not use a second spawned scene.
            z0 = _pose(scene, name)[0][2]
            _run(ctx, Pick(arm, name), audit)
            _check_pick(scene, arm, name, z0)
            if kind == "place":
                others = {other: _pose(scene, other)[0] for other in OBJECTS if other != name}
                _run(ctx, Place(arm, name, target), audit)
                _check_place(scene, arm, name, target, others)
        assert audit.samples > 0
        # A6's zero-contact criterion is absolute on successful episodes,
        # independent of the statistical skill-success tolerance.
        return {"ok": True, "detail": "", "contacts": sorted(audit.errors | audit.path_errors)}
    except Exception as exc:  # noqa: BLE001
        # Prerequisites, assertions, and unexpected controller exceptions all
        # remain in the denominator. No skipping/xfail or denominator shrink.
        contacts = sorted(audit.errors | audit.path_errors) if audit is not None else []
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "contacts": contacts}
    finally:
        audit = ctx = scene = None
        gc.collect()


def _gate(condition, rows):
    assert len(rows) == 20
    required = 20
    if CONDITIONS[condition][0] == "place":
        required = 19
    count = sum(row["ok"] for row in rows)
    failures = {seed: row["detail"] for seed, row in enumerate(rows) if not row["ok"]}
    contacts = {seed: row["contacts"] for seed, row in enumerate(rows) if row["ok"] and row["contacts"]}
    assert not contacts, f"{condition}: audit failures in successful episodes {contacts}"
    assert count >= required, f"{condition}: {count}/20, need {required}/20; {failures}"


@pytest.mark.parametrize("condition", CONDITIONS)
def test_a6_condition(condition, seeds, profiles):
    directory = os.environ.get("A6_RESULTS")
    stamp = _stamp()
    rows = []
    for seed in seeds:
        row = _episode(seed, condition, profiles[CONDITIONS[condition][3]])
        rows.append(row)
        print(f"A6 {condition} seed={seed}: {row}", flush=True)
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            record = {"condition": condition, "seed": seed, "stamp": stamp, **row}
            temporary = path / f"{condition}-{seed}.json.tmp"
            temporary.write_text(json.dumps(record))
            temporary.replace(path / f"{condition}-{seed}.json")
    if seeds == tuple(range(20)):
        _gate(condition, rows)
    else:
        print("PARTIAL RECORDING ONLY: test_a6_aggregate is the acceptance gate", flush=True)


def test_a6_aggregate(request):
    directory = os.environ.get("A6_RESULTS")
    if directory is None:
        # Ordinary full-suite invocation already gates all 20 seeds. Running
        # this node alone without records must never produce a false pass.
        assert os.environ.get("A6_SEEDS") is None
        selected = {item.name for item in request.session.items}
        assert all(f"test_a6_condition[{name}]" in selected for name in CONDITIONS), (
            "Standalone aggregate requires A6_RESULTS"
        )
        return
    assert os.environ.get("A6_SEEDS") is None, "Aggregate requires all seeds"
    stamp = _stamp()
    errors = []
    for condition in CONDITIONS:
        rows = []
        for seed in range(20):
            path = Path(directory) / f"{condition}-{seed}.json"
            assert path.is_file(), f"missing {path}"
            row = json.loads(path.read_text())
            assert (row["condition"], row["seed"], row["stamp"]) == (condition, seed, stamp), path
            assert type(row["ok"]) is bool
            rows.append(row)
        try:
            _gate(condition, rows)
        except AssertionError as exc:
            errors.append(str(exc))
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("injection", ["displaced", "unreachable", "missed_grasp", "no_support", "unheld"])
def test_a6_failure_injection(injection, monkeypatch):
    scene = ctx = audit = None
    try:
        scene = Scene(seed=0, dr_profile="dr_train")
        scene.hold_safe()
        ctx = TeacherContext(scene)
        audit = Audit(scene)
        skill = Pick("B", "mug")
        expected = "missed_grasp"
        if injection == "displaced":
            body = scene.model.body("mug").id
            adr = scene.model.jnt_qposadr[scene.model.body_jntadr[body]]
            scene.data.qpos[adr] -= 0.10
            mujoco.mj_forward(scene.model, scene.data)
        elif injection == "unreachable":
            def unreachable(*args, **kwargs):
                raise IKUnreachable("B.ee", np.array([10.0, 10.0, 10.0]))
            monkeypatch.setattr(ctx, "plan_ik", unreachable)
            expected = "ik_unreachable"
        elif injection == "missed_grasp":
            monkeypatch.setattr(ctx, "grasp_verified", lambda *a, **kw: False)
        elif injection == "no_support":
            z0 = _pose(scene, "mug")[0][2]
            _run(ctx, skill, audit)
            _check_pick(scene, "B", "mug", z0)  # failed prerequisite is a FAILURE
            monkeypatch.setattr(ctx, "object_support_force", lambda *a: 0.0)
            skill = Place("B", "mug", "placemat_2")
            expected = "no_support"
        elif injection == "unheld":
            skill = Place("B", "mug", "placemat_2")
            expected = "dropped"
        z0 = _pose(scene, "mug")[0][2]
        try:
            _run(ctx, skill, audit)
        except SkillFailed as exc:
            assert exc.phase and exc.skill
            assert exc.cause == expected, str(exc)
        else:
            assert injection == "displaced", f"{injection}: false success"
            _check_pick(scene, "B", "mug", z0)
    finally:
        audit = ctx = scene = None
        gc.collect()
