"""A6 per-skill verification: pick, place, and drawer skills over DR seeds.

The thresholds are the Phase 2 acceptance levels (pick/drawer >= 98%, place
>= 95%). Measured rates at the time of writing (see docs/PLAN_AMENDMENTS.md,
Amendment 3): pick 8/8 per object over seeds 0-7; drawer open/close 28/28
including both drawer-friction extremes; place mug 6/6, fork_1 6/6,
plate 4/6; cutlery-spoon place and the bottle are known-open (xfail below).

The seed budget here is deliberately bounded so the file stays a usable CI
gate; the wider verification runs are recorded in the amendment.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from dinner_table.contracts.geometry import PLACEMATS
from dinner_table.scene.builder import Scene
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.skills import CloseDrawer, OpenDrawer, Pick, Place, run_skill

pytestmark = pytest.mark.slow

SEEDS = (0, 1, 2, 3)

# object -> (arm, drawer must be open)
GRASPABLE = {
    "plate": ("A", False),
    "mug": ("B", False),
    "fork_1": ("A", True),
    "fork_2": ("A", True),
    "spoon_1": ("A", True),
    "spoon_2": ("A", True),
}
# object -> (arm, placement target)
PLACEABLE = {
    "mug": ("B", "placemat_2"),
    "fork_1": ("A", "fork_setting"),
}


def _episode(seed: int, arms: dict[str, str | None], targets: dict[str, str],
             drawer_open: bool = False) -> tuple[dict[str, str], Scene, TeacherContext]:
    """Run open-drawer (optional) + pick + place per object on ONE context."""
    scene = Scene(seed=seed, dr_profile="dr_train")
    scene.hold_safe()
    ctx = TeacherContext(scene)
    ctx.begin(300.0)
    outcomes: dict[str, str] = {}
    try:
        if drawer_open:
            for action in OpenDrawer("A").run(ctx):
                ctx.step(action)
        for name, arm in arms.items():
            target = targets.get(name)
            try:
                for action in Pick(arm, name).run(ctx):
                    ctx.step(action)
                if target is not None:
                    for action in Place(arm, name, target).run(ctx):
                        ctx.step(action)
            except SkillFailed as exc:
                outcomes[name] = f"{exc.phase}/{exc.cause}"
    except SkillFailed as exc:  # a shared precondition (e.g. the drawer) failed
        for name in arms:
            outcomes.setdefault(name, f"{exc.phase}/{exc.cause}")
    return outcomes, scene, ctx


def _precondition(arms: dict[str, str | None], targets: dict[str, str]):
    return {"arms": arms, "targets": targets,
            "drawer": any(n.startswith(("fork", "spoon")) for n in arms)}


@pytest.mark.parametrize("name", sorted(GRASPABLE))
def test_pick_success_rate(name: str) -> None:
    """Pick every catalog object with >= 98% success across the DR seeds."""
    arm, drawer = GRASPABLE[name]
    ok = 0
    details = []
    for seed in SEEDS:
        outcomes, scene, _ = _episode(seed, {name: arm}, {}, drawer_open=drawer)
        z0 = scene.object_pose(name)[0][2]
        pos, _ = scene.object_pose(name)
        lifted = float(pos[2]) - z0
        success = name not in outcomes and lifted > 0.011
        ok += success
        if not success:
            details.append(f"seed {seed}: {outcomes.get(name, 'no-lift')} lift={lifted:+.4f}")
    rate = ok / len(SEEDS)
    assert rate >= 0.98, f"pick {name}: {ok}/{len(SEEDS)} ({rate:.0%}); {details}"


@pytest.mark.parametrize("name,target", [("mug", "placemat_2"), ("fork_1", "fork_setting")])
def test_place_success_rate(name: str, target: str) -> None:
    """Place the proven objects with >= 95% success and < 8 mm xy error."""
    arm, _ = GRASPABLE.get(name, ("A", True))
    drawer = name.startswith(("fork", "spoon"))
    ok = 0
    details = []
    for seed in SEEDS:
        outcomes, scene, _ = _episode(seed, {name: arm}, {name: target},
                                      drawer_open=drawer)
        pos, _ = scene.object_pose(name)
        px, py, _ = PLACEMATS[target]
        err = float(np.hypot(pos[0] - px, pos[1] - py))
        success = name not in outcomes and err < 0.008
        ok += success
        if not success:
            details.append(f"seed {seed}: {outcomes.get(name, 'ok')} err={err:.4f}")
    rate = ok / len(SEEDS)
    assert rate >= 0.95, f"place {name}: {ok}/{len(SEEDS)} ({rate:.0%}); {details}"


@pytest.mark.parametrize("name,target", [("plate", "placemat_1"), ("spoon_2", "spoon_setting")])
@pytest.mark.xfail(reason="known-open (Amendment 3): placement drifts past the "
                          "8 mm tolerance for the rim pinch / wide-bowl cutlery",
                   strict=False)
def test_place_known_open(name: str, target: str) -> None:
    """Plate and wide-bowl cutlery placement: recorded, not yet within tolerance."""
    arm, _ = GRASPABLE[name]
    outcomes, scene, _ = _episode(0, {name: arm}, {name: target})
    pos, _ = scene.object_pose(name)
    px, py, _ = PLACEMATS[target]
    err = float(np.hypot(pos[0] - px, pos[1] - py))
    assert name not in outcomes and err < 0.008, f"{outcomes.get(name)} err={err:.4f}"


@pytest.mark.xfail(reason="known-open (Amendment 3): the bottle neck side-grasp "
                          "needs a single-axis IK mode our solver lacks",
                   strict=False)
def test_pick_bottle_known_open() -> None:
    outcomes, _, _ = _episode(0, {"bottle": "A"}, {})
    assert not outcomes, outcomes


@pytest.mark.parametrize("friction", [0.5, 2.0])
def test_drawer_open_close(friction: float, tmp_path) -> None:
    """Open then close the drawer at the sampled friction extremes (>= 98%)."""
    import yaml

    data = yaml.safe_load(Path("configs/scene/dr_train.yaml").read_text(encoding="utf-8"))
    data["drawer_friction_scale"] = [friction, friction]
    profile = tmp_path / f"fric_{friction}.yaml"
    profile.write_text(yaml.safe_dump(data), encoding="utf-8")

    ok = 0
    details = []
    for seed in SEEDS:
        scene = Scene(seed=seed, dr_profile=str(profile))
        scene.hold_safe()
        opened = run_skill(scene, OpenDrawer("A"), timeout=90.0)
        open_ok = opened["ok"] and scene.is_drawer_open()
        close_ok = False
        if open_ok:
            closed = run_skill(scene, CloseDrawer("A"), timeout=90.0)
            close_ok = closed["ok"] and float(scene.data.qpos[0]) < 0.12 * 0.116
        ok += open_ok and close_ok
        if not (open_ok and close_ok):
            details.append(f"seed {seed}: open={opened} close_ok={close_ok}")
    rate = ok / len(SEEDS)
    assert rate >= 0.98, f"drawer friction x{friction}: {ok}/{len(SEEDS)}; {details}"


@pytest.mark.parametrize("skill_name,expect_cause", [
    ("open_drawer_open", "already_open"),
    ("close_drawer_closed", "already_closed"),
])
def test_drawer_precondition_causes(skill_name: str, expect_cause: str) -> None:
    """Mis-stated drawer preconditions raise the attributable cause."""
    scene = Scene(seed=0, dr_profile="dr_train")
    scene.hold_safe()
    ctx = TeacherContext(scene)
    ctx.begin(60.0)
    if skill_name == "open_drawer_open":
        act = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_actuator")
        scene.data.ctrl[act] = 0.12
        scene.settle(3.0)
    with pytest.raises(SkillFailed) as excinfo:
        for action in (OpenDrawer("A") if skill_name.startswith("open")
                       else CloseDrawer("A")).run(ctx):
            ctx.step(action)
    assert excinfo.value.cause == expect_cause, excinfo.value


def test_failure_cause_is_attributable() -> None:
    """A displaced object must never be reported as a false success.

    With the plate shoved 12 cm the grasp either recovers at the new pose or
    fails with an attributable cause - it must not claim success with the
    object still at rest.
    """
    scene = Scene(seed=0, dr_profile="dr_train")
    scene.hold_safe()
    body = scene.model.body("plate").id
    adr = int(scene.model.jnt_qposadr[int(scene.model.body_jntadr[body])])
    scene.data.qpos[adr] += 0.12
    mujoco.mj_forward(scene.model, scene.data)
    z0 = float(scene.object_pose("plate")[0][2])
    ctx = TeacherContext(scene)
    ctx.begin(120.0)
    cause = None
    try:
        for action in Pick("A", "plate").run(ctx):
            ctx.step(action)
    except SkillFailed as exc:
        cause = f"{exc.phase}/{exc.cause}"
    lifted = float(scene.object_pose("plate")[0][2]) - z0
    if cause is None:
        assert lifted > 0.011, "reported success but the plate never left the table"
    else:
        assert cause.split("/")[1] in {"missed_grasp", "ik_unreachable", "path_blocked"}


def test_contact_audit_clean_episodes() -> None:
    """Successful episodes imply the scratch-data contact audit stayed clean.

    ``TeacherContext.step`` raises on any un-whitelisted arm contact, so a
    clean completion is itself the audit result; this pins that contract by
    running a full retrieval episode and asserting both the outcome and the
    final grasp/tilt invariants.
    """
    outcomes, scene, ctx = _episode(
        0, {"fork_1": "A"}, {"fork_1": "fork_setting"}, drawer_open=True,
    )
    assert outcomes == {}, outcomes
    pos, _ = scene.object_pose("fork_1")
    px, py, _ = PLACEMATS["fork_setting"]
    assert float(np.hypot(pos[0] - px, pos[1] - py)) < 0.008
    assert ctx.carrying["A"] is None
