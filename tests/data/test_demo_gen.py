"""A9 data engine: smoke episodes, determinism, focus slices, event mechanics.

Smoke expectations are pinned nominal (unperturbed) frame counts per
graph and seed; the +-10% band absorbs cross-machine physics drift, not
behavioral change. Perturbed determinism is asserted exactly: one seed is
one demonstration, bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from dinner_table.data.demo_gen import (
    FULL_COUNT,
    GRAPHS,
    allocate_graphs,
    generate_dataset,
    generate_episode,
    profile_with_overrides,
    split_for_seed,
)
from dinner_table.data.noise import Event, Perturber
from dinner_table.scene.builder import Scene
from dinner_table.scene.objects import OBJECT_CATALOG
from dinner_table.teacher.context import TeacherContext

pytestmark = pytest.mark.slow

SMOKE = (
    ("park_cycle", 0, 154),
    ("drawer_cycle", 0, 1297),
    ("plate_setting", 0, 1174),
    ("mug_setting", 0, 1133),
    ("pour_only", 0, 4537),
)
LOG_KEYS = {
    "episode_id",
    "seed",
    "dr_profile",
    "instruction",
    "task_id",
    "steps",
    "frames",
    "success",
    "water_fraction",
}
FRAME_KEYS = {"tick", "joints", "action", "skill", "phase", "goal_xyz", "event"}


def _read_log(out: Path, entry: dict) -> dict:
    path = out / entry["split"] / f"{entry['episode_id']}.json"
    assert path.is_file(), f"missing episode file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _check_schema(log: dict, entry: dict) -> None:
    assert LOG_KEYS <= set(log), set(log)
    assert log["seed"] == entry["seed"]
    assert log["episode_id"] == entry["episode_id"]
    assert log["success"] == entry["success"]
    assert len(log["frames"]) == entry["frames"] > 0
    for step in log["steps"]:
        assert step["outcome"] in ("success", "failed")
        assert step["attempts"] >= 1
    ticks = []
    for frame in log["frames"]:
        assert FRAME_KEYS <= set(frame), set(frame)
        assert len(frame["joints"]) == 12
        assert len(frame["action"]) == 12
        assert len(frame["goal_xyz"]) == 3
        ticks.append(frame["tick"])
    assert ticks == sorted(ticks), "frame ticks must increase monotonically"


@pytest.mark.parametrize("graph,seed,expected", SMOKE)
def test_smoke_episodes(graph: str, seed: int, expected: int, tmp_path: Path) -> None:
    """One nominal episode per skill family: valid log, expected duration."""
    entry = generate_episode(seed, graph, "dr_train", tmp_path, probability=0.0)
    assert entry["split"] == split_for_seed(seed)
    log = _read_log(tmp_path, entry)
    _check_schema(log, entry)
    assert log["success"], f"{graph} seed {seed} failed: {log['steps']}"
    assert abs(len(log["frames"]) - expected) <= 0.10 * expected, (
        f"{graph}: {len(log['frames'])} frames, expected {expected} +-10%"
    )


def test_determinism_seed_42(tmp_path: Path) -> None:
    """Re-running episode seed 42 reproduces the action history exactly."""
    first = generate_episode(42, "plate_setting", "dr_train", tmp_path / "a", probability=0.35)
    second = generate_episode(42, "plate_setting", "dr_train", tmp_path / "b", probability=0.35)
    assert first["episode_id"] == second["episode_id"]
    log_a = _read_log(tmp_path / "a", first)
    log_b = _read_log(tmp_path / "b", second)
    assert log_a["success"] and log_b["success"]
    assert [f["action"] for f in log_a["frames"]] == [f["action"] for f in log_b["frames"]]
    assert [f["joints"] for f in log_a["frames"]] == [f["joints"] for f in log_b["frames"]]
    assert [f["event"] for f in log_a["frames"]] == [f["event"] for f in log_b["frames"]]


def test_event_annotation(tmp_path: Path) -> None:
    """A forced schedule annotates its frames with the fired event names."""
    entry = generate_episode(0, "plate_setting", "dr_train", tmp_path, probability=1.0)
    log = _read_log(tmp_path, entry)
    annotated = [f for f in log["frames"] if f["event"]]
    assert annotated, "probability=1.0 must fire at least one event"
    for frame in annotated:
        for name in frame["event"].split(","):
            assert name in Perturber.EVENTS, frame["event"]


def _bottle_mass(seed: int, profile: str) -> float:
    scene = Scene(seed=seed, dr_profile=profile)
    return float(scene.model.body_mass[scene.model.body("bottle").id])


def test_focus_slice(tmp_path: Path) -> None:
    """Focus generates only the requested skill with the DR overrides applied."""
    focus = {"skills": ["pour"], "dr": {"bottle_mass": [1.5, 2.5]}, "count": 2}
    manifest = generate_dataset(focus=focus, out=tmp_path, seed_base=0)
    assert len(manifest["episodes"]) == 2
    assert manifest["focus"] == focus
    for entry in manifest["episodes"]:
        assert entry["graph"] == "pour_only"
        log = _read_log(tmp_path, entry)
        assert log["task_id"] == "pour_only"
        assert {s["skill"] for s in log["steps"]} <= {"pick", "hold", "pour"}
    profile = manifest["profile"]
    assert Path(profile).is_file()
    # The bottle body also carries the density-derived water proxy, which DR
    # never scales; only the vessel's own 0.08 kg nominal takes the override.
    vessel = OBJECT_CATALOG["bottle"].mass_kg
    identity = profile_with_overrides(
        "dr_train", {"mass_scale": [1.0, 1.0]}, tmp_path, name="identity"
    )
    water = _bottle_mass(0, identity) - vessel
    assert water > 0.0
    for entry in manifest["episodes"]:
        mass = _bottle_mass(entry["seed"], profile)
        assert 1.5 * vessel + water <= mass <= 2.5 * vessel + water, f"bottle mass {mass}"
    exact = profile_with_overrides("dr_train", {"bottle_mass": [2.0, 2.0]}, tmp_path, name="exact")
    assert _bottle_mass(5, exact) == pytest.approx(2.0 * vessel + water)


@pytest.mark.fast
def test_split_rule_is_seed_disjoint() -> None:
    """Every 10th seed validates; the same seed never changes split."""
    splits = [split_for_seed(seed) for seed in range(100)]
    assert splits.count("val") == 10
    assert {seed for seed in range(100) if split_for_seed(seed) == "val"} == set(range(9, 100, 10))


@pytest.mark.fast
def test_mix_allocation_matches_plan() -> None:
    """The full mix is exactly the dataset-plan episode budget."""
    names = allocate_graphs(FULL_COUNT, np.random.default_rng(0))
    assert len(names) == 3400
    assert (
        names.count("plate_setting") + names.count("mug_setting") + names.count("cutlery_pair")
        == 1200
    )
    assert names.count("drawer_cycle") == 400
    assert names.count("bottle_relay") == 500
    assert names.count("pour_only") == 400
    assert names.count("park_cycle") == 300
    assert names.count("dinner_canonical") + names.count("dinner_full") == 600


@pytest.mark.fast
def test_schedule_is_deterministic() -> None:
    """The same rng state yields the same events; ~35% of episodes fire."""
    graph = GRAPHS["pour_only"]
    first = Perturber().schedule(np.random.default_rng(7), graph, 1.0)
    second = Perturber().schedule(np.random.default_rng(7), graph, 1.0)
    assert first == second
    assert len(first) >= 1
    assert all(isinstance(e.tick, int) and e.tick > 0 for e in first)
    assert all(e.name in Perturber.EVENTS for e in first)
    hits = sum(
        1 for seed in range(1000) if Perturber().schedule(np.random.default_rng(seed), graph, 0.35)
    )
    assert 300 <= hits <= 400, f"episode rate {hits / 1000:.3f}, expected ~0.35"


@pytest.mark.fast
def test_perturbation_mechanics() -> None:
    """Each event type moves the quantity it names on a live context."""
    scene = Scene(seed=0, dr_profile="dr_train")
    scene.hold_safe()
    ctx = TeacherContext(scene)
    ctx.begin(60.0)
    perturber = Perturber()
    perturber.schedule(np.random.default_rng(0), GRAPHS["plate_setting"], 0.0)

    bid = scene.model.body("bottle").id
    mass = float(scene.model.body_mass[bid])
    pos0, _ = ctx.object("bottle")
    assert ctx.object_speed("bottle") == pytest.approx(0.0, abs=5e-3)
    perturber.apply(
        Event("object_kick", 0, object="bottle", vector=(1.0, 0.0, 0.0)), scene.data, ctx
    )
    assert ctx.object_speed("bottle") == pytest.approx(min(0.05 / mass, 0.5), abs=5e-3)
    scene.settle(1.0)
    pos1, _ = ctx.object("bottle")
    slid = float(np.linalg.norm(pos1[:2] - pos0[:2]))
    assert 0.001 < slid < 0.15, f"bottle slid {slid:.4f} m"

    grip0 = float(scene.data.qpos[ctx._jadr["A"]["gripper"]])
    perturber.apply(Event("gripper_slip", 0, arm="A", vector=(0.02,)), scene.data, ctx)
    grip1 = float(scene.data.qpos[ctx._jadr["A"]["gripper"]])
    assert grip1 == pytest.approx(grip0 + 0.02)

    perturber.apply(Event("waypoint_jitter", 0), scene.data, ctx)
    assert ctx._waypoint_jitter is not None
    assert ctx._waypoint_jitter[0] == pytest.approx(0.035)

    fork0, _ = ctx.object("fork_1")
    perturber.apply(
        Event("mid_skill_retarget", 0, object="fork_1", vector=(0.03, 0.0)), scene.data, ctx
    )
    fork1, _ = ctx.object("fork_1")
    assert float(fork1[0] - fork0[0]) == pytest.approx(0.03)

    jid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    dof = int(scene.model.jnt_dofadr[jid])
    damp0 = float(scene.model.dof_damping[dof])
    perturber.apply(Event("drawer_friction_spike", 0, vector=(2.5,)), scene.data, ctx)
    assert float(scene.model.dof_damping[dof]) == pytest.approx(damp0 * 2.5)
