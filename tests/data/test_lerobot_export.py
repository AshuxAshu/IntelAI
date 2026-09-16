"""A10 export: schema, determinism, balance, and the privilege audit.

The render-dependent tests need a display (macOS) or EGL (Linux CI): they run
the 20-episode smoke export through real cameras and the real LeRobot writer.
The replay, attribution, task-string, and split tests below them are headless:
they prove the same logic against a capturing sink and synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dinner_table.compat import lerobot_dataset as compat
from dinner_table.contracts.geometry import JOINT_NAMES
from dinner_table.data.demo_gen import FULL_COUNT, GRAPHS, generate_dataset, generate_episode
from dinner_table.data.lerobot_export import (
    ExportError,
    attribute_frames,
    export_dataset,
    export_episode,
    task_string,
)
from dinner_table.data.splits import SplitsError, coverage, partition
from dinner_table.policies.conditioning import (
    SKILLS,
    STATE_DIM,
    build_state,
    object_index,
    skill_index,
)
from dinner_table.reasoning.schema import RelativeTarget

pytestmark = pytest.mark.slow

SMOKE_COUNT = 20
SMOKE_SEED_BASE = 0
CATEGORY_PLAN = {
    "pick_place": 1200,
    "drawer": 400,
    "handoff": 500,
    "pour": 400,
    "home": 300,
    "full": 600,
}
GRAPH_CATEGORY = {
    "plate_setting": "pick_place",
    "mug_setting": "pick_place",
    "cutlery_pair": "pick_place",
    "drawer_cycle": "drawer",
    "bottle_relay": "handoff",
    "pour_only": "pour",
    "park_cycle": "home",
    "dinner_canonical": "full",
    "dinner_full": "full",
}


class CapturingSink:
    """Two-method dataset stand-in recording rows for headless logic tests."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.episodes = 0

    def add_frame(self, row: dict) -> None:
        self.rows.append(row)

    def save_episode(self, parallel_encoding: bool = True) -> None:
        self.episodes += 1


@pytest.fixture(scope="module")
def plate_log_path(tmp_path_factory):
    """One successful plate episode with saturation recording, generated once."""
    out = tmp_path_factory.mktemp("demos_plate")
    entry = generate_episode(0, "plate_setting", "dr_train", out, probability=0.0)
    return out / entry["split"] / f"{entry['episode_id']}.json"


@pytest.fixture(scope="module")
def smoke_export(tmp_path_factory):
    """20-episode smoke generation plus the full export, shared by schema tests."""
    demos = tmp_path_factory.mktemp("demos_smoke")
    root = tmp_path_factory.mktemp("lerobot_smoke")
    manifest = generate_dataset(SMOKE_COUNT, "dr_train", demos, None, SMOKE_SEED_BASE)
    result = export_dataset(demos, root)
    return {"demos": demos, "root": root, "manifest": manifest, "result": result}


def _load(split_dir: Path, split: str):
    return compat.load_dataset(f"dinner/{split}", split_dir)


def test_dataset_schema(smoke_export) -> None:
    """Exported splits load; features, dtypes, fps, videos, tasks per spec."""
    import av

    root = smoke_export["root"]
    for split in ("train", "val"):
        ds = _load(root / split, split)
        assert ds.fps == 25
        assert ds.features["observation.state"]["shape"] == (STATE_DIM,)
        assert ds.features["observation.state"]["dtype"] == "float32"
        assert ds.features["action"]["shape"] == (12,)
        assert ds.features["action"]["dtype"] == "float32"
        for camera in ("wrist_A", "wrist_B", "overhead"):
            key = f"observation.images.{camera}"
            assert ds.features[key]["dtype"] == "video"
            assert tuple(ds.features[key]["shape"]) == (128, 128, 3)
        for column in ("episode_index", "frame_index", "index", "task_index", "timestamp"):
            assert column in ds.hf_dataset.column_names, column
        assert ds.num_frames > 0, f"{split} split exported no rows"
        for camera in ("wrist_A", "wrist_B", "overhead"):
            mp4 = next((root / split / "videos" / f"observation.images.{camera}").rglob("*.mp4"))
            decoded = [f.to_ndarray(format="rgb24") for f in av.open(mp4).decode(video=0)]
            assert decoded, f"{mp4} decoded to no frames"
            assert decoded[0].shape == (128, 128, 3)
            assert decoded[0].dtype == np.uint8
            assert len(decoded) == ds.num_frames
        for task in ds.meta.tasks.index:
            assert task.split(" ")[0] in SKILLS, task


def test_dataset_determinism(tmp_path: Path) -> None:
    """Re-exporting one parallel episode reproduces pixels and columns exactly."""
    out = tmp_path / "demos"
    entry = generate_episode(0, "pour_only", "dr_train", out, probability=0.0)
    assert entry["success"]
    log_path = out / entry["split"] / f"{entry['episode_id']}.json"
    pixels = []
    columns = []
    for attempt in ("a", "b"):
        ds = compat.create_dataset("dinner/det", tmp_path / attempt)
        assert export_episode(log_path, "pour_only", ds) == len(
            json.loads(log_path.read_text(encoding="utf-8"))["frames"]
        )
        ds.finalize()
        back = compat.load_dataset("dinner/det", tmp_path / attempt)
        digest = hashlib.sha256()
        mp4 = next((tmp_path / attempt / "videos" / "observation.images.wrist_A").rglob("*.mp4"))
        import av

        for frame in av.open(mp4).decode(video=0):
            digest.update(frame.to_ndarray(format="rgb24").tobytes())
        pixels.append(digest.hexdigest())
        columns.append(
            (
                np.asarray(back.hf_dataset["observation.state"]),
                np.asarray(back.hf_dataset["action"]),
            )
        )
    assert pixels[0] == pixels[1]
    np.testing.assert_array_equal(columns[0][0], columns[1][0])
    np.testing.assert_array_equal(columns[0][1], columns[1][1])


def test_dataset_balance(smoke_export) -> None:
    """Coverage matches the successful episodes; categories meet scaled minima."""
    manifest = smoke_export["manifest"]
    report = json.loads((smoke_export["root"] / "coverage.json").read_text(encoding="utf-8"))
    successes = [e for e in manifest["episodes"] if e["success"]]
    assert report["episodes"] == len(successes) > 0
    expected: dict[str, int] = {}
    for entry in successes:
        log = json.loads(
            (smoke_export["demos"] / entry["split"] / f"{entry['episode_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        for skill in {step["skill"] for step in log["steps"] if step["outcome"] == "success"}:
            expected[skill] = expected.get(skill, 0) + 1
    assert report["skills"] == expected
    per_category: dict[str, int] = {}
    for entry in successes:
        category = GRAPH_CATEGORY[entry["graph"]]
        per_category[category] = per_category.get(category, 0) + 1
    for category, planned in CATEGORY_PLAN.items():
        if category not in per_category:
            continue
        minimum = max(1, round(planned * len(successes) / FULL_COUNT))
        assert per_category[category] >= minimum, (category, per_category)
    train_seeds = {e["seed"] for e in manifest["episodes"] if e["split"] == "train"}
    val_seeds = {e["seed"] for e in manifest["episodes"] if e["split"] == "val"}
    assert train_seeds and val_seeds
    assert not (train_seeds & val_seeds)


def test_no_privilege_channels(smoke_export, tmp_path: Path) -> None:
    """Dataset state columns equal the audited replay-built values exactly."""
    manifest = smoke_export["manifest"]
    entry = next(e for e in manifest["episodes"] if e["success"] and e["split"] == "train")
    log_path = smoke_export["demos"] / entry["split"] / f"{entry['episode_id']}.json"
    sink = CapturingSink()
    assert export_episode(log_path, entry["graph"], sink, render=False) > 0
    ds = _load(smoke_export["root"] / "train", "train")
    table = ds.hf_dataset
    episode_rows = [i for i in range(table.num_rows) if table[i]["episode_index"] == 0]
    assert len(episode_rows) == len(sink.rows)
    for row_idx, captured in zip(episode_rows, sink.rows):
        np.testing.assert_array_equal(
            np.asarray(table[row_idx]["observation.state"]), captured["observation.state"]
        )
        np.testing.assert_array_equal(np.asarray(table[row_idx]["action"]), captured["action"])


@pytest.mark.fast
def test_partition_is_seed_disjoint() -> None:
    """Splits follow the seed hash with zero overlap; seedless entries raise."""
    entries = [{"seed": seed} for seed in range(100)]
    splits = partition(entries)
    assert len(splits["train"]) == 90
    assert len(splits["val"]) == 10
    assert {e["seed"] for e in splits["val"]} == set(range(9, 100, 10))
    again = partition(entries)
    assert [e["seed"] for e in again["train"]] == [e["seed"] for e in splits["train"]]
    with pytest.raises(SplitsError, match="seed"):
        partition([{"graph": "plate_setting"}])


@pytest.mark.fast
def test_coverage_counts_successful_skills_once() -> None:
    """Cells count one episode once per executed skill under its profile."""
    entries = [
        {"episode_id": "a", "seed": 0, "profile": "dr_train"},
        {"episode_id": "b", "seed": 1, "profile": "custom.yaml"},
    ]
    logs = {
        "a": {
            "steps": [
                {"skill": "pick", "outcome": "success"},
                {"skill": "place", "outcome": "success"},
                {"skill": "pick", "outcome": "success"},
            ]
        },
        "b": {"steps": [{"skill": "pick", "outcome": "failed"}]},
    }
    report = coverage(entries, logs)
    assert report["episodes"] == 2
    assert report["skills"] == {"pick": 1, "place": 1}
    assert report["cells"] == {"pick|dr_train": 1, "place|dr_train": 1}
    with pytest.raises(SplitsError, match="missing log"):
        coverage([{"episode_id": "zzz", "seed": 2}], logs)


@pytest.mark.fast
def test_task_string_shapes() -> None:
    """Task strings follow the skill-object-arm-target vocabulary exactly."""
    assert task_string("pick", "mug", "B", "placemat_2") == "pick mug arm B -> placemat_2"
    assert task_string("pick", "plate", "A", None) == "pick plate arm A"
    assert task_string("home", None, "A", None) == "home arm A"
    relative = RelativeTarget(relation="left_of", anchor="plate")
    assert task_string("place", "fork_1", "A", relative) == "place fork_1 arm A -> left_of plate"


@pytest.mark.fast
def test_attribution_parallel_recovery_relative() -> None:
    """Parallel groups condition on the primary; recovery and relative resolve."""
    graph = GRAPHS["pour_only"]
    frames = [
        {"skill": "pick", "step_id": 1},
        {"skill": "pick", "step_id": 2},
        {"skill": "pour+hold", "step_id": 4},
        {"skill": "pour+hold", "step_id": 4},
    ]
    conds = attribute_frames(frames, graph)
    assert [(c.skill, c.arm, c.object, c.target) for c in conds] == [
        ("pick", "B", "mug", None),
        ("pick", "A", "bottle", None),
        ("pour", "A", "bottle", "mug"),
        ("pour", "A", "bottle", "mug"),
    ]
    assert conds[2].task == "pour bottle arm A -> mug"

    relay = GRAPHS["bottle_relay"]
    frames = [
        {"skill": "pick", "step_id": 1},
        {"skill": "handoff", "step_id": 2},
        {"skill": "home", "step_id": 2},
        {"skill": "handoff", "step_id": 3},
        {"skill": "place", "step_id": 4},
    ]
    conds = attribute_frames(frames, relay)
    assert (conds[2].skill, conds[2].arm, conds[2].object) == ("home", "A", None)
    assert (conds[3].skill, conds[3].arm) == ("handoff", "B")
    assert isinstance(conds[4].target, RelativeTarget)
    assert conds[4].task == "place bottle arm A -> left_of mug"
    with pytest.raises(ExportError, match="step id"):
        attribute_frames([{"skill": "pick", "step_id": 99}], relay)


@pytest.mark.fast
def test_replay_reproduces_logged_joints(plate_log_path: Path) -> None:
    """Saturation-faithful replay hits the logged joints bit-exactly."""
    log = json.loads(plate_log_path.read_text(encoding="utf-8"))
    sink = CapturingSink()
    assert export_episode(plate_log_path, "plate_setting", sink, render=False) == len(log["frames"])
    assert len(sink.rows) == len(log["frames"])
    for captured, frame in zip(sink.rows, log["frames"]):
        np.testing.assert_array_equal(
            np.asarray(captured["observation.state"])[0:10],
            np.asarray(frame["joints"])[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]].astype(np.float32),
        )
        np.testing.assert_array_equal(
            captured["action"], np.asarray(frame["action"]).astype(np.float32)
        )


@pytest.mark.fast
def test_state_construction_matches_build_state(plate_log_path: Path) -> None:
    """Rows equal build_state on replay joints; one-hots decode to the graph step."""
    log = json.loads(plate_log_path.read_text(encoding="utf-8"))
    sink = CapturingSink()
    export_episode(plate_log_path, "plate_setting", sink, render=False)
    conds = attribute_frames(log["frames"], GRAPHS["plate_setting"])
    non_gripper = [i for i, name in enumerate(JOINT_NAMES) if not name.endswith("gripper")]
    checked_static_goal = False
    pick_goal_z: list[float] = []
    for captured, frame, cond in zip(sink.rows, log["frames"], conds):
        state = np.asarray(captured["observation.state"])
        joints = np.asarray(frame["joints"])
        assert state.shape == (STATE_DIM,)
        np.testing.assert_array_equal(state[0:10], joints[non_gripper].astype(np.float32))
        assert state[10] == np.float32(joints[JOINT_NAMES.index("A.gripper")])
        assert state[11] == np.float32(joints[JOINT_NAMES.index("B.gripper")])
        if cond.arm == "A":
            assert state[12:14].tolist() == [1.0, 0.0]
        else:
            assert state[12:14].tolist() == [0.0, 1.0]
        assert int(np.argmax(state[14:23])) == skill_index(cond.skill)
        assert int(np.argmax(state[23:32])) == object_index(cond.object)
        rebuilt = build_state(joints, cond.skill, cond.arm, cond.object, state[32:35])
        np.testing.assert_array_equal(state, rebuilt)
        if frame["skill"] == "pick":
            pick_goal_z.append(float(state[34]))
        if frame["phase"] == "pregrasp" and frame["skill"] == "pick":
            np.testing.assert_allclose(
                state[32:35], np.asarray(frame["goal_xyz"]), atol=1e-3, rtol=1e-3
            )
            checked_static_goal = True
    assert checked_static_goal, "no static-phase pick frames to pin the goal"
    assert max(pick_goal_z) - min(pick_goal_z) > 0.01


@pytest.mark.fast
def test_export_skips_failed_episodes(tmp_path: Path) -> None:
    """Failed logs export zero rows without touching the simulator."""
    log_path = tmp_path / "failed.json"
    log_path.write_text(
        json.dumps(
            {
                "episode_id": "failed",
                "seed": 0,
                "dr_profile": "dr_train",
                "success": False,
                "steps": [],
                "frames": [],
            }
        ),
        encoding="utf-8",
    )
    sink = CapturingSink()
    assert export_episode(log_path, "plate_setting", sink, render=False) == 0
    assert sink.rows == []
    assert sink.episodes == 0
