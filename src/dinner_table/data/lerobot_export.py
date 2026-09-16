"""Episode logs to LeRobot datasets: deterministic re-render plus state columns.

Each successful demonstration is replayed tick for tick (logged actuator
vector, same substep count) while the cameras re-render; joints come from the
replay so images and state stay consistent by construction. Conditioning and
task strings are reconstructed from the manifest's task graph, and goals are
recomputed through the deployed ``goal_for_skill`` computation on replay GT —
never privileged channels beyond the goal vector.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from dinner_table.compat import lerobot_dataset as compat
from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import CONTROL_HZ, PHYSICS_HZ
from dinner_table.data.demo_gen import GRAPHS
from dinner_table.data.splits import coverage, partition, write_coverage
from dinner_table.policies.conditioning import build_state
from dinner_table.reasoning.schema import RelativeTarget, Step
from dinner_table.scene.builder import Scene
from dinner_table.scene.cameras import resize_overhead_policy
from dinner_table.teacher.context import saturation_ctrl
from dinner_table.teacher.teacher_policy import goal_of, group_parallel

logger = logging.getLogger(__name__)

SUBSTEPS = round((1.0 / CONTROL_HZ) * PHYSICS_HZ)


class ExportError(DinnerTableError):
    """Exception raised for dataset export failures."""


@dataclass(frozen=True)
class Conditioning:
    """Per-frame policy conditioning reconstructed from the task graph."""

    skill: str
    arm: str
    object: str | None
    target: str | RelativeTarget | None
    task: str


class _ScenePoses:
    """Read-only GT pose adapter so the export reuses ``goal_of`` verbatim."""

    def __init__(self, scene: Scene) -> None:
        self._scene = scene

    def object(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self._scene.object_pose(name)


def task_string(
    skill: str, object_name: str | None, arm: str, target: str | RelativeTarget | None
) -> str:
    """Skill string for the task channel, e.g. "pick mug arm B -> placemat_2"."""
    parts = [skill]
    if object_name is not None:
        parts.append(object_name)
    parts += ["arm", arm]
    if target is not None:
        if isinstance(target, str):
            parts += ["->", target]
        else:
            parts += ["->", f"{target.relation} {target.anchor}"]
    return " ".join(parts)


def _primary_skill(group: list[Step]) -> Step:
    # NOTE: mirrors teacher_policy._primary: the non-hold member drives a group.
    for step in group:
        if step.skill != "hold":
            return step
    return group[0]


def attribute_frames(frames: list[dict], graph) -> list[Conditioning]:
    """Attribute every frame to its graph step's (skill, arm, object, target).

    Each frame carries the execution group's step id, so adjacent same-skill
    steps need no heuristic splitting. Recovery "home" frames belong to the
    retried group. Parallel groups condition on the primary member.
    """
    groups = group_parallel(graph)
    by_id: dict[int, list[Step]] = {}
    for group in groups:
        for step in group:
            by_id[step.id] = group
    conds: list[Conditioning] = []
    for frame in frames:
        step_id = frame.get("step_id", -1)
        if step_id not in by_id:
            raise ExportError(f"frame references unknown step id {step_id}")
        primary = _primary_skill(by_id[step_id])
        if frame["skill"] == "home" and primary.skill != "home":
            conds.append(
                Conditioning(
                    "home", primary.arm, None, None, task_string("home", None, primary.arm, None)
                )
            )
        else:
            conds.append(
                Conditioning(
                    primary.skill,
                    primary.arm,
                    primary.object,
                    primary.target,
                    task_string(primary.skill, primary.object, primary.arm, primary.target),
                )
            )
    return conds


def _replay_ids(model) -> tuple[dict[str, tuple[int, int, int]], int, int]:
    """Actuator/joint addresses for saturation-faithful replay stepping."""
    ids: dict[str, tuple[int, int, int]] = {}
    for arm in ("A", "B"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}.gripper")
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}.gripper")
        ids[arm] = (act, int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]))
    drawer_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_actuator")
    drawer_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    return ids, drawer_act, int(model.jnt_qposadr[drawer_joint])


def replay_frames(log: dict, graph_name: str):
    """Yield (frame, conditioning, scene) per tick; the scene is pre-state.

    Rebuilds the episode's scene and replays the logged saturation-faithful
    controls. Shared by the LeRobot export and the YOLO label renderer.
    """
    if graph_name not in GRAPHS:
        raise ExportError(f"unknown task graph: {graph_name}")
    conds = attribute_frames(log["frames"], GRAPHS[graph_name])
    scene = Scene(seed=int(log["seed"]), dr_profile=str(log["dr_profile"]))
    scene.hold_safe()
    grip_ids, drawer_act, drawer_qadr = _replay_ids(scene.model)
    for frame, cond in zip(log["frames"], conds):
        yield frame, cond, scene
        _advance(scene, frame, grip_ids, drawer_act, drawer_qadr, log["episode_id"])


def export_episode(log_path: str | Path, graph_name: str, dataset, render: bool = True) -> int:
    """Replay one log into an open dataset; returns the row count (0 if failed).

    Failed demonstrations are skipped: they teach failure, so the taxonomy
    consumes them, not the policy dataset. ``render=False`` skips image capture
    for headless logic tests with a capturing sink.
    """
    log = json.loads(Path(log_path).read_text(encoding="utf-8"))
    if not log["success"]:
        logger.info("skipping failed episode %s", log["episode_id"])
        return 0
    for frame, cond, scene in replay_frames(log, graph_name):
        if render:
            wrist_a = scene.render("wrist_A")
            wrist_b = scene.render("wrist_B")
            overhead = resize_overhead_policy(scene.render("overhead"))
        else:
            wrist_a = wrist_b = overhead = None
        joints = scene.qpos_12()
        step = Step(id=1, skill=cond.skill, arm=cond.arm, object=cond.object, target=cond.target)
        # NOTE: per-tick recompute; the logged goal goes stale once the object moves.
        poses = _ScenePoses(scene)
        state = build_state(joints, cond.skill, cond.arm, cond.object, goal_of(poses, step))
        dataset.add_frame(
            compat.frame_dict(
                cond.task,
                wrist_a,
                wrist_b,
                overhead,
                state,
                np.asarray(frame["action"], dtype=np.float64),
            )
        )
    # NOTE: sequential encoding; the process pool breaks in locked-down runners.
    compat.save_episode(dataset, parallel_encoding=False)
    logger.info("exported episode %s (%d rows)", log["episode_id"], len(log["frames"]))
    return len(log["frames"])


def _advance(
    scene: Scene,
    frame: dict,
    grip_ids: dict[str, tuple[int, int, int]],
    drawer_act: int,
    drawer_qadr: int,
    episode_id: str,
) -> None:
    """Step one tick exactly as the teacher did: base targets plus the logged
    per-substep gripper saturation and drawer servo mode."""
    ctrl = frame.get("ctrl")
    sat = frame.get("sat")
    if not ctrl or not sat:
        raise ExportError(f"episode {episode_id} predates saturation recording")
    scene.set_targets(np.asarray(frame["action"], dtype=np.float64))
    scene.data.ctrl[drawer_act] = float(ctrl[drawer_act])
    for _ in range(SUBSTEPS):
        if sat["drawer_neutral"]:
            scene.data.ctrl[drawer_act] = float(scene.data.qpos[drawer_qadr])
        for arm in ("A", "B"):
            entry = sat[arm]
            if entry is not None:
                act, qadr, vadr = grip_ids[arm]
                scene.data.ctrl[act] = saturation_ctrl(
                    scene.model, scene.data, act, qadr, vadr, entry[0], entry[1]
                )
        mujoco.mj_step(scene.model, scene.data)


def export_dataset(
    demos: str | Path = "demos", root: str | Path = "datasets/dinner", repo_prefix: str = "dinner"
) -> dict:
    """Export a demos directory into train/val LeRobot datasets plus coverage."""
    demos_path = Path(demos)
    manifest = json.loads((demos_path / "manifest.json").read_text(encoding="utf-8"))
    splits = partition(manifest["episodes"])
    exported: list[dict] = []
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        dataset = compat.create_dataset(f"{repo_prefix}/{split}", Path(root) / split)
        rows = 0
        for entry in splits[split]:
            path = demos_path / entry["split"] / f"{entry['episode_id']}.json"
            episode_rows = export_episode(path, entry["graph"], dataset)
            rows += episode_rows
            if episode_rows > 0:
                exported.append(entry)
        dataset.finalize()
        counts[split] = rows
    logs = {
        entry["episode_id"]: json.loads(
            (demos_path / entry["split"] / f"{entry['episode_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        for entry in exported
    }
    report = coverage(exported, logs)
    coverage_path = write_coverage(report, Path(root) / "coverage.json")
    logger.info(
        "exported %d rows under %s; coverage at %s", sum(counts.values()), root, coverage_path
    )
    return {"splits": counts, "episodes": len(exported), "coverage": str(coverage_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Export demos to LeRobot datasets")
    parser.add_argument("--demos", default="demos", help="demo_gen output directory")
    parser.add_argument("--root", default="datasets/dinner", help="dataset output root")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = export_dataset(args.demos, args.root)
    print(f"exported {result['episodes']} episodes {result['splits']} to {args.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
