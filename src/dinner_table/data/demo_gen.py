"""Demonstration factory: roll the teacher under DR and noise, save episode logs.

Library plus CLI. ``generate_dataset`` runs the dataset-plan mix of task
graphs (or a ``--focus`` counter-data slice), each episode on its own seed
with a seeded ``Perturber`` attached, and writes one ``EpisodeLog`` JSON per
episode under ``<out>/<split>/`` plus a run ``manifest.json``. Splits are a
pure function of the seed, so train and val never share one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from dinner_table.config import DinnerTableError
from dinner_table.data.noise import Perturber
from dinner_table.reasoning.schema import Step, TaskGraph
from dinner_table.scene.builder import Scene
from dinner_table.scene.randomizer import load_dr_profile
from dinner_table.teacher.task_graphs import CANONICAL_GRAPHS
from dinner_table.teacher.teacher_policy import run_graph

logger = logging.getLogger(__name__)

PARK_GRAPH = TaskGraph(
    task_id="park_cycle",
    instruction="Park both arms at their home poses.",
    steps=[
        Step(id=1, skill="home", arm="A"),
        Step(id=2, skill="home", arm="B"),
    ],
)
GRAPHS: dict[str, TaskGraph] = {**CANONICAL_GRAPHS, "park_cycle": PARK_GRAPH}
MIX: tuple[tuple[str, int], ...] = (
    ("plate_setting", 450),
    ("mug_setting", 450),
    ("cutlery_pair", 300),
    ("drawer_cycle", 400),
    ("bottle_relay", 500),
    ("pour_only", 400),
    ("park_cycle", 300),
    ("dinner_canonical", 450),
    ("dinner_full", 150),
)
FULL_COUNT = sum(weight for _, weight in MIX)
SKILL_GRAPHS: dict[str, list[str]] = {
    "pick": ["plate_setting"],
    "place": ["plate_setting"],
    "open_drawer": ["drawer_cycle"],
    "close_drawer": ["drawer_cycle"],
    "handoff": ["bottle_relay"],
    "hold": ["pour_only"],
    "pour": ["pour_only"],
    "home": ["park_cycle"],
    "retract": ["park_cycle"],
}


class DemoGenError(DinnerTableError):
    """Exception raised for dataset generation failures."""


def split_for_seed(seed: int) -> str:
    """Split assignment: every 10th seed validates, the rest train."""
    if int(seed) % 10 == 9:
        return "val"
    return "train"


def resolve_config_path(config: str) -> Path:
    """Locate a DR profile by name under configs/scene/ or as a file path."""
    path = Path(config)
    if path.is_file():
        return path
    candidate = Path("configs/scene") / f"{config}.yaml"
    if candidate.is_file():
        return candidate
    fallback = Path("configs/scene") / config
    if fallback.is_file():
        return fallback
    raise DemoGenError(f"unable to locate dr profile: {config}")


def profile_with_overrides(
    config: str, overrides: dict, out: str | Path, name: str = "focus"
) -> str:
    """Write the base profile merged with focus DR overrides; returns its path."""
    base = yaml.safe_load(resolve_config_path(config).read_text(encoding="utf-8"))
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    profiles = Path(out) / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    path = profiles / f"{name}.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")
    return str(path)


def allocate_graphs(count: int, rng: np.random.Generator) -> list[str]:
    """Graph names for ``count`` episodes following the MIX weights, shuffled."""
    total = FULL_COUNT
    counts = [count * weight // total for _, weight in MIX]
    remainder = count - sum(counts)
    fractions = sorted(
        range(len(MIX)),
        key=lambda i: count * MIX[i][1] / total - counts[i],
        reverse=True,
    )
    for i in fractions[:remainder]:
        counts[i] += 1
    names = [name for (name, _), n in zip(MIX, counts) for _ in range(n)]
    rng.shuffle(names)
    return names


def generate_episode(
    seed: int, graph_name: str, profile: str, out_dir: str | Path, probability: float = 0.35
) -> dict:
    """Run one teacher episode under perturbation and save its log; returns the entry."""
    if graph_name not in GRAPHS:
        raise DemoGenError(f"unknown task graph: {graph_name}")
    graph = GRAPHS[graph_name]
    scene = Scene(seed=seed, dr_profile=profile)
    scene.hold_safe()
    perturber = Perturber()
    perturber.schedule(np.random.default_rng(seed), graph, probability)
    log = run_graph(scene, graph, seed, perturber)
    split = split_for_seed(seed)
    path = Path(out_dir) / split / f"{log.episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log.to_dict()), encoding="utf-8")
    logger.info(
        "seed %d graph %s split %s success=%s frames=%d",
        seed,
        graph_name,
        split,
        log.success,
        len(log.frames),
    )
    return {
        "episode_id": log.episode_id,
        "seed": int(seed),
        "graph": graph_name,
        "split": split,
        "success": bool(log.success),
        "frames": len(log.frames),
        "profile": str(profile),
    }


def generate_dataset(
    count: int = FULL_COUNT,
    config: str = "dr_train",
    out: str | Path = "demos",
    focus: dict | None = None,
    seed_base: int = 0,
) -> dict:
    """Run ``count`` episodes (or a focus slice) and write the run manifest."""
    if count <= 0:
        raise DemoGenError(f"count must be positive, got {count}")
    rng = np.random.default_rng(seed_base)
    graphs: list[str]
    overrides: dict = {}
    if focus is not None:
        skills = focus.get("skills", [])
        if not skills:
            raise DemoGenError("focus requires a non-empty skills list")
        for skill in skills:
            if skill not in SKILL_GRAPHS:
                raise DemoGenError(f"focus skill has no graph: {skill}")
        count = int(focus.get("count", count))
        pool = [g for skill in skills for g in SKILL_GRAPHS[skill]]
        graphs = [pool[i % len(pool)] for i in range(count)]
        rng.shuffle(graphs)
        overrides = dict(focus.get("dr", {}))
    else:
        graphs = allocate_graphs(count, rng)
    if overrides:
        profile = profile_with_overrides(config, overrides, out)
    else:
        profile = config
    probability = float(load_dr_profile(profile).perturb_event_probability)
    entries = []
    for index, graph_name in enumerate(graphs):
        seed = seed_base + index
        entries.append(generate_episode(seed, graph_name, profile, out, probability))
    entries.sort(key=lambda e: e["seed"])
    manifest = {
        "config": config,
        "focus": focus,
        "seed_base": int(seed_base),
        "profile": str(profile),
        "episodes": entries,
        "successes": sum(1 for e in entries if e["success"]),
    }
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Generate teacher demonstration episodes")
    parser.add_argument("--config", default="dr_train", help="scene DR profile")
    parser.add_argument(
        "--focus",
        default=None,
        help="counter-data JSON, e.g. "
        '\'{"skills": ["pour"], "dr": {"bottle_mass": [1.5, 2.5]}, '
        '"count": 300}\'',
    )
    parser.add_argument("--count", type=int, default=FULL_COUNT, help="episode count")
    parser.add_argument("--out", default="demos", help="output directory")
    parser.add_argument("--seed-base", type=int, default=0, help="first episode seed")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    focus = None
    if args.focus is not None:
        focus = json.loads(args.focus)
    manifest = generate_dataset(args.count, args.config, args.out, focus, args.seed_base)
    print(
        f"wrote {len(manifest['episodes'])} episodes "
        f"({manifest['successes']} successful) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
