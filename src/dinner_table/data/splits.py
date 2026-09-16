"""Seed-disjoint dataset partitioning and skill coverage accounting.

Splits are a pure function of the episode seed (mirroring the generator's
rule), so train and val can never share a seed. Coverage counts skill cells
for the quality report: every exported episode contributes its executed
skills, tagged with the DR profile it ran under.
"""

from __future__ import annotations

import json
from pathlib import Path

from dinner_table.config import DinnerTableError

VAL_MODULO = 10
VAL_REMAINDER = 9


class SplitsError(DinnerTableError):
    """Exception raised for split partitioning failures."""


def split_for_seed(seed: int) -> str:
    """Split assignment; identical to the generator's rule by construction."""
    if int(seed) % VAL_MODULO == VAL_REMAINDER:
        return "val"
    return "train"


def partition(episodes: list[dict]) -> dict[str, list[dict]]:
    """Group episode entries by seed hash; asserts zero seed overlap."""
    splits: dict[str, list[dict]] = {"train": [], "val": []}
    for entry in episodes:
        if "seed" not in entry:
            raise SplitsError(f"episode entry without a seed: {sorted(entry)}")
        splits[split_for_seed(entry["seed"])].append(entry)
    train_seeds = {entry["seed"] for entry in splits["train"]}
    val_seeds = {entry["seed"] for entry in splits["val"]}
    if train_seeds & val_seeds:
        raise SplitsError(f"seed overlap between splits: {sorted(train_seeds & val_seeds)}")
    return splits


def coverage(episodes: list[dict], logs: dict[str, dict]) -> dict:
    """Skill x DR-profile cell counts over exported episodes.

    ``episodes`` are manifest entries; ``logs`` maps episode_id to the loaded
    EpisodeLog dict. A cell counts one episode once per executed skill: the
    ``pour_only`` graph contributes pick, hold, and pour cells.
    """
    cells: dict[str, int] = {}
    skills: dict[str, int] = {}
    for entry in episodes:
        log = logs.get(entry["episode_id"])
        if log is None:
            raise SplitsError(f"missing log for episode {entry['episode_id']}")
        profile = str(entry.get("profile", log.get("dr_profile", "dr_train")))
        seen: set[str] = set()
        for step in log.get("steps", []):
            skill = str(step["skill"])
            if step.get("outcome") != "success" or skill in seen:
                continue
            seen.add(skill)
            skills[skill] = skills.get(skill, 0) + 1
            cell = f"{skill}|{profile}"
            cells[cell] = cells.get(cell, 0) + 1
    return {"skills": skills, "cells": cells, "episodes": len(episodes)}


def write_coverage(report: dict, out: str | Path) -> Path:
    """Write the coverage report as JSON; returns the path written."""
    path = Path(out)
    if path.suffix != ".json":
        path = path / "coverage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
