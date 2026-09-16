"""Teacher-oracle evaluation: run the canonical graphs over a seed range.

Library plus CLI. ``evaluate`` returns the report dict that the CLI writes to
JSON; the CLI exits nonzero when a graph misses the thresholds the teacher
end-to-end tests assert, so the same gate runs in CI and by hand.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from dinner_table.scene.builder import Scene
from dinner_table.teacher.task_graphs import CANONICAL_GRAPHS
from dinner_table.teacher.teacher_policy import EpisodeLog, run_graph

logger = logging.getLogger(__name__)

DEFAULT_REPORT = Path("artifacts/eval/teacher_report.json")
FIRST_ATTEMPT_MIN = 0.95
WITH_RETRY_MIN = 0.98


def parse_seeds(spec: str) -> list[int]:
    """Expand a seed spec like "0-19" or "0,3,7" into a list of seeds."""
    seeds: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def episode_summary(log: EpisodeLog) -> dict:
    """Per-episode record for the report (frames are omitted; they are bulk data)."""
    first_attempt = log.success and all(step.attempts == 1 for step in log.steps)
    return {
        "episode_id": log.episode_id,
        "seed": log.seed,
        "success": bool(log.success),
        "first_attempt": bool(first_attempt),
        "frames": len(log.frames),
        "water_fraction": log.water_fraction,
        "steps": [
            {
                "step_id": step.step_id,
                "skill": step.skill,
                "arm": step.arm,
                "outcome": step.outcome,
                "attempts": step.attempts,
                "phase_at_failure": step.phase_at_failure,
                "failure_cause": step.failure_cause,
            }
            for step in log.steps
        ],
    }


def _graph_report(episodes: list[dict]) -> dict:
    """Aggregate per-skill success and failure causes across a graph's episodes."""
    skills: dict[str, Counter] = {}
    causes: Counter = Counter()
    for episode in episodes:
        for step in episode["steps"]:
            counter = skills.setdefault(step["skill"], Counter())
            counter["runs"] += 1
            if step["outcome"] == "success":
                counter["successes"] += 1
            else:
                causes[f"{step['skill']}/{step['phase_at_failure']}/{step['failure_cause']}"] += 1
    total = max(len(episodes), 1)
    return {
        "episodes": episodes,
        "success_rate": sum(e["success"] for e in episodes) / total,
        "first_attempt_rate": sum(e["first_attempt"] for e in episodes) / total,
        "skills": {
            name: {
                "runs": int(counter["runs"]),
                "successes": int(counter["successes"]),
                "rate": counter["successes"] / max(counter["runs"], 1),
            }
            for name, counter in sorted(skills.items())
        },
        "failure_causes": dict(sorted(causes.items())),
    }


def evaluate(graph_names: list[str], seeds: list[int], dr_profile: str) -> dict:
    """Run each named graph over each seed and build the report dict."""
    if not graph_names or not seeds:
        raise ValueError("evaluation requires at least one graph and one seed")
    graphs: dict[str, dict] = {}
    for name in graph_names:
        if name not in CANONICAL_GRAPHS:
            raise KeyError(f"unknown task graph: {name}")
        graph = CANONICAL_GRAPHS[name]
        episodes = []
        for seed in seeds:
            scene = Scene(seed=seed, dr_profile=dr_profile)
            scene.hold_safe()
            log = run_graph(scene, graph, seed)
            episodes.append(episode_summary(log))
            logger.info("graph %s seed %d success=%s", name, seed, log.success)
        graphs[name] = _graph_report(episodes)
    report = {
        "dr_profile": dr_profile,
        "seeds": list(seeds),
        "graphs": graphs,
        "thresholds": {
            "first_attempt_min": FIRST_ATTEMPT_MIN,
            "with_retry_min": WITH_RETRY_MIN,
            "gated_graphs": list(graphs),
        },
    }
    report["violations"] = threshold_violations(report)
    report["ok"] = not report["violations"]
    return report


def threshold_violations(report: dict) -> list[str]:
    """Threshold breaches, using the same numbers the end-to-end tests assert.

    Every graph present in the report is gated: an evaluation only ever
    contains graphs the caller asked for, so a graph that ran but failed must
    fail the report, and a report with no graphs (or no episodes behind one)
    is a broken run, not a pass.
    """
    violations = []
    if not report["graphs"]:
        return ["evaluation contains no graphs"]
    for name, graph in report["graphs"].items():
        if not graph["episodes"]:
            violations.append(f"{name}: no episodes recorded")
            continue
        if graph["first_attempt_rate"] < FIRST_ATTEMPT_MIN:
            violations.append(
                f"{name}: first-attempt rate {graph['first_attempt_rate']:.2f} "
                f"below {FIRST_ATTEMPT_MIN:.2f}"
            )
        if graph["success_rate"] < WITH_RETRY_MIN:
            violations.append(
                f"{name}: with-retry rate {graph['success_rate']:.2f} "
                f"below {WITH_RETRY_MIN:.2f}"
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Evaluate the privileged teacher oracle")
    parser.add_argument("--graphs", default="dinner_canonical",
                        help="comma-separated graph names, or 'all'")
    parser.add_argument("--seeds", default="0-19", help="seed range, e.g. 0-19 or 0,3,7")
    parser.add_argument("--profile", default="dr_train", help="scene DR profile")
    parser.add_argument("--out", default=str(DEFAULT_REPORT), help="report JSON path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.graphs == "all":
        names = sorted(CANONICAL_GRAPHS)
    else:
        names = [n.strip() for n in args.graphs.split(",") if n.strip()]
    report = evaluate(names, parse_seeds(args.seeds), args.profile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, graph in report["graphs"].items():
        print(f"{name}: success {graph['success_rate']:.2f} "
              f"first-attempt {graph['first_attempt_rate']:.2f}")
    for violation in report["violations"]:
        print(f"threshold violated: {violation}")
    if report["ok"]:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
