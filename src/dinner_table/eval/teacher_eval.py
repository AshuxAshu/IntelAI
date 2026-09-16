"""Teacher-oracle evaluation: run the canonical graphs over a range of seeds.

Library plus CLI. ``evaluate`` returns the report dict the CLI writes to
``eval/teacher_report.json``; the CLI exits nonzero when any threshold from
``tests/teacher/test_teacher_e2e.py`` is violated, so the same gate runs in CI
and by hand.

The oracle is privileged and deterministic: a (graph, seed) pair always
produces the same episode, so a report is reproducible from its header alone.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from dinner_table.scene.builder import Scene
from dinner_table.teacher.teacher_policy import CANONICAL_GRAPHS, EpisodeLog, run_graph

logger = logging.getLogger(__name__)

DEFAULT_REPORT = Path("eval/teacher_report.json")
# The A8 acceptance thresholds. ``first_attempt`` counts episodes whose every
# step succeeded without a retry; ``with_retries`` counts episodes that
# finished successfully at all. Both are fractions of the episodes run.
CANONICAL_FIRST_ATTEMPT_MIN = 0.95
CANONICAL_WITH_RETRIES_MIN = 0.98


def parse_seeds(spec: str) -> list[int]:
    """Expand a ``"0-19"`` / ``"3"`` / ``"0-4,9"`` seed specification."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if sep:
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(lo))
    if not seeds:
        raise ValueError(f"no seeds in specification {spec!r}")
    return seeds


def _first_attempt(log: EpisodeLog) -> bool:
    """True when the episode succeeded and no step needed a retry."""
    return log.success and all(step.attempts <= 1 for step in log.steps)


def run_episode(graph_name: str, seed: int, profile: str = "dr_train") -> EpisodeLog:
    """Run one canonical graph on one seed and return its log."""
    if graph_name not in CANONICAL_GRAPHS:
        raise KeyError(f"unknown graph {graph_name!r}; have {sorted(CANONICAL_GRAPHS)}")
    scene = Scene(seed=seed, dr_profile=profile)
    return run_graph(scene, CANONICAL_GRAPHS[graph_name], seed, dr_profile=profile)


def _summarize(graph_name: str, logs: list[EpisodeLog]) -> dict:
    """Per-graph rates, per-skill rates, and the failure-cause histogram."""
    total = len(logs)
    skill_counts: dict[str, Counter] = {}
    causes: Counter = Counter()
    for log in logs:
        for step in log.steps:
            counts = skill_counts.setdefault(step.skill, Counter())
            counts[step.outcome] += 1
            if step.outcome != "success" and step.failure_cause is not None:
                causes[f"{step.skill}:{step.phase_at_failure}:{step.failure_cause}"] += 1
    skills = {
        name: {
            "attempted": sum(counts.values()),
            "success": counts["success"],
            "rate": counts["success"] / sum(counts.values()) if counts else 0.0,
        }
        for name, counts in sorted(skill_counts.items())
    }
    successes = sum(1 for log in logs if log.success)
    firsts = sum(1 for log in logs if _first_attempt(log))
    return {
        "graph": graph_name,
        "episodes": total,
        "success": successes,
        "success_rate": successes / total if total else 0.0,
        "first_attempt": firsts,
        "first_attempt_rate": firsts / total if total else 0.0,
        "skills": skills,
        "failure_causes": dict(causes.most_common()),
        "episode_ids": [log.episode_id for log in logs],
        "failed_seeds": [log.seed for log in logs if not log.success],
    }


def _thresholds_for(graph_name: str) -> dict[str, float]:
    """Acceptance thresholds a graph must clear; only the canonical is gated."""
    if graph_name == "canonical":
        return {
            "first_attempt_rate": CANONICAL_FIRST_ATTEMPT_MIN,
            "success_rate": CANONICAL_WITH_RETRIES_MIN,
        }
    return {}


def evaluate(
    seeds: Iterable[int],
    graphs: Iterable[str] | None = None,
    profile: str = "dr_train",
) -> dict:
    """Run the named graphs over ``seeds`` and return the report dict.

    The report carries every rate the A8 gate checks plus the failure-cause
    histogram the taxonomy consumes; ``violations`` is empty when every gated
    threshold is met.
    """
    seeds = list(seeds)
    names = list(graphs) if graphs is not None else list(CANONICAL_GRAPHS)
    report: dict = {
        "dr_profile": profile,
        "seeds": seeds,
        "graphs": {},
        "violations": [],
    }
    for name in names:
        logs: list[EpisodeLog] = []
        for seed in seeds:
            log = run_episode(name, seed, profile)
            logs.append(log)
            logger.info(
                "%s seed=%d -> %s (%d steps, %d frames)",
                name, seed, "ok" if log.success else "FAILED",
                len(log.steps), len(log.frames),
            )
        summary = _summarize(name, logs)
        for key, minimum in _thresholds_for(name).items():
            if summary[key] < minimum:
                report["violations"].append({
                    "graph": name,
                    "metric": key,
                    "value": summary[key],
                    "minimum": minimum,
                })
        report["graphs"][name] = summary
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry: evaluate the canonical graphs and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-19", help="seed range, e.g. 0-19 or 0-4,9")
    parser.add_argument(
        "--graphs", default="canonical",
        help="comma list of graph names, or 'all'",
    )
    parser.add_argument("--profile", default="dr_train", help="DR profile name")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    names = None if args.graphs == "all" else [n.strip() for n in args.graphs.split(",")]
    report = evaluate(parse_seeds(args.seeds), names, args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, summary in report["graphs"].items():
        print(
            f"{name}: {summary['success']}/{summary['episodes']} "
            f"({summary['success_rate']:.0%} with retries, "
            f"{summary['first_attempt_rate']:.0%} first attempt)"
        )
        for cause, count in summary["failure_causes"].items():
            print(f"    {count:3d}  {cause}")
    print(f"report written to {args.out}")
    for violation in report["violations"]:
        print(
            f"THRESHOLD VIOLATED  {violation['graph']}.{violation['metric']} "
            f"= {violation['value']:.3f} < {violation['minimum']:.3f}"
        )
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
