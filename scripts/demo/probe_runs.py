"""Probe which demonstration runs succeed, per seed.

Writes JSON: one record per (run, seed) with per-step outcomes, so video
composition and the accompanying report come from measurement rather than
assumption.

  uv run python scripts/demo/probe_runs.py --seeds 0-3 --out artifacts/demo/probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runs as run_registry  # noqa: E402
from engine import execute_run  # noqa: E402


def parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=sorted(run_registry.RUNS))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--profile", default="dr_train")
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo/probe.json"))
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for run_name in args.runs:
        steps = run_registry.RUNS.get(run_name)
        if steps is None:
            raise SystemExit(f"unknown run: {run_name}")
        for seed in seeds:
            t0 = time.perf_counter()
            result, _scene, _ctx = execute_run(run_name, steps, seed, args.profile)
            rec = result.to_dict()
            rec["wall_s"] = round(time.perf_counter() - t0, 1)
            records.append(rec)
            detail = " ".join(
                f"{s['skill']}{'/' + s['object'] if s['object'] else ''}="
                f"{s['outcome']}" + (f"({s['phase']}/{s['cause']})" if s["cause"] else "")
                for s in rec["steps"]
            )
            print(f"{'OK ' if rec['success'] else 'FAIL'} {run_name:14s} seed={seed:2d} "
                  f"({rec['wall_s']:5.1f}s) {detail}", flush=True)
            args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    ok = sum(1 for r in records if r["success"])
    print(f"\n{ok}/{len(records)} runs succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
