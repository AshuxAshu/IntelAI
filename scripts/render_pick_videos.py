"""Render multi-camera MP4s of teacher Pick episodes across seeds and DR profiles.

One MP4 per (profile, object, seed): a 2x2 grid of demo_cam | overhead |
wrist_A | wrist_B with a burned-in caption (object, profile, seed, skill
phase, sim time, outcome). Episodes that fail are still rendered and marked
FAILED — the videos exist to show what actually happens.

Resumable: existing non-empty MP4s are skipped unless --force. A manifest at
<out>/manifest.json records every episode's outcome.

Usage:
  uv run python scripts/render_pick_videos.py                          # everything
  uv run python scripts/render_pick_videos.py --seeds 0-4 --objects plate mug
  uv run python scripts/render_pick_videos.py --profiles dr_train --seeds 0-9
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from dinner_table.scene.builder import Scene
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.skills import Pick

OUT_DIR = Path("artifacts/videos")
PANEL_W, PANEL_H = 640, 480          # per-camera render size (native framebuffer)
GRID_W, GRID_H = PANEL_W * 2, PANEL_H * 2
FPS = 25
DRAWER_STEPS = 1500                    # 3 s of actuator opening at 500 Hz
TRAIL_TICKS = 40                       # 1.6 s hold after the skill ends
UTENSILS = ("fork_1", "fork_2", "spoon_1", "spoon_2")
PANELS = (("demo_cam", "overhead"), ("wrist_A", "wrist_B"))


def parse_seed_spec(spec: str) -> list[int]:
    """Parse '3' or '0-9' into a list of seed ints."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


class EpisodeRecorder:
    """Pipes raw RGB frames into an ffmpeg encode."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait()


def _caption(img: Image.Image, lines: list[str]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = 6 + i * 20
        draw.rectangle([4, y - 2, 10 + 8.5 * len(line), y + 17], fill=(0, 0, 0))
        draw.text((8, y), line, fill=(255, 255, 255))
    return img


def render_episode(
    object_name: str,
    seed: int,
    profile: str,
    out_path: Path,
    arm: str,
) -> dict:
    """Run one Pick episode and record the four-camera grid; returns the record."""
    t_start = time.perf_counter()
    scene = Scene(seed=seed, dr_profile=profile)
    model, data = scene.model, scene.data

    renderers = {
        cam: mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
        for cam in ( PANELS[0] + PANELS[1] )
    }
    grid = Image.new("RGB", (GRID_W, GRID_H))
    rec = EpisodeRecorder(out_path, GRID_W, GRID_H, FPS)

    status = "ok"
    cause = phase = ""
    sim_limit = 120.0

    def capture(lines: list[str]) -> None:
        for row, cams in enumerate(PANELS):
            for col, cam in enumerate(cams):
                r = renderers[cam]
                r.update_scene(data, camera=cam)
                panel = Image.fromarray(r.render())
                grid.paste(panel, (col * PANEL_W, row * PANEL_H))
        _caption(grid, lines)
        rec.write(np.asarray(grid))

    try:
        # Drawer opening phase for cutlery (rendered for context).
        if object_name in UTENSILS:
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_actuator")
            data.ctrl[act] = 0.12
            for step in range(DRAWER_STEPS):
                mujoco.mj_step(model, data)
                if step % 20 == 0:
                    capture([
                        f"{object_name} | {profile} | seed {seed}",
                        f"opening drawer | t={data.time:5.1f}s",
                    ])

        ctx = TeacherContext(scene)
        ctx.begin(sim_limit)
        skill = Pick(arm, object_name)
        for action in skill.run(ctx):
            ctx.step(action)
            capture([
                f"{object_name} | {profile} | seed {seed} | arm {arm}",
                f"phase: {skill.phase:8s} | t={data.time:5.1f}s",
            ])
        # Victory hold so the lift is visible before the cut; the carry audit
        # keeps verifying the grasp here, a genuine hold check.
        for _ in range(TRAIL_TICKS):
            ctx.step(action)
            capture([
                f"{object_name} | {profile} | seed {seed} | arm {arm}",
                f"phase: DONE     | t={data.time:5.1f}s",
            ])
    except SkillFailed as exc:
        status = "failed"
        phase, cause = exc.phase, exc.cause
        for _ in range(TRAIL_TICKS):
            mujoco.mj_step(model, data)
            capture([
                f"{object_name} | {profile} | seed {seed} | arm {arm}",
                f"FAILED at {phase}: {cause[:38]} | t={data.time:5.1f}s",
            ])

    rec.close()
    for r in renderers.values():
        r.close()
    return {
        "object": object_name, "seed": seed, "profile": profile, "arm": arm,
        "status": status, "phase": phase, "cause": cause,
        "frames": rec.frames, "sim_time_s": round(float(data.time), 1),
        "wall_s": round(time.perf_counter() - t_start, 1),
        "file": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+",
                        default=["plate", "mug", "fork_1", "fork_2", "spoon_1", "spoon_2"])
    parser.add_argument("--profiles", nargs="+",
                        default=["default", "dr_train", "eval_extreme"])
    parser.add_argument("--seeds", nargs="+", type=str, default=["0-9"],
                        help="seed numbers or ranges, e.g. 3 or 0-9")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest path (defaults to <out>/manifest.json); "
                             "give each parallel worker its own to avoid clobbering")
    parser.add_argument("--force", action="store_true",
                        help="re-render even if the MP4 exists")
    args = parser.parse_args()

    seeds: list[int] = []
    for spec in args.seeds:
        seeds.extend(parse_seed_spec(spec))

    ARMS = {"plate": "A", "mug": "B",
            "fork_1": "A", "fork_2": "A", "spoon_1": "A", "spoon_2": "A"}

    manifest_path = args.manifest or (args.out / "manifest.json")
    manifest = {"episodes": []}
    if manifest_path.is_file() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    total = len(args.profiles) * len(args.objects) * len(seeds)
    done = 0
    for profile in args.profiles:
        for object_name in args.objects:
            arm = ARMS.get(object_name)
            if arm is None:
                raise SystemExit(f"no arm mapping for object {object_name}")
            for seed in seeds:
                done += 1
                out_path = args.out / profile / f"{object_name}_seed{seed:02d}.mp4"
                if out_path.is_file() and out_path.stat().st_size > 0 and not args.force:
                    print(f"[{done}/{total}] skip (exists) {out_path}")
                    continue
                record = render_episode(object_name, seed, profile, out_path, arm)
                manifest["episodes"].append(record)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"[{done}/{total}] {record['status']:7s} {out_path} "
                      f"({record['frames']} frames, {record['wall_s']}s wall)")
    failures = [e for e in manifest["episodes"] if e["status"] != "ok"]
    print(f"\nDone: {len(manifest['episodes'])} episodes rendered, "
          f"{len(failures)} failed pickups (marked FAILED in filenames/manifest).")


if __name__ == "__main__":
    main()
