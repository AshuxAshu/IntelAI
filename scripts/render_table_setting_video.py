"""Render multi-camera verification video for table-setting skills."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from dinner_table.contracts.geometry import CONTROL_HZ
from dinner_table.scene.builder import Scene
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.skills import (
    CloseDrawer,
    OpenDrawer,
    Pick,
    Place,
)

PANEL_W, PANEL_H = 640, 480
GRID_W, GRID_H = PANEL_W * 2, PANEL_H * 2
FPS = CONTROL_HZ
PANELS = (("demo_cam", "overhead"), ("wrist_A", "wrist_B"))


def caption_frame(img: Image.Image, lines: list[str]) -> None:
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = 10 + i * 22
        draw.rectangle([6, y - 2, 10 + 9 * len(line), y + 18], fill=(10, 10, 10))
        draw.text((10, y), line, fill=(240, 240, 240))


class VideoRecorder:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "22", "-preset", "veryfast",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.write(frame.tobytes())
            self.frames += 1

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait()


def render_full_setting(seed: int, out_path: Path, dr_profile: str = "dr_train") -> dict:
    t0 = time.perf_counter()
    scene = Scene(seed=seed, dr_profile=dr_profile)
    scene.hold_safe()
    model, data = scene.model, scene.data

    renderers = {cam: mujoco.Renderer(model, height=PANEL_H, width=PANEL_W) for cam in (PANELS[0] + PANELS[1])}
    grid = Image.new("RGB", (GRID_W, GRID_H))
    rec = VideoRecorder(out_path, GRID_W, GRID_H, FPS)

    ctx = TeacherContext(scene)
    ctx.begin(300.0)

    def capture(active_skill: str, phase: str) -> None:
        for row, cams in enumerate(PANELS):
            for col, cam in enumerate(cams):
                r = renderers[cam]
                r.update_scene(data, camera=cam)
                panel = Image.fromarray(r.render())
                grid.paste(panel, (col * PANEL_W, row * PANEL_H))
        lines = [
            f"Table Setting | Profile: {dr_profile} | Seed: {seed}",
            f"Skill: {active_skill} | Phase: {phase} | Sim Time: {data.time:5.2f}s",
        ]
        caption_frame(grid, lines)
        rec.write(np.asarray(grid))

    steps_plan = [
        ("open_drawer", OpenDrawer("A")),
        ("pick fork_1", Pick("A", "fork_1")),
        ("place fork_1", Place("A", "fork_1", "fork_setting")),
        ("pick spoon_1", Pick("A", "spoon_1")),
        ("place spoon_1", Place("A", "spoon_1", "spoon_setting")),
        ("close_drawer", CloseDrawer("A")),
        ("pick plate", Pick("A", "plate")),
        ("place plate", Place("A", "plate", "placemat_1")),
        ("pick mug", Pick("B", "mug")),
        ("place mug", Place("B", "mug", "placemat_2")),
    ]

    status = "ok"
    executed = []
    try:
        for name, skill in steps_plan:
            last_action = None
            for action in skill.run(ctx):
                ctx.step(action)
                last_action = action
                capture(name, getattr(skill, "phase", "running"))
            if last_action is not None:
                for _ in range(12):
                    ctx.step(last_action)
                    capture(name, "completed")
            executed.append(name)
    except SkillFailed as exc:
        status = f"failed at {exc.skill}/{exc.phase}: {exc.cause}"
        for _ in range(25):
            capture(exc.skill, f"FAILED: {exc.cause[:30]}")
    except Exception as exc:
        status = f"error: {exc}"
        for _ in range(25):
            capture("error", f"ERROR: {str(exc)[:30]}")
    finally:
        rec.close()
        for r in renderers.values():
            r.close()

    wall_s = round(time.perf_counter() - t0, 1)
    return {
        "seed": seed,
        "profile": dr_profile,
        "status": status,
        "steps_executed": len(executed),
        "total_steps": len(steps_plan),
        "frames": rec.frames,
        "wall_s": wall_s,
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render table setting verification video")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", type=str, default="dr_train")
    parser.add_argument("--out", type=Path, default=Path("artifacts/videos/table_setting_seed0.mp4"))
    args = parser.parse_args()

    print(f"Rendering table setting video (seed {args.seed}, profile {args.profile})...")
    res = render_full_setting(args.seed, args.out, args.profile)
    print(f"Result: {res}")


if __name__ == "__main__":
    main()
