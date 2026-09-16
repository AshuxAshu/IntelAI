"""Debug video renderer for failing teacher-skill scenarios.

Renders one MP4 per scenario: side-by-side (3/4 overview | gripper-tracking
closeup) with a burned-in caption of the skill phase, jaw forces, and object
lift. Use these to watch what the gripper actually does during a failing
grasp/close/lift.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from dinner_table.contracts.geometry import HOME_JOINTS
from dinner_table.scene.builder import Scene
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.grasp_catalog import GraspCatalog
from dinner_table.teacher.kinematics import arm_q, site_pose
from dinner_table.teacher.skills import Pick

OUT_DIR = Path("artifacts/debug_videos")
W, H = 640, 480
FPS = 30


def _caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = 8 + i * 18
        draw.rectangle([4, y - 2, 8 + 8.5 * len(line), y + 16], fill=(0, 0, 0))
        draw.text((8, y), line, fill=(255, 255, 255))
    return np.asarray(img)


def render_scenario(
    label: str,
    arm: str,
    object_name: str,
    seed: int,
    open_drawer: bool = False,
    dr_profile: str = "dr_train",
    bypass_path_check: bool = False,
) -> None:
    scene = Scene(seed=seed, dr_profile=dr_profile)
    model, data = scene.model, scene.data
    if open_drawer:
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_actuator")
        data.ctrl[act] = 0.12
        for _ in range(1500):
            mujoco.mj_step(model, data)

    overview = mujoco.Renderer(model, height=H, width=W)
    closeup = mujoco.Renderer(model, height=H, width=W)
    ov_cam = mujoco.MjvCamera()
    ov_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ov_cam.lookat[:] = [0.0, -0.15, 0.45]
    ov_cam.distance = 1.35
    ov_cam.azimuth = -135
    ov_cam.elevation = -25
    cl_cam = mujoco.MjvCamera()
    cl_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cl_cam.distance = 0.40
    cl_cam.elevation = -20
    cl_cam.azimuth = -135

    ctx = TeacherContext(scene)
    ctx.begin(90.0)
    if bypass_path_check:
        # Debug only: execute the planned paths without the clearance check so
        # the video shows the collision the check would have rejected.
        ctx.check_path = lambda arm_, points: None
    catalog = GraspCatalog()
    skill = Pick(arm, object_name)
    outcome = "ok"
    frames: list[np.ndarray] = []
    z0 = float(scene.object_pose(object_name)[0][2])
    try:
        for action in skill.run(ctx):
            ctx.step(action)
            fixed, moving = ctx.finger_forces(arm, object_name)
            z = float(scene.object_pose(object_name)[0][2])
            site, _ = site_pose(data, f"{arm}.ee")

            overview.update_scene(data, camera=ov_cam)
            ov = overview.render()
            cl_cam.lookat[:] = site
            closeup.update_scene(data, camera=cl_cam)
            cl = closeup.render()
            combined = np.concatenate([ov, cl], axis=1)
            caption = [
                f"{label}: seed {seed}  pick {arm}/{object_name}  phase={skill.phase}",
                f"jaws fixed/moving {fixed:5.2f}/{moving:5.2f} N   object lift {z - z0:+.3f} m   t={data.time:5.1f}s",
            ]
            frames.append(_caption(combined, caption))
            if len(frames) > 2100:
                raise SkillFailed("pick", "timeout", "render cap")
    except SkillFailed as exc:
        outcome = f"{exc.phase}/{exc.cause}"

    for renderer in (overview, closeup):
        renderer.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}.mp4"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{2 * W}x{H}", "-r", str(FPS), "-i", "-",
            "-pix_fmt", "yuv420p", "-crf", "23", str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        ffmpeg.stdin.write(f.astype(np.uint8).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"{label}: {outcome:40s} -> {out_path} ({len(frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render failing teacher-skill scenarios.")
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()
    all_scenarios = [
        ("plate_seed4", "A", "plate", 4, False, False),
        ("plate_seed2", "A", "plate", 2, False, False),
        ("plate_seed9", "A", "plate", 9, False, False),
        ("mug_seed0", "B", "mug", 0, False, False),
        ("mug_seed5", "B", "mug", 5, False, False),
        ("fork_seed0_pathcheck_bypassed", "A", "fork_1", 0, True, True),
        ("fork_seed1_pathcheck_bypassed", "A", "fork_1", 1, True, True),
    ]
    wanted = args.scenarios
    for label, arm, obj, seed, drawer, bypass in all_scenarios:
        if wanted and label not in wanted:
            continue
        render_scenario(label, arm, obj, seed, open_drawer=drawer,
                        bypass_path_check=bypass)


if __name__ == "__main__":
    main()
