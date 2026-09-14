"""CLI debug renderer for planned grasps, corridors, and object targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from dinner_table.contracts.geometry import HOME_JOINTS
from dinner_table.scene.builder import Scene
from dinner_table.scene.contact_sheet import _write_png
from dinner_table.teacher.ik import IKUnreachable, solve_ik
from dinner_table.teacher.kinematics import set_arm_q, site_pose
from dinner_table.teacher.planner import CorridorBlocked, plan_corridor


def _fk_along(scene: Scene, arm: str, waypoints: list[np.ndarray]) -> list[np.ndarray]:
    """Compute ee positions along the interpolated corridor path."""
    pts: list[np.ndarray] = []
    for wp_idx in range(len(waypoints)):
        q_a = waypoints[wp_idx]
        pts.append(_fk(scene, arm, q_a))
        if wp_idx + 1 < len(waypoints):
            q_b = waypoints[wp_idx + 1]
            for step in range(1, 10):
                alpha = step / 10.0
                pts.append(_fk(scene, arm, (1.0 - alpha) * q_a + alpha * q_b))
    return pts


def _fk(scene: Scene, arm: str, q: np.ndarray) -> np.ndarray:
    """Forward kinematics ee position for a joint configuration."""
    set_arm_q(scene.data, arm, q)
    mujoco.mj_forward(scene.model, scene.data)
    pos, _ = site_pose(scene.data, f"{arm}.ee")
    return pos


def render_debug(seed: int, arm: str, target: tuple[float, float, float], out_path: Path) -> Path:
    """Render the planned corridor, ee positions, and target from two view angles."""
    scene = Scene(seed=seed, dr_profile="default")
    q0 = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
    target_pos = np.array(target, dtype=np.float64)

    status: list[str] = []
    try:
        q_goal = solve_ik(
            scene.model,
            scene.data,
            f"{arm}.ee",
            target_pos,
            np.array([0.0, 0.0, -1.0]),
            q0=q0,
        )
        status.append("ik: solved")
    except IKUnreachable:
        scene.close()
        raise SystemExit(f"target {target} unreachable for arm {arm}")
    try:
        waypoints = plan_corridor(scene, arm, q0, q_goal)
        status.append(f"corridor: {len(waypoints)} waypoints")
    except CorridorBlocked as exc:
        status.append(f"corridor: blocked ({exc})")
        waypoints = [q0, q_goal]

    path_pts = _fk_along(scene, arm, waypoints)

    frames = []
    for cam in ("overhead", "demo_cam"):
        img = scene.render(cam).copy()
        frames.append(img)
    # overlay the target and path via a small site marker is not possible post-render;
    # instead annotate with pixel projection through camera intrinsics
    for img, cam in zip(frames, ("overhead", "demo_cam"), strict=False):
        _project_annotations(img, scene, cam, target_pos, path_pts, arm)
    grid = np.concatenate([frames[0], frames[1]], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(grid, out_path)
    scene.close()
    for line in status:
        print(line)
    return out_path


def _project_annotations(
    img: np.ndarray,
    scene: Scene,
    cam: str,
    target_pos: np.ndarray,
    path_pts: list[np.ndarray],
    arm: str,
) -> None:
    """Project the target point and corridor path onto the image as pixels."""
    cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    cam_pos = np.array(scene.data.cam_xpos[cam_id])
    cam_mat = np.array(scene.data.cam_xmat[cam_id]).reshape(3, 3)
    k_mat = scene.camera_intrinsics(cam)
    h, w, _ = img.shape

    def project(p: np.ndarray) -> tuple[int, int]:
        p_cam = cam_mat.T @ (p - cam_pos)
        u = int(k_mat[0, 0] * (p_cam[0] / -p_cam[2]) + k_mat[0, 2])
        v = int(k_mat[1, 1] * (-p_cam[1] / -p_cam[2]) + k_mat[1, 2])
        return u, v

    for idx, pt in enumerate(path_pts):
        u, v = project(pt)
        if 0 <= u < w and 0 <= v < h:
            color = (0, 255, 0)
            if idx == 0 or idx == len(path_pts) - 1:
                color = (255, 0, 0)
            img[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3] = color
    u, v = project(target_pos)
    if 0 <= u < w and 0 <= v < h:
        img[max(0, v - 4) : v + 5, max(0, u - 4) : u + 5] = (255, 255, 0)


def main() -> None:
    """CLI entrypoint for corridor debug rendering."""
    parser = argparse.ArgumentParser(description="Render a planned grasp corridor for debugging.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arm", type=str, default="A", choices=("A", "B"))
    parser.add_argument("--target", type=float, nargs=3, default=[0.22, 0.10, 0.42])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if args.out is not None:
        target_path = Path(args.out)
    else:
        target_path = Path("artifacts") / f"corridor_{args.arm}_{args.seed}.png"
    saved = render_debug(args.seed, args.arm, tuple(args.target), target_path)
    print(f"Debug render saved to: {saved}")


if __name__ == "__main__":
    main()
