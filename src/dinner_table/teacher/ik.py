"""Weighted damped least-squares inverse kinematics solver for SO-101 robotic arms."""

from __future__ import annotations

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import ARM_MOUNTS, HOME_JOINTS
from dinner_table.teacher.kinematics import (
    joint_limits,
    set_arm_q,
    site_jacobian,
    site_pose,
)

MAX_ARM_REACH_M = 0.60
SHOULDER_Z_OFFSET_M = 0.11


class IKUnreachable(DinnerTableError):
    """Exception raised when an inverse kinematics target cannot be reached within tolerances."""

    def __init__(self, site: str, target_pos: np.ndarray) -> None:
        self.site = site
        self.target_pos = np.array(target_pos, dtype=np.float64, copy=True)
        super().__init__(
            f"Target position {self.target_pos.tolist()} is unreachable for site '{self.site}'."
        )


def _resolve_arm(identifier: str) -> str:
    """Resolve arm prefix 'A' or 'B' from site name or arm identifier."""
    if identifier.startswith("A"):
        return "A"
    if identifier.startswith("B"):
        return "B"
    raise IKUnreachable(identifier, np.zeros(3, dtype=np.float64))


def _shoulder_pivot_pos(arm: str) -> np.ndarray:
    """Compute world position of the shoulder lift pivot for an arm."""
    mount_xyz = np.array(ARM_MOUNTS[arm], dtype=np.float64)
    mount_xyz[2] = mount_xyz[2] + SHOULDER_Z_OFFSET_M
    return mount_xyz


def _get_basin_seeds(arm: str, q0: np.ndarray) -> list[np.ndarray]:
    """Generate ordered list of diverse kinematic seed configurations for basin hopping."""
    seeds: list[np.ndarray] = [np.array(q0, dtype=np.float64, copy=True)]
    home_q = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
    seeds.append(home_q)
    # Folded elbow-down postures that reach tabletop height with a downward approach.
    seeds.append(np.array([1.70, -0.21, -1.73, -1.24, -2.90], dtype=np.float64))
    seeds.append(np.array([-1.70, -0.21, -1.73, -1.24, 2.90], dtype=np.float64))
    seeds.append(np.array([0.60, -1.04, -0.80, -1.31, 2.20], dtype=np.float64))
    seeds.append(np.array([-0.60, -1.04, -0.80, -1.31, -2.20], dtype=np.float64))
    seeds.append(np.array([0.49, 0.53, 1.57, 1.00, -1.78], dtype=np.float64))
    seeds.append(np.array([-0.49, 0.53, 1.57, 1.00, 1.78], dtype=np.float64))
    seeds.append(np.array([0.0, -1.30, 0.90, -1.20, 0.0], dtype=np.float64))
    seeds.append(np.array([-1.57, 0.4, 0.8, -0.4, 0.0], dtype=np.float64))
    seeds.append(np.array([1.57, 0.4, 0.8, -0.4, 0.0], dtype=np.float64))
    return seeds


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site: str,
    target_pos: np.ndarray,
    target_approach: np.ndarray,
    q0: np.ndarray,
    pos_tol: float = 2e-3,
    ang_tol: float = 0.05235987755982988,
    iters: int = 100,
    damping: float = 0.05,
) -> np.ndarray:
    """Solve inverse kinematics using weighted damped least squares with multi-seed basin hopping."""
    arm = _resolve_arm(site)
    t_pos = np.array(target_pos, dtype=np.float64)
    t_app = np.array(target_approach, dtype=np.float64)
    app_norm = float(np.linalg.norm(t_app))
    if app_norm > 1e-9:
        t_app = t_app / app_norm
    else:
        t_app = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    pivot = _shoulder_pivot_pos(arm)
    dist_from_pivot = float(np.linalg.norm(t_pos - pivot))
    if dist_from_pivot > MAX_ARM_REACH_M:
        raise IKUnreachable(site, t_pos)

    q_min, q_max = joint_limits(model, arm)
    weight_diag = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3], dtype=np.float64)
    w_mat = np.diag(weight_diag)
    lambda_sq = damping**2
    eye6 = np.eye(6, dtype=np.float64)

    seeds = _get_basin_seeds(arm, q0)
    saved_qpos = np.array(data.qpos, dtype=np.float64, copy=True)

    try:
        for seed_idx, seed in enumerate(seeds):
            q = np.clip(seed, q_min, q_max)
            if seed_idx == 0:
                steps_for_seed = iters
            else:
                steps_for_seed = 35

            for _ in range(steps_for_seed):
                set_arm_q(data, site, q)
                mujoco.mj_forward(model, data)
                cur_pos, cur_rot = site_pose(data, site)

                pos_err = t_pos - cur_pos
                cur_approach = cur_rot[:, 2]
                ang_err = np.cross(cur_approach, t_app)

                pos_err_norm = float(np.linalg.norm(pos_err))
                ang_err_norm = float(np.linalg.norm(ang_err))

                if pos_err_norm < pos_tol and ang_err_norm < ang_tol:
                    return q

                err = np.concatenate([pos_err, ang_err])
                j_mat = site_jacobian(model, data, site)

                wj = w_mat @ j_mat
                a_mat = wj @ wj.T + lambda_sq * eye6
                dq = wj.T @ np.linalg.solve(a_mat, w_mat @ err)

                max_step = float(np.max(np.abs(dq)))
                if max_step > 0.2:
                    dq = dq * (0.2 / max_step)

                q = np.clip(q + dq, q_min, q_max)
    finally:
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)

    raise IKUnreachable(site, t_pos)


def ik_above(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: str,
    grasp_pose: tuple[np.ndarray, np.ndarray] | np.ndarray | object,
    height: float,
) -> np.ndarray:
    """Solve inverse kinematics for an approach waypoint offset along world positive Z."""
    clean_arm = _resolve_arm(arm)
    if isinstance(grasp_pose, (tuple, list)):
        base_pos = np.array(grasp_pose[0], dtype=np.float64)
    elif hasattr(grasp_pose, "position"):
        base_pos = np.array(grasp_pose.position, dtype=np.float64)
    else:
        base_pos = np.array(grasp_pose, dtype=np.float64)

    target_pos = base_pos + np.array([0.0, 0.0, height], dtype=np.float64)
    target_approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    q0 = np.array(HOME_JOINTS[clean_arm][:5], dtype=np.float64)
    site_name = f"{clean_arm}.ee"

    return solve_ik(
        model=model,
        data=data,
        site=site_name,
        target_pos=target_pos,
        target_approach=target_approach,
        q0=q0,
    )
