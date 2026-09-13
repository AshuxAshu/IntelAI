"""Forward kinematics and Jacobian utilities for dual SO-101 robotic arms."""

from __future__ import annotations

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError

ARM_HINGE_SUFFIXES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


class KinematicsError(DinnerTableError):
    """Exception raised for kinematics queries and configuration errors."""


def _resolve_arm(identifier: str) -> str:
    """Resolve arm prefix 'A' or 'B' from an arm name or site name."""
    if identifier.startswith("A"):
        return "A"
    if identifier.startswith("B"):
        return "B"
    raise KinematicsError(
        f"Cannot resolve arm from identifier: {identifier}. Must start with 'A' or 'B'."
    )


def _arm_joint_names(arm: str) -> tuple[str, ...]:
    """Return the ordered canonical names for the arm five hinge joints."""
    prefix = _resolve_arm(arm)
    return tuple(f"{prefix}.{suffix}" for suffix in ARM_HINGE_SUFFIXES)


def _arm_dof_indices(model: mujoco.MjModel, arm: str) -> np.ndarray:
    """Return DOF velocity addresses for the arm five hinge joints."""
    names = _arm_joint_names(arm)
    dof_list: list[int] = []
    for name in names:
        j_id = model.joint(name).id
        dof_list.append(int(model.jnt_dofadr[j_id]))
    return np.array(dof_list, dtype=np.int32)


def _arm_qpos_indices(model: mujoco.MjModel, arm: str) -> np.ndarray:
    """Return generalized coordinate addresses for the arm five hinge joints."""
    names = _arm_joint_names(arm)
    qpos_list: list[int] = []
    for name in names:
        j_id = model.joint(name).id
        qpos_list.append(int(model.jnt_qposadr[j_id]))
    return np.array(qpos_list, dtype=np.int32)


def site_pose(data: mujoco.MjData, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return world frame position and rotation matrix for a site."""
    try:
        site_view = data.site(site)
        pos = np.array(site_view.xpos, dtype=np.float64, copy=True)
        rot_mat = np.array(site_view.xmat, dtype=np.float64, copy=True).reshape(3, 3)
        return pos, rot_mat
    except Exception as exc:
        raise KinematicsError(f"Failed to query site pose for '{site}': {exc}") from exc


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> np.ndarray:
    """Return 6x5 Jacobian matrix restricted to the arm five hinge joints."""
    try:
        site_id = model.site(site).id
    except Exception as exc:
        raise KinematicsError(f"Unknown site '{site}': {exc}") from exc

    arm = _resolve_arm(site)
    dof_indices = _arm_dof_indices(model, arm)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    full_j = np.vstack([jacp, jacr])
    return full_j[:, dof_indices].copy()


def joint_limits(model: mujoco.MjModel, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper joint limits of shape (5,) for the arm hinge joints."""
    names = _arm_joint_names(arm)
    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        j_id = model.joint(name).id
        lower.append(float(model.jnt_range[j_id, 0]))
        upper.append(float(model.jnt_range[j_id, 1]))
    return np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)


def set_arm_q(data: mujoco.MjData, arm: str, q_arm: np.ndarray) -> None:
    """Write the arm five hinge joint positions into data.qpos leaving gripper untouched."""
    if len(q_arm) != 5:
        raise KinematicsError(f"Expected 5 joint values for arm '{arm}', got {len(q_arm)}.")
    names = _arm_joint_names(arm)
    for idx, name in enumerate(names):
        data.joint(name).qpos[:] = float(q_arm[idx])
