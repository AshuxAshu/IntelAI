"""Forward-kinematics helpers for the privileged teacher controller.

All functions operate on the compiled MuJoCo model/data produced by
``dinner_table.scene.builder.Scene``. Arm joints are addressed by the
canonical names from the frozen geometry contract; the gripper joint is
deliberately excluded everywhere — the teacher treats gripper aperture
separately from arm pose.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINT_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
N_ARM_JOINTS = len(ARM_JOINT_SUFFIXES)


def arm_of(site: str) -> str:
    """Return the arm prefix ("A" or "B") owning the given site name."""
    arm = site.split(".", 1)[0]
    if arm not in ("A", "B"):
        raise ValueError(f"site name does not carry an arm prefix: {site}")
    return arm


def joint_id(model, arm: str, suffix: str) -> int:
    """Resolve a named arm joint to its MuJoCo joint id."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}.{suffix}")
    if jid == -1:
        raise ValueError(f"unknown arm joint: {arm}.{suffix}")
    return jid


def site_pose(data, site: str) -> tuple[np.ndarray, np.ndarray]:
    """World (3,) position and (3,3) rotation of a MuJoCo site."""
    model = data.model
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid == -1:
        raise ValueError(f"unknown site: {site}")
    pos = np.array(data.site_xpos[sid], dtype=np.float64)
    rot = np.array(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    return pos, rot


def site_jacobian(model, data, site: str) -> np.ndarray:
    """(6, n_arm_joints) Jacobian via mujoco.mj_jacSite, columns restricted to the arm's 5 joints."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid == -1:
        raise ValueError(f"unknown site: {site}")
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)
    arm = arm_of(site)
    dof_cols = [model.jnt_dofadr[joint_id(model, arm, s)] for s in ARM_JOINT_SUFFIXES]
    return np.vstack([jacp[:, dof_cols], jacr[:, dof_cols]])


def joint_limits(model, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """(5,) lower/upper for the arm's 5 hinge joints, from the model."""
    lower = np.zeros(N_ARM_JOINTS, dtype=np.float64)
    upper = np.zeros(N_ARM_JOINTS, dtype=np.float64)
    for i, suffix in enumerate(ARM_JOINT_SUFFIXES):
        rng = model.jnt_range[joint_id(model, arm, suffix)]
        lower[i], upper[i] = float(rng[0]), float(rng[1])
    return lower, upper


def set_arm_q(data, arm: str, q_arm: np.ndarray) -> None:
    """Write the arm's 5 joint positions into data.qpos (grippers untouched)."""
    q_arm = np.asarray(q_arm, dtype=np.float64)
    if q_arm.shape != (N_ARM_JOINTS,):
        raise ValueError(f"q_arm must have shape ({N_ARM_JOINTS},), got {q_arm.shape}")
    model = data.model
    for i, suffix in enumerate(ARM_JOINT_SUFFIXES):
        data.qpos[model.jnt_qposadr[joint_id(model, arm, suffix)]] = q_arm[i]


def arm_q(data, arm: str) -> np.ndarray:
    """Read the arm's current 5 joint positions (gripper excluded)."""
    model = data.model
    q = np.zeros(N_ARM_JOINTS, dtype=np.float64)
    for i, suffix in enumerate(ARM_JOINT_SUFFIXES):
        q[i] = float(data.qpos[model.jnt_qposadr[joint_id(model, arm, suffix)]])
    return q
