"""Servo compensation loop canceling gravity sag between IK targets and settled poses."""

from __future__ import annotations

import mujoco
import numpy as np

from dinner_table.scene.builder import Scene
from dinner_table.teacher.ik import solve_ik
from dinner_table.teacher.kinematics import set_arm_q, site_jacobian, site_pose

SAG_TOL_M = 0.002
SAG_SETTLE_STEPS = 250
SAG_MAX_ITERATIONS = 5
SAG_MAX_DQ = 0.15
SAG_DAMPING = 0.02
SAG_RAMP_TICKS = 25
SAG_TICK_STEPS = 20

_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def _arm_joint_ids(model: mujoco.MjModel, arm: str) -> list[int]:
    """Return joint ids for one arm's five hinge joints in canonical order."""
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}.{n}") for n in _JOINT_NAMES]


def _position_jacobian(
    model: mujoco.MjModel, data: mujoco.MjData, arm: str, site: str
) -> np.ndarray:
    """Return the 3x5 positional Jacobian of the site for the arm's hinges."""
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    dof_cols = [model.jnt_dofadr[jid] for jid in _arm_joint_ids(model, arm)]
    return jacp[:, dof_cols].copy()


def _joint_limits_for(model: mujoco.MjModel, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper limits for the arm's five hinge joints."""
    jids = _arm_joint_ids(model, arm)
    lo = np.array([model.jnt_range[j, 0] for j in jids], dtype=np.float64)
    hi = np.array([model.jnt_range[j, 1] for j in jids], dtype=np.float64)
    return lo, hi


def converge_ee(
    scene: Scene,
    arm: str,
    target_pos: np.ndarray,
    target_approach: np.ndarray,
    q0: np.ndarray,
    pos_tol: float = SAG_TOL_M,
) -> np.ndarray:
    """Solve IK then iteratively correct commanded joints so the settled tool hits the target.

    The position servos droop under gravity, so commanding the raw IK solution
    lands the tool several millimeters low. Each iteration commands the current
    solution, lets physics settle, measures the Cartesian residual, and applies
    a small damped-least-squares joint correction through the site Jacobian.
    Corrections are bounded and applied to the commanded targets only, so the
    posture can never jump to a distant IK branch. The loop is closed-loop, so
    it stays correct under domain-randomized mass and friction scaling.
    """
    model = scene.model
    data = scene.data
    site_name = f"{arm}.ee"
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    act_ids = {
        n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}.{n}")
        for n in _JOINT_NAMES
    }
    lo, hi = _joint_limits_for(model, arm)
    target = np.array(target_pos, dtype=np.float64)

    command = solve_ik(model, data, site_name, target, target_approach, q0=q0)
    command = np.clip(command, lo, hi)

    # Interpolate from the live posture to the IK solution at control rate
    # first: commanding the full posture in one jump whips the arm through a
    # large transient and it can fall into a torque-saturated drooped posture
    # that the position servos cannot climb out of.
    jids = _arm_joint_ids(model, arm)
    start = np.array([float(data.qpos[model.jnt_qposadr[j]]) for j in jids], dtype=np.float64)
    for step in range(1, SAG_RAMP_TICKS + 1):
        alpha = float(step) / float(SAG_RAMP_TICKS)
        ramp = start + alpha * (command - start)
        for i, n in enumerate(_JOINT_NAMES):
            data.ctrl[act_ids[n]] = ramp[i]
        for _ in range(SAG_TICK_STEPS):
            mujoco.mj_step(model, data)

    for _ in range(SAG_MAX_ITERATIONS):
        for i, n in enumerate(_JOINT_NAMES):
            data.ctrl[act_ids[n]] = command[i]
        for _ in range(SAG_SETTLE_STEPS):
            mujoco.mj_step(model, data)
        actual = np.array(data.site_xpos[ee_id], dtype=np.float64)
        residual = target - actual
        if float(np.linalg.norm(residual)) <= pos_tol:
            return command

        # The Jacobian must be evaluated at the settled state, with the site
        # frame reflecting the live arm posture.
        mujoco.mj_forward(model, data)
        jac = _position_jacobian(model, data, arm, site_name)
        jjt = jac @ jac.T + (SAG_DAMPING**2) * np.eye(3)
        dq = jac.T @ np.linalg.solve(jjt, residual)
        max_dq = float(np.max(np.abs(dq)))
        if max_dq > SAG_MAX_DQ:
            dq = dq * (SAG_MAX_DQ / max_dq)
        command = np.clip(command + dq, lo, hi)
    return command


def settled_ee_error(
    scene: Scene,
    arm: str,
    target_pos: np.ndarray,
) -> float:
    """Measure the distance between the settled tool and a target position."""
    ee_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}.ee")
    actual = np.array(scene.data.site_xpos[ee_id], dtype=np.float64)
    return float(np.linalg.norm(actual - np.array(target_pos, dtype=np.float64)))
