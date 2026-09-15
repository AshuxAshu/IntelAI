"""Weighted damped-least-squares IK for the 5-DOF SO-101 arms.

A 5-DOF arm cannot satisfy a full 6-D pose, so position rows are weighted
above orientation rows (the approach vector) and gripper yaw is planned
separately via wrist roll by the skill layer. Failure is explicit —
``IKUnreachable`` — never a garbage solution.
"""

from __future__ import annotations

import mujoco
import numpy as np

from dinner_table.contracts.geometry import HOME_JOINTS
from dinner_table.teacher.kinematics import (
    arm_of,
    arm_q,
    joint_limits,
    set_arm_q,
    site_jacobian,
    site_pose,
)

POS_TOL = 2e-3
ANG_TOL = np.deg2rad(3.0)
MAX_ITERS = 100
DAMPING = 0.05
STEP_CAP = 0.2  # rad per iteration; keeps the linearization valid
FINAL_ORIENT_WEIGHT = 0.3
# Orientation weight ramp: converge position first (3 well-conditioned rows),
# then bring the approach vector in gradually. The 5x5 task system is
# ill-conditioned (wrist roll is nearly null for approach alignment), so a
# cold start at full weights diverges from distant seeds.
ORIENT_WEIGHT_RAMP = (0.0, 0.03, 0.1, FINAL_ORIENT_WEIGHT)
N_RANDOM_RESTARTS = 4
# Structured grasp-pose seeds: the SO-101's approach alignment is nearly null in
# wrist roll, so distinct roll basins need distinct seeds. Pan spread covers the
# extreme lateral reaches; roll -2.4 is the canonical pregrasp wrist of the arm.
STRUCTURED_SEED_PANS = (-1.85, -1.2, 0.0, 1.2, 1.85)
STRUCTURED_SEED_ROLLS = (-2.4, 0.0, 2.4)
TASK_WEIGHTS = np.diag(
    [1.0, 1.0, 1.0, FINAL_ORIENT_WEIGHT, FINAL_ORIENT_WEIGHT, FINAL_ORIENT_WEIGHT]
)


class IKUnreachable(Exception):
    """Raised when no in-limit joint configuration attains the target pose."""

    def __init__(self, site: str, target_pos) -> None:
        self.site = site
        self.target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        super().__init__(f"IK failed for site {site} at target {self.target_pos.tolist()}")


def _errors(data, site: str, target_pos: np.ndarray, target_approach: np.ndarray):
    pos, rot = site_pose(data, site)
    err = np.concatenate([target_pos - pos, np.cross(rot[:, 2], target_approach)])
    return err


def _dls_phase(
    model,
    data,
    site: str,
    arm: str,
    target_pos: np.ndarray,
    target_approach: np.ndarray,
    q: np.ndarray,
    orient_weight: float,
    pos_tol: float,
    ang_tol: float,
    iters: int,
    damping: float,
) -> tuple[np.ndarray, float, float]:
    """Run weighted damped least squares at one orientation weight.

    Returns (q, pos_err, ang_err). The weighted step is
    ``dq = J^T W (W J J^T W^T + damping^2 I)^-1 W e`` — the least-squares
    solution of ``W J dq = W e``.
    """
    lower, upper = joint_limits(model, arm)
    w = np.diag([1.0, 1.0, 1.0, orient_weight, orient_weight, orient_weight])
    eye6 = np.eye(6, dtype=np.float64)
    # The orientation error is a cross product, so its norm is sin(angle);
    # comparing against sin(ang_tol) guarantees the true angle is within tolerance.
    ang_err_tol = np.sin(ang_tol)
    pos_err = ang_err = np.inf
    for _ in range(iters):
        set_arm_q(data, arm, q)
        mujoco.mj_forward(model, data)
        err = _errors(data, site, target_pos, target_approach)
        pos_err = float(np.linalg.norm(err[:3]))
        ang_err = float(np.linalg.norm(err[3:]))
        if pos_err < pos_tol and ang_err < ang_err_tol:
            break
        if orient_weight == 0.0 and pos_err < pos_tol:
            break  # position phase done; orientation handled by later phases
        jac = site_jacobian(model, data, site)
        dq = jac.T @ w @ np.linalg.solve(w @ jac @ jac.T @ w.T + damping**2 * eye6, w @ err)
        step = float(np.max(np.abs(dq)))
        if step > STEP_CAP:
            dq *= STEP_CAP / step
        q = np.clip(q + dq, lower, upper)
    return q, pos_err, ang_err


def _iterate(
    model,
    data,
    site: str,
    arm: str,
    target_pos: np.ndarray,
    target_approach: np.ndarray,
    q0: np.ndarray,
    pos_tol: float,
    ang_tol: float,
    iters: int,
    damping: float,
) -> np.ndarray | None:
    """Solve from one seed via the orientation-weight continuation; None on failure."""
    q = np.asarray(q0, dtype=np.float64).copy()
    for i, orient_weight in enumerate(ORIENT_WEIGHT_RAMP):
        budget = iters if orient_weight in (0.0, FINAL_ORIENT_WEIGHT) else 40
        q, pos_err, ang_err = _dls_phase(
            model,
            data,
            site,
            arm,
            target_pos,
            target_approach,
            q,
            orient_weight,
            pos_tol,
            ang_tol,
            budget,
            damping,
        )
        if pos_err < pos_tol and ang_err < np.sin(ang_tol):
            return q
        if pos_err > 0.01:
            return None  # seed too far from any solution; try the next seed
    return None


def solve_ik(
    model,
    data,
    site: str,
    target_pos,
    target_approach,
    q0,
    pos_tol: float = POS_TOL,
    ang_tol: float = ANG_TOL,
    iters: int = MAX_ITERS,
    damping: float = DAMPING,
) -> np.ndarray:
    """Solve IK for the arm site to (target_pos, target_approach) starting from q0.

    Joint limits are clipped every iteration; failure raises ``IKUnreachable``.
    """
    arm = arm_of(site)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_approach = np.asarray(target_approach, dtype=np.float64)
    target_approach = target_approach / np.linalg.norm(target_approach)
    lower, upper = joint_limits(model, arm)

    q_seed = np.asarray(q0, dtype=np.float64).copy()
    if q_seed.shape != (5,):
        raise ValueError(f"q0 must have shape (5,), got {q_seed.shape}")

    home = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
    seeds = [q_seed]
    if np.linalg.norm(home - q_seed) > 1e-9:
        seeds.append(home)
    # Structured grasp-pose seeds: vary shoulder pan and wrist roll around home.
    for pan in STRUCTURED_SEED_PANS:
        for roll in STRUCTURED_SEED_ROLLS:
            variant = home.copy()
            variant[0] = pan
            variant[4] = roll
            if np.linalg.norm(variant - q_seed) > 1e-9:
                seeds.append(variant)
    # Deterministic random restarts keyed to the target so results reproduce
    # exactly across runs and machines.
    restart_key = int(abs(float(target_pos.sum())) * 1e6) ^ int(
        abs(float(target_approach.sum())) * 1e6
    )
    restart_rng = np.random.default_rng(restart_key & 0x7FFFFFFF)
    for _ in range(N_RANDOM_RESTARTS):
        seeds.append(restart_rng.uniform(lower, upper))

    for seed in seeds:
        q = _iterate(
            model,
            data,
            site,
            arm,
            target_pos,
            target_approach,
            seed,
            pos_tol,
            ang_tol,
            iters,
            damping,
        )
        if q is not None:
            return q
    raise IKUnreachable(site, target_pos)


def ik_above(model, data, arm: str, grasp_pose, height: float) -> np.ndarray:
    """IK to the grasp pose offset +height along world +Z, approach axis -Z."""
    grasp_pos = np.asarray(grasp_pose[0], dtype=np.float64)
    target_pos = grasp_pos + np.array([0.0, 0.0, float(height)], dtype=np.float64)
    target_approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return solve_ik(model, data, f"{arm}.ee", target_pos, target_approach, arm_q(data, arm))
