"""Weighted damped-least-squares IK for the 5-DOF SO-101 arms.

A 5-DOF arm cannot satisfy a full 6-D pose, so position rows are weighted
above orientation rows (the approach vector) and gripper yaw is planned
separately via wrist roll by the skill layer. Failure is explicit —
``IKUnreachable`` — never a garbage solution.

Two solve paths:

- **Approach-only** (``target_lateral=None``): cross-product orientation error
  with an orientation-weight continuation ramp — the A5 solver, bit-identical.
- **Approach + lateral axis** (``target_lateral`` given): axis-space
  formulation — errors ``0.08 * (target_axis - current_axis)`` with
  axis-space Jacobian rows. The vector-difference error
  has no +/- basin ambiguity (a cross product vanishes at the mirrored
  solution), which matters because the gripper jaws are symmetric.
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
    return np.concatenate([target_pos - pos, np.cross(rot[:, 2], target_approach)])


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

    Returns (q, pos_err, ang_err) where ang_err is the worst of the approach
    and lateral angular errors. The weighted step is
    ``dq = J^T W (W J J^T W^T + damping^2 I)^-1 W e`` — the least-squares
    solution of ``W J dq = W e``.
    """
    lower, upper = joint_limits(model, arm)
    w = np.diag([1.0, 1.0, 1.0, orient_weight, orient_weight, orient_weight])
    eye_n = np.eye(6, dtype=np.float64)
    # The orientation errors are cross products, so their norms are sin(angle);
    # comparing against sin(ang_tol) guarantees the true angles are in tolerance.
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
        dq = jac.T @ w @ np.linalg.solve(w @ jac @ jac.T @ w.T + damping**2 * eye_n, w @ err)
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
    for orient_weight in ORIENT_WEIGHT_RAMP:
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


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix so that skew(v) @ w == v x w."""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
    )


AXIS_SCALE = 0.08  # orientation-to-position error scale, tuned on the SO-101 envelope
AXIS_DAMPING2 = 1e-5
AXIS_STEP_CAP = 0.06
AXIS_ITERS = 180
AXIS_MARGIN = 0.004  # rad of joint-range margin kept inside the joint limits


def _axis_space_phase(
    model,
    data,
    site: str,
    arm: str,
    target_pos: np.ndarray,
    target_approach: np.ndarray,
    target_lateral: np.ndarray | None,
    q: np.ndarray,
    pos_tol: float,
    ang_tol: float,
    axis_index: int = 2,
) -> tuple[np.ndarray, float, float]:
    """One axis-space solve from seed q.

    ``axis_index`` selects which site axis ``target_approach`` constrains: 2 is
    the finger direction (top-down grasps), 1 is the jaw-spread axis (the side
    grasp, where only "fingers horizontal" matters and the reach direction is
    left to the solver). ``target_lateral`` adds the site-X row pair when given.

    Returns (q, pos_err, worst_axis_err) where worst_axis_err is the larger
    axis misalignment in radians.
    """
    lower, upper = joint_limits(model, arm)
    lower = lower + AXIS_MARGIN
    upper = upper - AXIS_MARGIN
    axis_tol = 2.0 * np.sin(ang_tol / 2.0)
    pos_err = axis_err = np.inf
    for _ in range(AXIS_ITERS):
        set_arm_q(data, arm, q)
        mujoco.mj_forward(model, data)
        pos, rot = site_pose(data, site)
        primary = rot[:, axis_index]
        rows = [target_pos - pos, AXIS_SCALE * (target_approach - primary)]
        jac = site_jacobian(model, data, site)
        blocks = [jac[:3], -AXIS_SCALE * _skew(primary) @ jac[3:]]
        if target_lateral is not None:
            xax = rot[:, 0]
            rows.append(AXIS_SCALE * (target_lateral - xax))
            blocks.append(-AXIS_SCALE * _skew(xax) @ jac[3:])
        err = np.concatenate(rows)
        pos_err = float(np.linalg.norm(err[:3]))
        axis_err = max(
            float(np.linalg.norm(err[3 * i : 3 * i + 3])) for i in range(1, len(rows))
        ) / AXIS_SCALE
        if pos_err < pos_tol and axis_err < axis_tol:
            break
        aug = np.vstack(blocks)
        dq = aug.T @ np.linalg.solve(
            aug @ aug.T + np.eye(len(err), dtype=np.float64) * AXIS_DAMPING2, err
        )
        step = float(np.max(np.abs(dq)))
        if step > AXIS_STEP_CAP:
            dq *= AXIS_STEP_CAP / step
        q = np.clip(q + dq, lower, upper)
    return q, pos_err, axis_err


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
    target_lateral=None,
    axis_index: int = 2,
) -> np.ndarray:
    """Solve IK for the arm site to (target_pos, target_approach[, target_lateral]).

    ``target_lateral`` optionally constrains the site's local X axis direction.
    ``axis_index`` selects the site axis
    that ``target_approach`` pins: 2 (default) is the finger direction, 1 is the
    jaw-spread axis used by the bottle's side grasp. Joint limits are clipped
    every iteration; failure raises ``IKUnreachable``.
    """
    arm = arm_of(site)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_approach = np.asarray(target_approach, dtype=np.float64)
    target_approach = target_approach / np.linalg.norm(target_approach)
    if target_lateral is not None:
        target_lateral = np.asarray(target_lateral, dtype=np.float64)
        target_lateral = target_lateral / np.linalg.norm(target_lateral)
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
        if target_lateral is None and axis_index == 2:
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
        else:
            q, pos_err, axis_err = _axis_space_phase(
                model, data, site, arm, target_pos, target_approach,
                target_lateral, seed, pos_tol, ang_tol, axis_index,
            )
            if pos_err < pos_tol and axis_err < 2.0 * np.sin(ang_tol / 2.0):
                return q
    raise IKUnreachable(site, target_pos)


def ik_above(model, data, arm: str, grasp_pose, height: float, target_lateral=None) -> np.ndarray:
    """IK to the grasp pose offset +height along world +Z, approach axis -Z."""
    grasp_pos = np.asarray(grasp_pose[0], dtype=np.float64)
    target_pos = grasp_pos + np.array([0.0, 0.0, float(height)], dtype=np.float64)
    target_approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return solve_ik(
        model, data, f"{arm}.ee", target_pos, target_approach, arm_q(data, arm),
        target_lateral=target_lateral,
    )
