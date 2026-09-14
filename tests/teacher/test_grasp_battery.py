"""Grasp feasibility battery for table setting objects across domain randomization seeds."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from dinner_table.contracts.geometry import HOME_JOINTS
from dinner_table.scene.builder import Scene
from dinner_table.teacher.ik import IKUnreachable, solve_ik
from dinner_table.teacher.kinematics import joint_limits, set_arm_q, site_pose

pytestmark = pytest.mark.fast

SEEDS_COUNT = 10
POS_TOL_M = 0.002
ANG_TOL_RAD = 0.05235987755982988


def _check_solution(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: str,
    q: np.ndarray,
    target_pos: np.ndarray,
    target_app: np.ndarray,
    pos_tol: float = POS_TOL_M,
    ang_tol: float = ANG_TOL_RAD,
) -> None:
    """Validate that solved joint angles strictly satisfy position, orientation, and joint limits."""
    q_min, q_max = joint_limits(model, arm)
    assert np.all(q >= q_min - 1e-4), (
        f"Joint lower limit violation on arm {arm}: q={q}, q_min={q_min}"
    )
    assert np.all(q <= q_max + 1e-4), (
        f"Joint upper limit violation on arm {arm}: q={q}, q_max={q_max}"
    )

    set_arm_q(data, arm, q)
    mujoco.mj_forward(model, data)
    fk_pos, fk_rot = site_pose(data, f"{arm}.ee")

    pos_err = float(np.linalg.norm(fk_pos - target_pos))
    tool_z = fk_rot[:, 2]
    app_unit = target_app / np.linalg.norm(target_app)
    ang_err = float(np.linalg.norm(np.cross(tool_z, app_unit)))

    assert pos_err <= pos_tol, (
        f"FK position error {pos_err * 1000:.2f} mm exceeds tolerance {pos_tol * 1000:.1f} mm"
    )
    assert ang_err <= ang_tol, (
        f"FK angular error {np.degrees(ang_err):.2f} deg exceeds tolerance {np.degrees(ang_tol):.1f} deg"
    )


def test_plate_rim_grasp_battery() -> None:
    """Validate that plate rim grasps solve across 10 randomized seeds with sub-2mm error."""
    solved_count = 0

    for seed in range(SEEDS_COUNT):
        scene = Scene(seed=seed, dr_profile="dr_train")
        scene.settle(1.0)
        model, data = scene.model, scene.data

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        pos = np.array(data.xpos[bid])

        solved = False
        for arm, mount_x in (("A", 0.45), ("B", -0.45)):
            v = pos[:2] - np.array([mount_x, 0.0])
            dist = float(np.linalg.norm(v))
            for factor in (-0.080, -0.070, 0.070, 0.080):
                rim_offset = (v / dist) * factor
                target = pos + np.array([rim_offset[0], rim_offset[1], 0.0])
                approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                q0 = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
                try:
                    q_sol = solve_ik(model, data, f"{arm}.ee", target, approach, q0=q0)
                    _check_solution(model, data, arm, q_sol, target, approach)
                    solved = True
                    break
                except (IKUnreachable, AssertionError):
                    pass
            if solved:
                break

        if solved:
            solved_count += 1
        scene.close()

    assert solved_count == SEEDS_COUNT, (
        f"Plate grasp solved only {solved_count}/{SEEDS_COUNT} seeds"
    )


def test_mug_handle_grasp_battery() -> None:
    """Validate that mug handle grasps solve across 10 randomized seeds with sub-2mm error."""
    solved_count = 0

    for seed in range(SEEDS_COUNT):
        scene = Scene(seed=seed, dr_profile="dr_train")
        scene.settle(1.0)
        model, data = scene.model, scene.data

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        mpos = np.array(data.xpos[bid], dtype=np.float64)
        mmat = np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3)

        mgp = mpos + mmat @ np.array([0.05, 0.0, 0.0])
        mappr = mpos - mgp
        mappr_norm = float(np.linalg.norm(mappr))
        mappr = mappr / mappr_norm

        q0 = np.array(HOME_JOINTS["B"][:5], dtype=np.float64)
        q_sol = solve_ik(model, data, "B.ee", mgp, mappr, q0=q0)
        _check_solution(model, data, "B", q_sol, mgp, mappr)
        solved_count += 1
        scene.close()

    assert solved_count == SEEDS_COUNT, f"Mug grasp solved only {solved_count}/{SEEDS_COUNT} seeds"


def test_bottle_grasp_battery() -> None:
    """Validate that bottle grasps solve along line-of-approach across 10 randomized seeds."""
    solved_count = 0

    for seed in range(SEEDS_COUNT):
        scene = Scene(seed=seed, dr_profile="dr_train")
        scene.settle(1.0)
        model, data = scene.model, scene.data

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
        pos = np.array(data.xpos[bid])
        bgp = pos + np.array([0.0, 0.0, 0.05])

        solved = False
        for az_idx in range(16):
            az = np.array(
                [np.cos(az_idx * np.pi / 8.0), np.sin(az_idx * np.pi / 8.0), 0.0],
                dtype=np.float64,
            )
            for arm in ("A", "B"):
                q0 = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
                try:
                    q_sol = solve_ik(model, data, f"{arm}.ee", bgp, az, q0=q0)
                    _check_solution(model, data, arm, q_sol, bgp, az)
                    solved = True
                    break
                except (IKUnreachable, AssertionError):
                    continue
            if solved:
                break
        if solved:
            solved_count += 1
        scene.close()

    assert solved_count == SEEDS_COUNT, (
        f"Bottle grasp solved only {solved_count}/{SEEDS_COUNT} seeds"
    )


def test_placemat_placement_battery() -> None:
    """Validate that placemat 1 and 2 target placements solve cleanly with natural downward approach."""
    tilt_rad = float(np.radians(45.0))

    for seed in range(SEEDS_COUNT):
        scene = Scene(seed=seed, dr_profile="dr_train")
        model, data = scene.model, scene.data

        # Placemat 1 (Arm A side)
        t1 = np.array([0.22, 0.10, 0.42], dtype=np.float64)
        v1 = t1[:2] - np.array([0.45, 0.0], dtype=np.float64)
        v1 = v1 / float(np.linalg.norm(v1))
        app1 = np.array(
            [v1[0] * np.sin(tilt_rad), v1[1] * np.sin(tilt_rad), -np.cos(tilt_rad)],
            dtype=np.float64,
        )
        q0_a = np.array(HOME_JOINTS["A"][:5], dtype=np.float64)
        q_sol1 = solve_ik(model, data, "A.ee", t1, app1, q0=q0_a)
        _check_solution(model, data, "A", q_sol1, t1, app1)

        # Placemat 2 (Arm B side)
        t2 = np.array([-0.22, 0.10, 0.42], dtype=np.float64)
        v2 = t2[:2] - np.array([-0.45, 0.0], dtype=np.float64)
        v2 = v2 / float(np.linalg.norm(v2))
        app2 = np.array(
            [v2[0] * np.sin(tilt_rad), v2[1] * np.sin(tilt_rad), -np.cos(tilt_rad)],
            dtype=np.float64,
        )
        q0_b = np.array(HOME_JOINTS["B"][:5], dtype=np.float64)
        q_sol2 = solve_ik(model, data, "B.ee", t2, app2, q0=q0_b)
        _check_solution(model, data, "B", q_sol2, t2, app2)

        scene.close()
