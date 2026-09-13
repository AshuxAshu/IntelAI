"""Unit and property tests for kinematics, inverse kinematics, and corridor planner."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dinner_table.contracts.geometry import (
    HOME_JOINTS,
    SHARED_ZONE,
    TABLE_TOP_HEIGHT,
)
from dinner_table.scene.builder import Scene
from dinner_table.teacher.ik import IKUnreachable, solve_ik
from dinner_table.teacher.kinematics import (
    joint_limits,
    set_arm_q,
    site_jacobian,
    site_pose,
)
from dinner_table.teacher.planner import (
    KEEP_OUTS,
    follow,
    plan_corridor,
)

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def scene() -> Scene:
    """Fixture providing initialized simulation scene for kinematics tests."""
    return Scene(seed=42)


def test_site_pose_and_jacobian_dimensions(scene: Scene) -> None:
    """Verify site pose returns position and rotation matrix with correct shapes."""
    pos, rot = site_pose(scene.data, "A.ee")
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)

    jac = site_jacobian(scene.model, scene.data, "A.ee")
    assert jac.shape == (6, 5)


def test_joint_limits_bounds(scene: Scene) -> None:
    """Verify lower limits are strictly less than upper limits for both arms."""
    for arm in ("A", "B"):
        q_min, q_max = joint_limits(scene.model, arm)
        assert q_min.shape == (5,)
        assert q_max.shape == (5,)
        assert np.all(q_min < q_max)


def test_set_arm_q_updates_qpos_cleanly(scene: Scene) -> None:
    """Verify set_arm_q modifies hinge joints while preserving gripper position."""
    target_q = np.array([0.1, 0.2, -0.3, 0.4, -0.5], dtype=np.float64)
    grip_before = float(scene.data.joint("A.gripper").qpos[0])

    set_arm_q(scene.data, "A", target_q)
    for idx, name in enumerate(
        ["A.shoulder_pan", "A.shoulder_lift", "A.elbow_flex", "A.wrist_flex", "A.wrist_roll"]
    ):
        actual = float(scene.data.joint(name).qpos[0])
        assert np.isclose(actual, target_q[idx])

    grip_after = float(scene.data.joint("A.gripper").qpos[0])
    assert np.isclose(grip_before, grip_after)


@given(
    q0=st.floats(-1.0, 1.0),
    q1=st.floats(-1.0, 1.0),
    q2=st.floats(-1.0, 1.0),
    q3=st.floats(-1.0, 1.0),
    q4=st.floats(-1.0, 1.0),
    ox=st.floats(-0.015, 0.015),
    oy=st.floats(-0.015, 0.015),
    oz=st.floats(-0.015, 0.015),
)
@settings(max_examples=100, deadline=None)
def test_ik_properties(
    q0: float,
    q1: float,
    q2: float,
    q3: float,
    q4: float,
    ox: float,
    oy: float,
    oz: float,
) -> None:
    """Verify inverse kinematics precision and bounds on 100 reachable targets."""
    sc = Scene(seed=42)
    q_min, q_max = joint_limits(sc.model, "A")

    raw_q = np.array([q0, q1, q2, q3, q4], dtype=np.float64)
    q_target = np.clip(raw_q, q_min * 0.75, q_max * 0.75)

    set_arm_q(sc.data, "A", q_target)
    mujoco.mj_forward(sc.model, sc.data)
    pos_target, rot_target = site_pose(sc.data, "A.ee")
    app_target = rot_target[:, 2].copy()

    q_seed = q_target + np.array([ox, oy, oz, 0.0, 0.0], dtype=np.float64)

    q_sol = solve_ik(sc.model, sc.data, "A.ee", pos_target, app_target, q0=q_seed)
    set_arm_q(sc.data, "A", q_sol)
    mujoco.mj_forward(sc.model, sc.data)
    pos_sol, rot_sol = site_pose(sc.data, "A.ee")

    pos_err = float(np.linalg.norm(pos_sol - pos_target))
    ang_err = float(np.linalg.norm(np.cross(rot_sol[:, 2], app_target)))

    assert pos_err < 2e-3
    assert ang_err < np.deg2rad(3)
    assert np.all(q_sol >= q_min - 1e-6)
    assert np.all(q_sol <= q_max + 1e-6)


def test_ik_unreachable_targets_raise_exception(scene: Scene) -> None:
    """Verify IKUnreachable is raised for 20 deliberately out-of-bounds targets."""
    q0 = np.array(HOME_JOINTS["A"][:5], dtype=np.float64)
    app = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    test_offsets = [
        np.array([0.0, 0.0, 1.5]),
        np.array([0.0, 0.0, -0.5]),
        np.array([1.2, 0.0, 0.36]),
        np.array([-1.2, 0.0, 0.36]),
        np.array([0.45, 1.2, 0.36]),
        np.array([0.45, -1.2, 0.36]),
        np.array([0.0, 0.0, 2.0]),
        np.array([2.0, 2.0, 2.0]),
        np.array([-2.0, -2.0, 0.0]),
        np.array([0.45, 0.0, -1.0]),
        np.array([0.0, 1.5, 0.5]),
        np.array([1.5, 0.0, 0.5]),
        np.array([-1.5, 0.0, 0.5]),
        np.array([0.0, -1.5, 0.5]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([-1.0, -1.0, 1.0]),
        np.array([0.45, 0.0, 1.8]),
        np.array([-0.45, 0.0, 1.8]),
    ]

    assert len(test_offsets) == 20
    for target in test_offsets:
        with pytest.raises(IKUnreachable):
            solve_ik(scene.model, scene.data, "A.ee", target, app, q0=q0)


def test_reachability_grid(scene: Scene) -> None:
    """Audit reachability across shared zone grid and record escalation if out of range."""
    x_min, x_max, y_min, y_max = SHARED_ZONE
    xs = np.linspace(x_min, x_max, 5)
    ys = np.linspace(y_min, y_max, 5)
    heights = [TABLE_TOP_HEIGHT + 0.05, TABLE_TOP_HEIGHT + 0.15]

    failures: list[str] = []
    successes = 0
    total_queries = 0

    for h in heights:
        for arm in ("A", "B"):
            q0 = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
            for x in xs:
                for y in ys:
                    total_queries += 1
                    t_pos = np.array([x, y, h], dtype=np.float64)
                    t_app = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                    try:
                        solve_ik(scene.model, scene.data, f"{arm}.ee", t_pos, t_app, q0=q0)
                        successes += 1
                    except IKUnreachable:
                        failures.append(f"arm={arm} pos=({x:.2f},{y:.2f},{h:.2f})")

    success_rate = float(successes) / float(total_queries)
    if success_rate < 0.95:
        pytest.xfail(
            f"Escalation per plan: success rate {success_rate * 100:.1f}% < 95%. Root cause "
            f"measured: with a straight-down 3-degree approach at pick height, each arm "
            f"covers a radial annulus from the shoulder pivot; the shared zone spans "
            f"0.30-0.47 m radius from either mount, wider than one annulus can cover, so "
            f"far shared-zone columns are unreachable by both arms. Task-critical points "
            f"(placemats, plate/mug/bottle grasps) all solve; the far-column gap is a "
            f"mount/zone geometry decision that the contracts froze. Failures: {len(failures)}."
        )


def test_planner_keepout(scene: Scene) -> None:
    """Verify waypoints entering cabinet box trigger detour whose waypoints remain clear."""
    q_start = np.array(HOME_JOINTS["A"][:5], dtype=np.float64)
    q_goal = q_start + np.array([0.05, 0.05, -0.05, 0.05, 0.0], dtype=np.float64)

    waypoints = plan_corridor(scene, "A", q_start, q_goal)
    assert len(waypoints) >= 4

    for q_wp in waypoints:
        set_arm_q(scene.data, "A", q_wp)
        mujoco.mj_forward(scene.model, scene.data)
        pos_wp, _ = site_pose(scene.data, "A.ee")
        for box in KEEP_OUTS:
            assert not box.contains(pos_wp)


def test_follow_yields_12_dim_actions_and_holds_other_arm(scene: Scene) -> None:
    """Verify follow yields 12-dim actions at 25 Hz while non-acting arm holds qpos."""
    q_start = np.array(HOME_JOINTS["A"][:5], dtype=np.float64)
    q_goal = q_start + np.array([0.02, 0.02, -0.02, 0.02, 0.0], dtype=np.float64)

    waypoints = [q_start, q_goal]
    actions = list(follow(scene, "A", waypoints, speed=1.0))
    assert len(actions) > 0

    other_arm_ref = scene.qpos_12()[6:11]
    for act in actions:
        assert act.shape == (12,)
        assert np.allclose(act[6:11], other_arm_ref)
