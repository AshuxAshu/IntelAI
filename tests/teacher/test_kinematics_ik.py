"""Property and integration tests for teacher kinematics, IK, and the corridor planner.

Amendment 1 notes: the official SO-101 is a 5-DOF arm whose approach alignment
is nearly null in wrist roll and whose top-down envelope ends ~10 cm above the
table. Property targets are therefore sampled from grasp-oriented
configurations (approach within 30 deg of straight down, site in the useful
workspace) and pre-verified feasible by a warm-started solve; the reachability
grid checks both-arm coverage at working height and union coverage at hover
height.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from dinner_table.contracts.geometry import (
    ARM_MOUNTS,
    HOME_JOINTS,
    SHARED_ZONE,
    TABLE_TOP_HEIGHT,
)
from dinner_table.scene.builder import Scene
from dinner_table.teacher.ik import IKUnreachable, ik_above, solve_ik
from dinner_table.teacher.kinematics import (
    arm_q,
    joint_limits,
    set_arm_q,
    site_jacobian,
    site_pose,
)
from dinner_table.teacher.planner import (
    CorridorBlocked,
    corridor_violation,
    follow,
    in_keepout,
    plan_corridor,
)

N_PROPERTY_CASES = 100
N_UNREACHABLE_CASES = 20
SEED = 20260915
DOWN = np.array([0.0, 0.0, -1.0])


def _fk(scene: Scene, arm: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    set_arm_q(scene.data, arm, q)
    mujoco.mj_forward(scene.model, scene.data)
    pos, rot = site_pose(scene.data, f"{arm}.ee")
    return pos, rot


def _grasp_oriented_case(
    scene: Scene, rng: np.random.Generator, arm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sample (target_pos, target_approach, q_true) from a grasp-oriented config."""
    model, data = scene.model, scene.data
    lower, upper = joint_limits(model, arm)
    mount = np.array(ARM_MOUNTS[arm])
    for _ in range(500):
        q_true = rng.uniform(lower, upper)
        set_arm_q(data, arm, q_true)
        mujoco.mj_forward(model, data)
        pos, rot = site_pose(data, f"{arm}.ee")
        grasp_like = (
            rot[:, 2] @ DOWN > np.cos(np.deg2rad(30))
            and 0.40 <= pos[2] <= 0.50
            and np.linalg.norm(pos - mount) < 0.35
        )
        if grasp_like:
            offset_dir = rng.normal(size=3)
            offset_dir /= np.linalg.norm(offset_dir)
            target_pos = pos + offset_dir * rng.uniform(0.0, 0.03)
            return target_pos, rot[:, 2].copy(), q_true
    return None


def test_ik_properties() -> None:
    """Grasp-oriented targets get in-limit solutions (<2 mm, <3 deg); far targets raise.

    Targets whose warm-started reference solve also fails are joint-limit-corner
    poses whose small offset exits the workspace; they are skipped as
    infeasible-by-construction. At least 95% of the verified-feasible targets
    must solve from the home seed.
    """
    scene = Scene(seed=0, dr_profile="default")
    model, data = scene.model, scene.data
    rng = np.random.default_rng(SEED)

    solved = feasible = 0
    for case in range(N_PROPERTY_CASES):
        arm = "A" if case % 2 == 0 else "B"
        sample = _grasp_oriented_case(scene, rng, arm)
        if sample is None:
            continue
        target_pos, target_approach, q_true = sample
        try:
            solve_ik(model, data, f"{arm}.ee", target_pos, target_approach, q_true)
        except IKUnreachable:
            continue  # offset target infeasible even from the true config
        feasible += 1

        try:
            q_sol = solve_ik(
                model,
                data,
                f"{arm}.ee",
                target_pos,
                target_approach,
                np.array(HOME_JOINTS[arm][:5], dtype=np.float64),
            )
        except IKUnreachable:
            continue  # basin miss from the cold start; counted against the rate
        lower, upper = joint_limits(model, arm)
        assert np.all(q_sol >= lower - 1e-9) and np.all(q_sol <= upper + 1e-9), (
            f"case {case}: solution outside joint limits"
        )
        pos_sol, rot_sol = _fk(scene, arm, q_sol)
        pos_err = np.linalg.norm(pos_sol - target_pos)
        ang_err = np.degrees(np.arccos(np.clip(float(rot_sol[:, 2] @ target_approach), -1.0, 1.0)))
        assert pos_err < 2e-3, f"case {case}: position error {pos_err:.4f} m"
        assert ang_err < 3.0, f"case {case}: approach error {ang_err:.2f} deg"
        solved += 1

    assert feasible >= 50, f"only {feasible} feasible targets sampled"
    success_rate = solved / feasible
    assert success_rate >= 0.95, (
        f"cold-start IK success {success_rate:.0%} < 95% on {feasible} feasible targets"
    )

    # Deliberately unreachable targets: far outside any arm's workspace envelope.
    for case in range(N_UNREACHABLE_CASES):
        arm = "A" if case % 2 == 0 else "B"
        reach_out = 1.0 if arm == "A" else -1.0  # toward and past the other arm
        target_pos = np.array(
            [
                reach_out * (0.55 + 0.2 * rng.uniform()),
                rng.uniform(-0.2, 0.2),
                TABLE_TOP_HEIGHT + rng.uniform(0.0, 0.3),
            ]
        )
        with pytest.raises(IKUnreachable):
            solve_ik(
                model,
                data,
                f"{arm}.ee",
                target_pos,
                DOWN,
                np.array(HOME_JOINTS[arm][:5], dtype=np.float64),
            )


def test_jacobian_shape() -> None:
    """Site Jacobian is (6, 5) and finite at a nominal pose."""
    scene = Scene(seed=0, dr_profile="default")
    for arm in ("A", "B"):
        jac = site_jacobian(scene.model, scene.data, f"{arm}.ee")
        assert jac.shape == (6, 5)
        assert np.all(np.isfinite(jac))


def test_reachability_grid() -> None:
    """5x5 grid over SHARED_ZONE: both arms at table+0.05, union reach at table+0.10."""
    scene = Scene(seed=0, dr_profile="default")
    model, data = scene.model, scene.data
    xs = np.linspace(SHARED_ZONE[0], SHARED_ZONE[1], 5)
    ys = np.linspace(SHARED_ZONE[2], SHARED_ZONE[3], 5)

    def solved(arm: str, pos: np.ndarray) -> bool:
        try:
            solve_ik(
                model,
                data,
                f"{arm}.ee",
                pos,
                DOWN,
                np.array(HOME_JOINTS[arm][:5], dtype=np.float64),
            )
            return True
        except IKUnreachable:
            return False

    # Working height: the shared zone must be dual-reach (handoff/pour territory).
    both_fails = 0
    for x in xs:
        for y in ys:
            if not (
                solved("A", np.array([x, y, TABLE_TOP_HEIGHT + 0.05]))
                and solved("B", np.array([x, y, TABLE_TOP_HEIGHT + 0.05]))
            ):
                both_fails += 1
    both_rate = 1.0 - both_fails / 25
    assert both_rate >= 0.95, f"dual-reach at table+0.05: {both_rate:.0%} < 95%"

    # Hover height: every cell reachable by at least one arm (union lens).
    union_fails = 0
    for x in xs:
        for y in ys:
            if not (
                solved("A", np.array([x, y, TABLE_TOP_HEIGHT + 0.10]))
                or solved("B", np.array([x, y, TABLE_TOP_HEIGHT + 0.10]))
            ):
                union_fails += 1
    union_rate = 1.0 - union_fails / 25
    assert union_rate >= 0.95, f"union reach at table+0.10: {union_rate:.0%} < 95%"


def test_planner_keepout() -> None:
    """Cabinet/under-table positions are rejected; planned corridors stay keep-out free."""
    # A waypoint inside the cabinet box is rejected by the corridor predicate.
    assert in_keepout(np.array([-0.28, 0.11, 0.40])) == "cabinet_body"
    assert in_keepout(np.array([-0.28, 0.11, 0.50])) is None  # above the cabinet
    assert in_keepout(np.array([0.20, 0.00, 0.30])) == "table_surface"  # under the table top
    assert in_keepout(np.array([0.20, 0.00, 0.50])) is None
    assert corridor_violation("A", np.array([-0.28, 0.11, 0.40])) == "cabinet_body"

    # Mutual-exclusion: each arm's exclusive territory is off-limits to the other arm,
    # while the shared zone stays open to both.
    assert corridor_violation("A", np.array([0.30, 0.00, 0.50])) == "arm_B_zone"
    assert corridor_violation("B", np.array([-0.30, 0.00, 0.50])) == "arm_A_zone"
    assert corridor_violation("A", np.array([-0.30, 0.00, 0.50])) is None
    assert corridor_violation("B", np.array([0.30, 0.00, 0.50])) is None
    assert corridor_violation("A", np.array([0.00, -0.22, 0.50])) is None
    assert corridor_violation("B", np.array([0.00, -0.22, 0.50])) is None

    scene = Scene(seed=0, dr_profile="default")
    model, data = scene.model, scene.data

    # A goal whose site sits under the table top cannot be planned: even the detour ends there.
    q_goal_bad = solve_ik(
        model,
        data,
        "A.ee",
        np.array([-0.25, 0.00, 0.30]),
        DOWN,
        np.array(HOME_JOINTS["A"][:5], dtype=np.float64),
    )
    with pytest.raises(CorridorBlocked):
        plan_corridor(scene, "A", np.array(HOME_JOINTS["A"][:5], dtype=np.float64), q_goal_bad)

    # A healthy A-zone corridor: every returned waypoint stays outside all keep-outs.
    q_goal = solve_ik(
        model,
        data,
        "A.ee",
        np.array([-0.08, -0.20, TABLE_TOP_HEIGHT + 0.05]),
        DOWN,
        np.array(HOME_JOINTS["A"][:5], dtype=np.float64),
    )
    waypoints = plan_corridor(scene, "A", np.array(HOME_JOINTS["A"][:5], dtype=np.float64), q_goal)
    assert len(waypoints) == 4
    for q in waypoints:
        pos, _ = _fk(scene, "A", q)
        assert in_keepout(pos) is None, f"waypoint FK {pos.tolist()} inside a keep-out"

    # Detour: a start config dipping below the table top triggers the lift-first fallback,
    # and the detour path (after the lift) stays outside all keep-outs.
    q_start_low = solve_ik(
        model,
        data,
        "A.ee",
        np.array([-0.25, 0.00, 0.34]),
        DOWN,
        np.array(HOME_JOINTS["A"][:5], dtype=np.float64),
    )
    detour = plan_corridor(scene, "A", q_start_low, q_goal)
    assert len(detour) == 3, f"expected detour path, got {len(detour)} waypoints"
    q_lift = detour[1]
    pos_lift, _ = _fk(scene, "A", q_lift)
    assert abs(pos_lift[2] - (TABLE_TOP_HEIGHT + 0.10)) < 5e-3
    assert in_keepout(pos_lift) is None
    for i in range(33):
        alpha = i / 32
        q_sample = (1.0 - alpha) * q_lift + alpha * detour[2]
        pos, _ = _fk(scene, "A", q_sample)
        assert in_keepout(pos) is None, f"detour sample {i} FK {pos.tolist()} inside a keep-out"


def test_follow_merges_and_respects_velocity_limits() -> None:
    """follow yields 12-dim merged targets at 25 Hz honoring per-joint velocity limits."""
    scene = Scene(seed=0, dr_profile="default")
    scene.hold_safe()
    q_start = arm_q(scene.data, "A")
    q_goal = q_start + np.array([0.0, 0.2, -0.4, 0.2, 0.0])
    held_other = scene.qpos_12()[6:12].copy()
    gripper_a = scene.qpos_12()[5]

    actions = list(follow(scene, "A", [q_start, q_goal], speed=1.0))
    assert len(actions) >= 2
    for a in actions:
        assert a.shape == (12,)
        assert a[5] == gripper_a  # acting arm's gripper holds
        np.testing.assert_allclose(a[6:12], held_other)  # other arm holds everything

    dt = 1.0 / 25
    limits = np.array([2.0, 2.0, 2.5, 3.0, 3.0])
    prev = actions[0][0:5]
    for a in actions[1:]:
        step = np.abs(a[0:5] - prev)
        assert np.all(step <= limits * dt + 1e-9), f"velocity limit violated: {step}"
        prev = a[0:5]
    np.testing.assert_allclose(actions[-1][0:5], q_goal, atol=1e-9)

    # A slower speed stretches the trajectory, never speeds it up.
    actions_slow = list(follow(scene, "A", [q_start, q_goal], speed=0.5))
    assert len(actions_slow) >= len(actions)


def test_ik_above_targets_vertical_offset() -> None:
    """ik_above solves to the grasp pose lifted along world +Z with approach -Z."""
    scene = Scene(seed=0, dr_profile="default")
    scene.hold_safe()
    grasp_pos = np.array([-0.08, -0.20, TABLE_TOP_HEIGHT + 0.05])
    q = ik_above(scene.model, scene.data, "A", (grasp_pos, DOWN), 0.05)
    pos, rot = _fk(scene, "A", q)
    assert np.linalg.norm(pos - (grasp_pos + np.array([0.0, 0.0, 0.05]))) < 2e-3
    assert np.degrees(np.arccos(np.clip(float(rot[:, 2] @ DOWN), -1.0, 1.0))) < 3.0
