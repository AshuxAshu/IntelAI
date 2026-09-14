"""Tests for the gravity-sag servo compensation loop."""

from __future__ import annotations

import numpy as np
import pytest

from dinner_table.contracts.geometry import HOME_JOINTS
from dinner_table.scene.builder import Scene
from dinner_table.teacher.servo import converge_ee, settled_ee_error

pytestmark = pytest.mark.fast

TILT_45_A = np.array([-0.7071068, 0.0, -0.7071068], dtype=np.float64)
TILT_45_B = np.array([0.7071068, 0.0, -0.7071068], dtype=np.float64)


@pytest.mark.parametrize("profile", ["default", "dr_train", "eval_extreme"])
def test_converge_ee_placemat_targets(profile: str) -> None:
    """Validate the converge loop cancels gravity sag below 5 mm across DR profiles."""
    for seed in (0, 1):
        scene = Scene(seed=seed, dr_profile=profile)
        cases = (
            ("A", np.array([0.22, 0.10, 0.42], dtype=np.float64), TILT_45_A),
            ("B", np.array([-0.22, 0.10, 0.42], dtype=np.float64), TILT_45_B),
        )
        for arm, target, approach in cases:
            q0 = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
            converge_ee(scene, arm, target, approach, q0)
            err = settled_ee_error(scene, arm, target)
            assert err < 0.005, (
                f"arm {arm} seed {seed} profile {profile}: converged error {err * 1000:.2f} mm"
            )
        scene.close()


def test_converge_ee_improves_raw_sag() -> None:
    """Validate convergence lands closer to the target than the raw IK command."""
    scene = Scene(seed=0, dr_profile="default")
    target = np.array([0.22, 0.10, 0.42], dtype=np.float64)
    q0 = np.array(HOME_JOINTS["A"][:5], dtype=np.float64)
    converge_ee(scene, "A", target, TILT_45_A, q0)
    converged_err = settled_ee_error(scene, "A", target)

    # Raw command baseline in a fresh scene.
    scene.close()
    scene = Scene(seed=0, dr_profile="default")
    from dinner_table.teacher.ik import solve_ik
    from dinner_table.teacher.kinematics import set_arm_q

    q = solve_ik(scene.model, scene.data, "A.ee", target, TILT_45_A, q0=q0)
    set_arm_q(scene.data, "A", q)
    scene.settle(2.0)
    raw_err = settled_ee_error(scene, "A", target)
    scene.close()

    assert converged_err < raw_err, (
        f"converged {converged_err * 1000:.2f} mm should beat raw {raw_err * 1000:.2f} mm"
    )
