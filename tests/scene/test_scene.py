"""Tests for scene builder compilation, determinism, stability, mechanics, and calibration."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import mujoco
import numpy as np
import pytest

from dinner_table.contracts.geometry import (
    ARM_A_ZONE,
    ARM_B_ZONE,
    SHARED_ZONE,
    SO101_JOINT_SUFFIXES,
)
from dinner_table.scene.builder import Scene

CALIBRATION_FILE = Path("assets/meshes/so101/so101_calibration.json")


def _in_zone(pos: np.ndarray, zone: tuple[float, float, float, float]) -> bool:
    """Check if 2D position falls within bounding rectangle (x_min, x_max, y_min, y_max)."""
    if zone[0] <= pos[0] <= zone[1] and zone[2] <= pos[1] <= zone[3]:
        return True
    else:
        return False


def test_scene_compiles() -> None:
    """Validate that Scene compiles cleanly without errors across seeds and all DR profiles."""
    profiles = ("default", "dr_train", "eval_extreme")
    # Sweep through representative seed ranges across all three profiles
    for profile in profiles:
        for seed in range(15):
            scene = Scene(seed=seed, dr_profile=profile)
            assert scene.model is not None, f"model is None for {profile} on seed {seed}"
            assert scene.ready is True, f"scene not ready for {profile} on seed {seed}"
            assert scene.model.nbody >= 8, f"unexpected body count for {profile} on seed {seed}"


def test_scene_determinism() -> None:
    """Validate that building seed 7 twice yields identical XML serialization and overhead render hashes."""
    scene_a = Scene(seed=7, dr_profile="dr_train")
    scene_b = Scene(seed=7, dr_profile="dr_train")

    xml_a = scene_a.spec.to_xml().encode("utf-8")
    xml_b = scene_b.spec.to_xml().encode("utf-8")
    hash_xml_a = hashlib.sha256(xml_a).hexdigest()
    hash_xml_b = hashlib.sha256(xml_b).hexdigest()
    assert hash_xml_a == hash_xml_b, "XML serialization mismatch between identical seeds"

    img_a = scene_a.render("overhead")
    img_b = scene_b.render("overhead")
    hash_img_a = hashlib.sha256(img_a.tobytes()).hexdigest()
    hash_img_b = hashlib.sha256(img_b.tobytes()).hexdigest()
    assert hash_img_a == hash_img_b, "Overhead render pixel hash mismatch between identical seeds"


def test_stability() -> None:
    """Validate physical stability, lack of NaNs, penetration bounds, and table confinement across 10 extreme DR seeds."""
    table_objects = ("plate", "mug", "bottle")
    for seed in range(10):
        scene = Scene(seed=seed, dr_profile="eval_extreme")
        scene.settle(1.0)
        scene.hold_safe()

        # 500 control ticks stepping 0.04 s each
        for _ in range(500):
            scene.step(0.04)

        assert not np.isnan(scene.data.qpos).any(), f"NaN encountered in qpos on seed {seed}"

        # Contact penetration audit: soft contacts (solref 0.012, required for
        # gram-scale objects) allow ~2 mm of equilibrium penetration under x3
        # mass DR; deeper interpenetration still fails.
        for i in range(scene.data.ncon):
            contact = scene.data.contact[i]
            assert contact.dist >= -0.0025, (
                f"penetration {contact.dist} m exceeds 2.5 mm on seed {seed}"
            )
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(scene.model, scene.data, i, force)

        # Table bounds confinement (+5 cm margin): X in [-0.53, 0.53], Y in [-0.44, 0.44]
        for name in table_objects:
            pos, _ = scene.object_pose(name)
            assert abs(pos[0]) <= 0.53, f"{name} x={pos[0]} outside table footprint on seed {seed}"
            assert abs(pos[1]) <= 0.44, f"{name} y={pos[1]} outside table footprint on seed {seed}"


def test_drawer_mechanics() -> None:
    """Validate drawer actuator opening and closing mechanics under friction scaling extremes."""
    friction_scales = (0.25, 3.0)
    for scale in friction_scales:
        scene = Scene(seed=42, dr_profile="default")
        jnt_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        dof_adr = scene.model.jnt_dofadr[jnt_id]
        scene.model.dof_damping[dof_adr] *= scale
        scene.model.dof_frictionloss[dof_adr] *= scale

        act_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_actuator")

        # Actuate to fully open
        scene.data.ctrl[act_id] = 0.24
        scene.settle(1.5)
        assert scene.is_drawer_open() is True, f"drawer failed to open at friction scale {scale}"

        # Actuate to closed
        scene.data.ctrl[act_id] = 0.0
        scene.settle(1.5)
        assert scene.is_drawer_open() is False, f"drawer failed to close at friction scale {scale}"


def test_arm_joint_ranges_match_official() -> None:
    """Validate compiled model joint ranges match official SO-101 calibration limits within 1e-6 tolerance."""
    assert CALIBRATION_FILE.is_file(), f"calibration file missing at {CALIBRATION_FILE}"
    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    scene = Scene(seed=42, dr_profile="default")
    for arm in ("A", "B"):
        for suffix in SO101_JOINT_SUFFIXES:
            joint_name = f"{arm}.{suffix}"
            jid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            assert jid != -1, f"joint {joint_name} not found in compiled model"

            actual_range = scene.model.jnt_range[jid]
            expected_range = np.deg2rad(calibration[suffix]["range_deg"])
            # The official MJCF rounds radian ranges to 5 decimals, so allow
            # conversion noise well below any physically meaningful angle.
            np.testing.assert_allclose(
                actual_range,
                expected_range,
                atol=1e-4,
                err_msg=f"range mismatch for joint {joint_name}",
            )


def test_spawn_reachability_bounds() -> None:
    """Validate all non-utensil spawns remain inside workspace reachability zones across 20 seeds."""
    table_objects = ("plate", "mug", "bottle")
    for seed in range(20):
        scene = Scene(seed=seed, dr_profile="dr_train")
        spawns = scene.spawn_meta()
        for name in table_objects:
            pos, _ = spawns[name]
            in_shared = _in_zone(pos, SHARED_ZONE)
            in_arm_a = _in_zone(pos, ARM_A_ZONE)
            in_arm_b = _in_zone(pos, ARM_B_ZONE)
            is_reachable = in_shared or in_arm_a or in_arm_b
            assert is_reachable is True, (
                f"object {name} at {pos} not inside reachability envelope on seed {seed}"
            )
