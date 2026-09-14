"""Tests for scene builder compilation, determinism, stability, mechanics, and calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
from dinner_table.teacher.kinematics import set_arm_q

pytestmark = pytest.mark.fast

CALIBRATION_FILE = Path("assets/meshes/so101/so101_calibration.json")


def _in_zone(pos: np.ndarray, zone: tuple[float, float, float, float]) -> bool:
    """Check if 2D position falls within bounding rectangle (x_min, x_max, y_min, y_max)."""
    return bool(zone[0] <= pos[0] <= zone[1] and zone[2] <= pos[1] <= zone[3])


def test_scene_compiles() -> None:
    """Validate that Scene compiles cleanly without errors or warnings across seeds and profiles."""
    profiles = ("default", "dr_train", "eval_extreme")
    for profile in profiles:
        for seed in range(100):
            scene = Scene(seed=seed, dr_profile=profile)
            assert scene.model is not None, f"model is None for {profile} on seed {seed}"
            assert scene.ready is True, f"scene not ready for {profile} on seed {seed}"
            assert scene.model.nbody >= 8, f"unexpected body count for {profile} on seed {seed}"
            warning_counts = [int(w.number) for w in scene.data.warning]
            assert sum(warning_counts) == 0, (
                f"MuJoCo warnings raised for {profile} on seed {seed}: {warning_counts}"
            )


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

        # Contact penetration audit: no penetration deeper than 1 mm
        for i in range(scene.data.ncon):
            contact = scene.data.contact[i]
            assert contact.dist >= -0.001, (
                f"penetration {contact.dist} m exceeds 1 mm on seed {seed}"
            )
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(scene.model, scene.data, i, force)

        # Table bounds confinement (+5 cm margin): X in [-0.50, 0.50], Y in [-0.30, 0.30]
        for name in table_objects:
            pos, _ = scene.object_pose(name)
            assert abs(pos[0]) <= 0.50, f"{name} x={pos[0]} outside table footprint on seed {seed}"
            assert abs(pos[1]) <= 0.30, f"{name} y={pos[1]} outside table footprint on seed {seed}"


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
            np.testing.assert_allclose(
                actual_range,
                expected_range,
                atol=1e-6,
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


def _jaw_gap_center(model: mujoco.MjModel, data: mujoco.MjData, arm: str) -> np.ndarray:
    """Return the world midpoint of the two jaw geoms at grasp aperture."""
    gripper_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}.gripper")
    lo = model.jnt_range[gripper_jnt][0]
    data.qpos[model.jnt_qposadr[gripper_jnt]] = lo + 0.5 * (0.0 - lo)
    mujoco.mj_forward(model, data)
    centers: list[np.ndarray] = []
    for name in (f"{arm}_jaw_fixed", f"{arm}_jaw_moving"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid and model.geom_group[g] == 0:
                centers.append(np.array(data.geom_xpos[g]))
    return (centers[0] + centers[1]) / 2.0


def test_ee_site_at_jaw_gap_center() -> None:
    """Validate the ee and grasp sites coincide with the jaw pinch point within 2 mm."""
    scene = Scene(seed=42, dr_profile="default")
    for arm in ("A", "B"):
        gap_center = _jaw_gap_center(scene.model, scene.data, arm)
        for site_name in (f"{arm}.ee", f"{arm}.grasp"):
            sid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            assert sid != -1, f"site {site_name} missing in compiled model"
            offset = float(np.linalg.norm(scene.data.site_xpos[sid] - gap_center))
            assert offset <= 0.002, (
                f"{site_name} is {offset * 1000:.1f} mm from the jaw gap center (limit 2 mm)"
            )


def test_grasp_hold_stability() -> None:
    """Validate a utensil held between the jaws does not slip over a 2 s hold."""
    scene = Scene(seed=42, dr_profile="default")
    model, data = scene.model, scene.data
    grasp_q = solve_ik_for_hold(scene)
    set_arm_q(data, "B", grasp_q)
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "B.ee")
    ee_pos = np.array(data.site_xpos[ee_id])

    fork_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fork_1")
    adr = model.jnt_qposadr[model.body_jntadr[fork_id]]
    data.qpos[adr : adr + 3] = ee_pos
    data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    gripper_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B.gripper")
    lo = model.jnt_range[gripper_jnt][0]
    data.qpos[model.jnt_qposadr[gripper_jnt]] = 0.115
    mujoco.mj_forward(model, data)

    act_ids = {
        n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"B.{n}")
        for n in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
    }
    for i, n in enumerate(
        ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    ):
        data.ctrl[act_ids[n]] = grasp_q[i]
    data.ctrl[act_ids["gripper"]] = lo
    data.qvel[:] = 0.0
    for _ in range(500):
        mujoco.mj_step(model, data)

    jaw_contacts = sum(
        1
        for i in range(data.ncon)
        if model.geom_bodyid[data.contact[i].geom1] == fork_id
        or model.geom_bodyid[data.contact[i].geom2] == fork_id
    )
    rel_before = float(np.linalg.norm(data.site_xpos[ee_id] - data.xpos[fork_id]))
    fork_z = float(data.xpos[fork_id][2])
    assert jaw_contacts >= 2, f"grasp did not engage: {jaw_contacts} contacts on fork"
    assert fork_z > 0.40, f"fork fell out of the grasp: z={fork_z:.3f}"

    for _ in range(1000):
        mujoco.mj_step(model, data)
    rel_after = float(np.linalg.norm(data.site_xpos[ee_id] - data.xpos[fork_id]))
    drift_mm = abs(rel_after - rel_before) * 1000.0
    assert drift_mm < 5.0, f"held fork drifted {drift_mm:.2f} mm over a 2 s hold (limit 5 mm)"


def solve_ik_for_hold(scene: Scene) -> np.ndarray:
    """Solve a reachable down-facing grasp pose over the B-side table."""
    from dinner_table.contracts.geometry import HOME_JOINTS
    from dinner_table.teacher.ik import solve_ik

    q0 = np.array(HOME_JOINTS["B"][:5], dtype=np.float64)
    target = np.array([-0.22, 0.10, 0.45], dtype=np.float64)
    return solve_ik(
        scene.model,
        scene.data,
        "B.ee",
        target,
        np.array([0.0, 0.0, -1.0]),
        q0=q0,
    )
