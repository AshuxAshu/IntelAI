"""Tests for scene object catalog, spawn pose sampling bounds, and MuJoCo instantiation."""

from __future__ import annotations

import mujoco
import numpy as np

from dinner_table.scene.objects import (
    OBJECT_CATALOG,
    UTENSIL_XY_JITTER_M,
    DrProfile,
    instantiate,
    sample_spawns,
)


def test_sample_spawns_bounds_and_quaternion_norm() -> None:
    """Validate spawn poses remain strictly within jitter bounds with normalized quaternions across seeds."""
    dr = DrProfile(spawn_xy_jitter_m=0.06, spawn_yaw_jitter_rad=0.44)
    tolerance = 1e-6
    for seed in range(100):
        rng = np.random.default_rng(seed)
        spawns = sample_spawns(rng, dr)
        for name, (pos, quat) in spawns.items():
            spec = OBJECT_CATALOG[name]
            anchor = spec.spawn_anchor
            if name == "drawer_top":
                jitter = 0.0
            elif name.startswith("spoon") or name.startswith("fork"):
                jitter = UTENSIL_XY_JITTER_M
            else:
                jitter = dr.spawn_xy_jitter_m

            assert abs(pos[0] - anchor[0]) <= jitter + tolerance, (
                f"x out of bounds for {name} on seed {seed}"
            )
            assert abs(pos[1] - anchor[1]) <= jitter + tolerance, (
                f"y out of bounds for {name} on seed {seed}"
            )
            assert abs(pos[2] - anchor[2]) <= tolerance, f"z modified for {name} on seed {seed}"

            quat_norm = float(np.linalg.norm(quat))
            assert abs(quat_norm - 1.0) <= tolerance, (
                f"quaternion not normalized for {name} on seed {seed}"
            )


def test_sample_spawns_deterministic_bitwise() -> None:
    """Validate identical seed yields bitwise identical spawn positions and orientations."""
    dr = DrProfile(spawn_xy_jitter_m=0.06, spawn_yaw_jitter_rad=0.44)
    for seed in range(20):
        rng1 = np.random.default_rng(seed)
        spawns1 = sample_spawns(rng1, dr)
        rng2 = np.random.default_rng(seed)
        spawns2 = sample_spawns(rng2, dr)
        for name in OBJECT_CATALOG:
            assert np.array_equal(spawns1[name][0], spawns2[name][0]), (
                f"position mismatch for {name} on seed {seed}"
            )
            assert np.array_equal(spawns1[name][1], spawns2[name][1]), (
                f"orientation mismatch for {name} on seed {seed}"
            )


def test_utensil_spawns_inside_drawer_footprint() -> None:
    """Validate all utensil spawns remain inside the drawer envelope across 100 seeds."""
    dr = DrProfile(spawn_xy_jitter_m=0.06, spawn_yaw_jitter_rad=0.44)
    utensils = ("spoon_1", "spoon_2", "fork_1", "fork_2")
    for seed in range(100):
        rng = np.random.default_rng(seed)
        spawns = sample_spawns(rng, dr)
        for name in utensils:
            pos, _ = spawns[name]
            assert -0.30 <= pos[0] <= -0.14, f"utensil {name} x={pos[0]} outside [-0.30, -0.14]"
            assert 0.09 <= pos[1] <= 0.13, f"utensil {name} y={pos[1]} outside [0.09, 0.13]"


def test_instantiate_on_bare_spec_compiles_with_catalog_mass() -> None:
    """Validate each catalog object instantiates cleanly on a bare MjSpec with exact mass."""
    for name, obj_spec in OBJECT_CATALOG.items():
        spec = mujoco.MjSpec()
        pos = np.array(obj_spec.spawn_anchor, dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        instantiate(spec, name, (pos, quat))
        model = spec.compile()
        assert model is not None, f"compilation failed for {name}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id != -1, f"body {name} missing in compiled model"
        compiled_mass = float(model.body_mass[body_id])
        assert abs(compiled_mass - obj_spec.mass_kg) <= 1e-6, (
            f"mass mismatch for {name}: expected {obj_spec.mass_kg}, got {compiled_mass}"
        )
