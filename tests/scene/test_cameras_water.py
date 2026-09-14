"""Acceptance tests for camera rig visibility, depth accuracy, water settlement, and isolation."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from dinner_table.contracts.geometry import POLICY_IMAGE_SIZE
from dinner_table.scene import water
from dinner_table.scene.builder import Scene
from dinner_table.scene.cameras import CameraRig, resize_overhead_policy

pytestmark = pytest.mark.fast

EXTENTS = {
    "plate": (0.09, 0.012),
    "mug": (0.04, 0.04),
    "bottle": (0.035, 0.10),
}


def test_cameras_visibility() -> None:
    """Validate that non-utensil objects have at least 95 percent area inside overhead view across 50 seeds."""
    img_w = 640.0
    img_h = 480.0

    for seed in range(50):
        scene = Scene(seed=seed, dr_profile="dr_train")
        scene.settle(0.5)

        cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        cam_pos = scene.data.cam_xpos[cam_id]
        cam_mat = scene.data.cam_xmat[cam_id].reshape(3, 3)
        k_mat = scene.camera_intrinsics("overhead")

        for name, (radius, half_h) in EXTENTS.items():
            bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = scene.data.xpos[bid]

            u_coords: list[float] = []
            v_coords: list[float] = []

            for dx in (-radius, radius):
                for dy in (-radius, radius):
                    for dz in (-half_h, half_h):
                        corner_pt = pos + np.array([dx, dy, dz], dtype=np.float64)
                        p_cam = cam_mat.T @ (corner_pt - cam_pos)
                        opt_x = p_cam[0]
                        opt_y = -p_cam[1]
                        opt_z = -p_cam[2]
                        u_val = float(k_mat[0, 0] * (opt_x / opt_z) + k_mat[0, 2])
                        v_val = float(k_mat[1, 1] * (opt_y / opt_z) + k_mat[1, 2])
                        u_coords.append(u_val)
                        v_coords.append(v_val)

            u_min = min(u_coords)
            u_max = max(u_coords)
            v_min = min(v_coords)
            v_max = max(v_coords)

            bbox_area = max(1e-6, (u_max - u_min) * (v_max - v_min))
            clip_u_min = max(0.0, u_min)
            clip_u_max = min(img_w, u_max)
            clip_v_min = max(0.0, v_min)
            clip_v_max = min(img_h, v_max)

            if clip_u_max > clip_u_min and clip_v_max > clip_v_min:
                inter_area = (clip_u_max - clip_u_min) * (clip_v_max - clip_v_min)
            else:
                inter_area = 0.0

            ratio = inter_area / bbox_area
            assert ratio >= 0.95, (
                f"Object {name} on seed {seed} visibility {ratio:.3f} is below 0.95 threshold"
            )
        scene.close()


def test_depth_range() -> None:
    """Validate that overhead render depth on tabletop matches geometric expectation within 5 mm."""
    scene = Scene(seed=42, dr_profile="default")
    depth = scene.render_depth()
    assert depth.shape == (480, 640), f"unexpected depth shape {depth.shape}"

    expected_depth = 1.35 - 0.36
    # Sample center tabletop area
    center_depth = float(depth[240, 320])
    depth_error = abs(center_depth - expected_depth)
    assert depth_error <= 0.005, (
        f"tabletop depth {center_depth:.5f} m deviates {depth_error:.5f} m from expectation {expected_depth:.5f} m (> 5 mm)"
    )
    scene.close()


def test_water_settle() -> None:
    """Validate water fill fraction and lack of NaNs after 2.0 s settle across 10 seeds under G1."""
    for seed in range(10):
        scene = Scene(seed=seed, dr_profile="dr_train")
        scene.settle(2.0)

        assert not np.isnan(scene.data.qpos).any(), f"NaN in qpos on seed {seed}"

        fill_bottle = scene.fill_fraction("bottle")
        fill_mug = scene.fill_fraction("mug")

        assert fill_bottle >= 0.90, f"bottle fill fraction {fill_bottle} < 0.90 on seed {seed}"
        assert fill_mug <= 0.10, f"mug fill fraction {fill_mug} unexpectedly high on seed {seed}"
        scene.close()


def test_water_dynamic_scale_encoding() -> None:
    """Validate that visual proxy liquid levels scale dynamically via model geom size."""
    scene = Scene(seed=42, dr_profile="default")
    assert scene.fill_fraction("bottle") >= 0.90
    assert scene.fill_fraction("mug") <= 0.05

    water.set_fill_fraction(scene.model, "bottle", 0.55)
    np.testing.assert_allclose(scene.fill_fraction("bottle"), 0.55, atol=1e-3)

    water.set_fill_fraction(scene.model, "mug", 0.45)
    np.testing.assert_allclose(scene.fill_fraction("mug"), 0.45, atol=1e-3)
    scene.close()


def test_camera_rig_and_policy_resize() -> None:
    """Validate CameraRig observation bundle, intrinsics caching, and policy image resizing."""
    scene = Scene(seed=42, dr_profile="default")
    rig = CameraRig(scene)

    obs = rig.observe(scene.data)
    assert obs.wrist_A.shape == (POLICY_IMAGE_SIZE[0], POLICY_IMAGE_SIZE[1], 3)
    assert obs.wrist_B.shape == (POLICY_IMAGE_SIZE[0], POLICY_IMAGE_SIZE[1], 3)
    assert obs.overhead.shape == (480, 640, 3)
    assert obs.depth.shape == (480, 640)
    assert obs.overhead.dtype == np.uint8
    assert obs.depth.dtype == np.float32

    # Cached intrinsics match
    k_overhead = rig.intrinsics("overhead")
    np.testing.assert_allclose(k_overhead, scene.camera_intrinsics("overhead"))

    # Policy resize produces (128, 128, 3)
    policy_img = resize_overhead_policy(obs.overhead)
    assert policy_img.shape == (128, 128, 3)
    assert policy_img.dtype == np.uint8
    scene.close()
