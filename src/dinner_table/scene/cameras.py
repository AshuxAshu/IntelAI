"""Multi-camera sensor rig encapsulating offscreen RGB and depth renderers."""

from __future__ import annotations

from dataclasses import dataclass
import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import (
    CAMERA_NAMES,
    OVERHEAD_RESOLUTION,
    POLICY_IMAGE_SIZE,
)


class CameraRigError(DinnerTableError):
    """Exception raised for camera rendering or configuration errors."""


@dataclass(frozen=True)
class Observation:
    """Bundle of synchronized visual observations from all robot and scene cameras."""

    wrist_A: np.ndarray
    wrist_B: np.ndarray
    overhead: np.ndarray
    depth: np.ndarray


class CameraRig:
    """Multi-camera sensor rig managing offscreen renderers and cached intrinsics."""

    def __init__(self, model_or_scene: object) -> None:
        """Initialize dedicated offscreen renderers for all cameras in CAMERA_NAMES."""
        if hasattr(model_or_scene, "model"):
            self._model: mujoco.MjModel = getattr(model_or_scene, "model")
            self._scene = model_or_scene
        else:
            self._model = model_or_scene
            self._scene = None

        self._renderers: dict[str, mujoco.Renderer] = {}
        for cam_name in CAMERA_NAMES:
            if cam_name in ("wrist_A", "wrist_B"):
                h, w = POLICY_IMAGE_SIZE
            else:
                h, w = OVERHEAD_RESOLUTION
            self._renderers[cam_name] = mujoco.Renderer(self._model, h, w)

        self._depth_renderer = mujoco.Renderer(
            self._model, OVERHEAD_RESOLUTION[0], OVERHEAD_RESOLUTION[1]
        )
        self._depth_renderer.enable_depth_rendering()

        self._cached_intrinsics: dict[str, np.ndarray] = {}
        for cam_name in CAMERA_NAMES:
            self._cached_intrinsics[cam_name] = self._compute_intrinsics(cam_name)

    def _compute_intrinsics(self, camera: str) -> np.ndarray:
        """Compute 3x3 intrinsic matrix K from camera vertical field of view."""
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id == -1:
            raise CameraRigError(f"camera not found in model: {camera}")
        fovy = float(self._model.cam_fovy[cam_id])

        if camera in ("wrist_A", "wrist_B"):
            h, w = POLICY_IMAGE_SIZE
        else:
            h, w = OVERHEAD_RESOLUTION

        f = float(h) / (2.0 * np.tan(np.deg2rad(fovy) * 0.5))
        return np.array(
            [
                [f, 0.0, float(w) * 0.5],
                [0.0, f, float(h) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def intrinsics(self, camera: str) -> np.ndarray:
        """Return cached analytical camera intrinsic matrix K without recomputation."""
        if camera not in self._cached_intrinsics:
            raise CameraRigError(f"unknown camera name: {camera}")
        return self._cached_intrinsics[camera]

    def render(self, camera: str, data: mujoco.MjData) -> np.ndarray:
        """Render uint8 RGB image from specified camera name."""
        if camera not in self._renderers:
            raise CameraRigError(f"unknown camera name: {camera}")
        renderer = self._renderers[camera]
        renderer.update_scene(data, camera=camera)
        return renderer.render()

    def render_depth(self, data: mujoco.MjData) -> np.ndarray:
        """Render float32 depth map in meters from the overhead camera."""
        self._depth_renderer.update_scene(data, camera="overhead")
        return self._depth_renderer.render()

    def observe(self, data: mujoco.MjData) -> Observation:
        """Render and return all 4 camera views synchronously."""
        wrist_a = self.render("wrist_A", data)
        wrist_b = self.render("wrist_B", data)
        overhead = self.render("overhead", data)
        depth = self.render_depth(data)

        return Observation(
            wrist_A=wrist_a,
            wrist_B=wrist_b,
            overhead=overhead,
            depth=depth,
        )


def resize_overhead_policy(rgb: np.ndarray) -> np.ndarray:
    """Produce the 128x128x3 uint8 policy view via center crop and resize."""
    h, w, _ = rgb.shape
    if w > h:
        start_x = (w - h) // 2
        cropped = rgb[:, start_x : start_x + h]
    else:
        if h > w:
            start_y = (h - w) // 2
            cropped = rgb[start_y : start_y + w, :]
        else:
            cropped = rgb

    target_h, target_w = POLICY_IMAGE_SIZE
    ch, cw, _ = cropped.shape
    row_idx = (np.arange(target_h) * ch // target_h).astype(int)
    col_idx = (np.arange(target_w) * cw // target_w).astype(int)
    return cropped[row_idx[:, None], col_idx]
