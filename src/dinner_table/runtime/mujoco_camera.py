"""MuJoCo-to-physicalai camera adapter: offscreen renders as Camera devices.

Subclasses Intel's Camera ABC so `physicalai run` YAML instantiation works.
The constructor deliberately takes the live scene object plus a camera name
(not a pre-built renderer): the scene must be shared with the robot, and the
runtime config wires both adapters to one scene instance.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
from physicalai.capture import Camera, Frame


class _SceneProtocol(Protocol):
    """Structural view of the scene surface this adapter consumes."""

    def render(self, camera: str) -> np.ndarray: ...


class MuJoCoCamera(Camera):
    """One named MuJoCo camera exposed through Intel's Camera ABC."""

    def __init__(self, scene: _SceneProtocol, camera_name: str, **kw: object) -> None:
        super().__init__(**kw)
        self._scene = scene
        self.camera_name = camera_name
        self._seq = 0

    def connect(self, timeout: float = 5.0) -> None:
        """Reset the frame counter; rendering needs no hardware setup."""
        self._seq = 0

    def read(self, timeout: float = 2.0) -> Frame:
        return self.read_latest()

    def read_latest(self) -> Frame:
        """Render the current simulation state from this camera."""
        self._seq += 1
        return Frame(
            data=self._scene.render(self.camera_name),
            timestamp=time.monotonic(),
            sequence=self._seq,
        )

    def _do_disconnect(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def device_id(self) -> str:
        return f"mujoco:{self.camera_name}"
