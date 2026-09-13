"""MuJoCo-to-physicalai robot adapter: the simulation as a first-class Robot.

One robot = both arms. 12 joint_names in Intel's SO-101 order (A./B.
prefixed), exactly the layout of observation.state in the training dataset.
Physics advances inside send_action so the runtime's tick clock stays
authoritative. The Scene is consumed structurally: the isolation guard
forbids deployed code from importing the builder, so the adapter declares
only the scene surface it uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from physicalai.robot import Robot


class _SceneProtocol(Protocol):
    """Structural view of the scene surface this adapter consumes."""

    ready: bool

    def reset(self) -> None: ...

    def hold_safe(self) -> None: ...

    def qpos_12(self) -> np.ndarray: ...

    def set_targets(self, action: np.ndarray) -> None: ...

    def step(self, duration: float) -> None: ...


@dataclass(frozen=True)
class MuJoCoObservation:
    """Payload for the RobotObservation protocol.

    # NOTE: the installed physicalai RobotObservation is a pure Protocol and
    # cannot be instantiated; this dataclass satisfies it structurally.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None
    images: dict | None

    @property
    def state(self) -> np.ndarray:
        """Inference state vector: the proprioceptive input of build_state."""
        return self.joint_positions


class MuJoCoBimanualRobot(Robot):
    """Both SO-101 arms presented to Intel's runtime as one robot."""

    joint_names: ClassVar[list[str]] = [
        f"{arm}.{joint}"
        for arm in ("A", "B")
        for joint in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
    ]

    def __init__(self, scene: _SceneProtocol) -> None:
        self._scene = scene
        self._connects = 0

    def connect(self) -> None:
        """Idempotent connect; the scene resets only on the first one."""
        if self._connects == 0:
            self._scene.reset()
        self._connects += 1

    def disconnect(self) -> None:
        """Leave the robot stationary: both arms interpolating to home."""
        self._scene.hold_safe()

    def is_connected(self) -> bool:
        return bool(self._scene.ready)

    def get_observation(self) -> MuJoCoObservation:
        """Current proprioception; the state property feeds build_state."""
        return MuJoCoObservation(
            joint_positions=np.asarray(self._scene.qpos_12(), dtype=np.float64),
            timestamp=time.monotonic(),
            sensor_data=None,
            images=None,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Write position targets, then advance physics to the goal time."""
        self._scene.set_targets(np.asarray(action, dtype=np.float64))
        self._scene.step(goal_time)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """No hardware devices: this robot is the simulation."""
        return ()
