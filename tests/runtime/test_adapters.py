"""Runtime adapter tests: the simulation as a physicalai Robot and Cameras."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dinner_table.contracts.geometry import HOME_JOINTS, JOINT_NAMES, POLICY_CAMERA_NAMES
from dinner_table.runtime.mujoco_camera import MuJoCoCamera
from dinner_table.runtime.mujoco_robot import MuJoCoBimanualRobot

TICK_DURATION_S = 5.0
TICK_FPS = 25.0
EXPECTED_TICKS = 125
TICK_TOLERANCE = 2


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = [
    pytest.mark.openvino,
    pytest.mark.skipif(not _has("physicalai"), reason="physicalai not installed"),
]


@pytest.fixture(scope="module")
def scene():
    """One shared deterministic scene for all adapter tests."""
    from dinner_table.scene.builder import Scene

    return Scene(seed=42, dr_profile="default")


class _HomeChunkSource:
    """Scripted inference stand-in of B's own: HOME-action chunks every call."""

    def __init__(self) -> None:
        self._home = np.concatenate([HOME_JOINTS["A"], HOME_JOINTS["B"]])

    def predict_action_chunk(self, observation: dict) -> np.ndarray:
        """A (25, 12) chunk holding both arms at the home pose."""
        return np.tile(self._home, (25, 1)).astype(np.float32)


class TestRobotProtocol:
    def test_robot_protocol(self, scene):
        from physicalai.robot import verify_robot

        robot = MuJoCoBimanualRobot(scene)
        verify_robot(robot)
        assert robot.joint_names == list(JOINT_NAMES)
        assert robot.device_ids == ()

    def test_observation_state_is_proprioception(self, scene):
        robot = MuJoCoBimanualRobot(scene)
        robot.connect()
        observation = robot.get_observation()
        assert observation.joint_positions.shape == (12,)
        assert observation.sensor_data is None
        assert observation.images is None
        np.testing.assert_allclose(observation.state, observation.joint_positions)


class TestCameraProtocol:
    def test_camera_protocol(self, scene):
        from physicalai.capture import Camera

        for name in POLICY_CAMERA_NAMES:
            camera = MuJoCoCamera(scene, name)
            assert isinstance(camera, Camera)
            camera.connect()
            frames = [camera.read_latest() for _ in range(3)]
            for frame in frames:
                assert frame.data.ndim == 3
                assert frame.data.shape[2] == 3
                assert frame.data.dtype == np.uint8
            if name.startswith("wrist"):
                assert frames[0].data.shape[:2] == (128, 128)
            else:
                assert frames[0].data.shape[:2] == (480, 640)
            sequences = [frame.sequence for frame in frames]
            assert sequences == sorted(set(sequences)) and len(set(sequences)) == 3
            timestamps = [frame.timestamp for frame in frames]
            assert timestamps == sorted(timestamps)
            assert camera.device_id == f"mujoco:{name}"

    def test_read_matches_read_latest(self, scene):
        camera = MuJoCoCamera(scene, "overhead")
        camera.connect()
        first = camera.read()
        second = camera.read_latest()
        assert first.sequence == 1
        assert second.sequence == 2


class TestTickIntegration:
    def test_five_second_loop_drives_real_physics(self, scene):
        from physicalai.runtime import PolicyRuntime, SyncExecution

        robot = MuJoCoBimanualRobot(scene)
        cameras = {name: MuJoCoCamera(scene, name) for name in POLICY_CAMERA_NAMES}
        source = _HomeChunkSource()
        runtime = PolicyRuntime(
            robot=robot,
            model=source,
            execution=SyncExecution(),
            fps=TICK_FPS,
            cameras=cameras,
        )
        with runtime:
            stats = runtime.run(duration_s=TICK_DURATION_S)
        # NOTE: the pinned RunStats carries no last_run_reason field; a normal
        # return with the expected step count is the duration-completion proof.
        assert abs(stats.steps - EXPECTED_TICKS) <= TICK_TOLERANCE
        assert stats.total_pops > 0
        joints = robot.get_observation().joint_positions
        home = np.concatenate([HOME_JOINTS["A"], HOME_JOINTS["B"]])
        np.testing.assert_allclose(joints, home, atol=0.05)


class TestKeyNaming:
    def test_camera_keys_match_policy_names(self, scene):
        cameras = {name: MuJoCoCamera(scene, name) for name in POLICY_CAMERA_NAMES}
        assert set(cameras) == set(POLICY_CAMERA_NAMES)
        assert set(cameras) == {"wrist_A", "wrist_B", "overhead"}
