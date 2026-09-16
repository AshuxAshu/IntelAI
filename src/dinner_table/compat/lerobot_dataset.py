"""LeRobot 0.5.1 dataset adapter (rule R3: upstream drift lives here).

Two adaptations for the pinned API: ``task`` is a required per-frame key of
``add_frame`` (the plan sketch passes it to ``save_episode``, which 0.5.1 does
not accept), and video features validate against a declared (H, W, C) shape.
Call sites use this module's constructors and never touch lerobot directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dinner_table.contracts.geometry import CONTROL_HZ, JOINT_NAMES
from dinner_table.policies.conditioning import CONDITIONING_OBJECTS, SKILLS, STATE_DIM

FPS = CONTROL_HZ
IMAGE_SHAPE = (128, 128, 3)
ACTION_DIM_LEROBOT = 12

_STATE_NAMES = (
    [name for name in JOINT_NAMES if not name.endswith("gripper")]
    + ["A_gripper", "B_gripper", "arm_A", "arm_B"]
    + [f"skill_{skill}" for skill in SKILLS]
    + [f"object_{obj}" for obj in CONDITIONING_OBJECTS]
    + ["goal_x", "goal_y", "goal_z"]
)
assert len(_STATE_NAMES) == STATE_DIM

FEATURES: dict[str, dict] = {
    "observation.images.wrist_A": {
        "dtype": "video",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist_B": {
        "dtype": "video",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channels"],
    },
    "observation.images.overhead": {
        "dtype": "video",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": _STATE_NAMES},
    "action": {"dtype": "float32", "shape": (ACTION_DIM_LEROBOT,), "names": list(JOINT_NAMES)},
}


def create_dataset(repo_id: str, root: str | Path) -> LeRobotDataset:
    """Create a write-mode dinner dataset with the frozen feature layout."""
    return LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=FEATURES, root=root)


def _image(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.uint8)


def frame_dict(
    task: str,
    wrist_a: np.ndarray | None,
    wrist_b: np.ndarray | None,
    overhead: np.ndarray | None,
    state: np.ndarray,
    action: np.ndarray,
) -> dict:
    """Assemble one writer row; images are (128, 128, 3) uint8, state (35,).

    The ``task`` key rides on every row (the 0.5.1 adaptation); images are None
    only for headless logic tests with a capturing sink.
    """
    return {
        "task": task,
        "observation.images.wrist_A": _image(wrist_a),
        "observation.images.wrist_B": _image(wrist_b),
        "observation.images.overhead": _image(overhead),
        "observation.state": np.asarray(state, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
    }


def save_episode(dataset: LeRobotDataset, parallel_encoding: bool = True) -> None:
    """Flush the buffered episode to parquet plus per-camera mp4s."""
    dataset.save_episode(parallel_encoding=parallel_encoding)


def load_dataset(repo_id: str, root: str | Path) -> LeRobotDataset:
    """Open a written dataset from disk for reading."""
    return LeRobotDataset(repo_id, root=root)
