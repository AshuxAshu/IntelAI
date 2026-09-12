"""Frozen geometry and naming constants for the dinner-table world.

World frame: origin at table center on the floor. +X points toward arm A's
mount, +Y toward the drawer cabinet at the back edge, +Z up. All lengths in
meters, all angles in radians.
"""

from __future__ import annotations

TABLE_TOP_HEIGHT = 0.36
TABLE_SIZE = (0.90, 0.50)
CABINET_Y = 0.62
DRAWER_TRAVEL = 0.24

ARM_MOUNTS = {
    "A": (0.45, 0.0, TABLE_TOP_HEIGHT),
    "B": (-0.45, 0.0, TABLE_TOP_HEIGHT),
}
ARM_ORIENTATIONS = {  # Euler XYZ quaternions are computed by the builder; these are yaw angles
    "A": -1.5707963267948966,  # arm A faces -X (toward table center)
    "B": 1.5707963267948966,  # arm B faces +X (toward table center)
}

SO101_JOINT_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_NAMES = tuple(
    f"{arm}.{j}" for arm in ("A", "B") for j in SO101_JOINT_SUFFIXES
)  # canonical 12-joint order; matches physicalai's SO101 naming with A./B. prefixes
GRIPPER_JOINTS = ("A.gripper", "B.gripper")
ACTION_DIM = 12  # one position target per JOINT_NAMES entry

CAMERA_NAMES = ("overhead", "wrist_A", "wrist_B", "demo_cam")
POLICY_CAMERA_NAMES = ("wrist_A", "wrist_B", "overhead")
OVERHEAD_RESOLUTION = (480, 640)  # (H, W)
POLICY_IMAGE_SIZE = (128, 128)
VLM_IMAGE_SIZE = (512, 512)

# Horizontal workspace zones: (x_min, x_max, y_min, y_max), applied at any height above the table.
SHARED_ZONE = (-0.15, 0.15, -0.15, 0.15)
ARM_A_ZONE = (-0.15, 0.45, -0.25, 0.25)
ARM_B_ZONE = (-0.45, 0.15, -0.25, 0.25)

PLACEMATS = {
    "placemat_1": (0.22, 0.10, TABLE_TOP_HEIGHT),
    "placemat_2": (-0.22, 0.10, TABLE_TOP_HEIGHT),
}
DRAWER_HANDLE = (0.0, CABINET_Y - 0.11, TABLE_TOP_HEIGHT + 0.19)
HOME_JOINTS = {
    "A": (0.0, -0.35, 1.20, -0.85, 0.0, 0.5),
    "B": (0.0, -0.35, 1.20, -0.85, 0.0, 0.5),
}  # per-arm safe pose: 5 joint angles (rad) + gripper aperture (0-1)

CONTROL_HZ = 25
PHYSICS_HZ = 500
