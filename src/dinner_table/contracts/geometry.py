"""Frozen geometry and naming constants for the dinner-table world.

World frame: origin at table center on the floor. +X points toward arm B's
mount, +Y away from the operator (both arms are front-mounted and face +Y),
+Z up. All lengths in meters, all angles in radians.

Amendment 1 (see docs/PLAN_AMENDMENTS.md): the arms are the official
Menagerie SO-101 MJCF, mounted side by side on the operator-facing front
edge (layout adopted from the reference solution — the original opposing-end
mounts at x = +/-0.45 exceeded the real SO-101 reach and made the shared
zone and the drawer unreachable). Arm A takes the reference's left-arm role
(plate / drawer / cutlery side), arm B the right-arm role (mug side).
"""

from __future__ import annotations

TABLE_TOP_HEIGHT = 0.36
TABLE_SIZE = (0.96, 0.78)
CABINET_X = -0.24  # tabletop cutlery caddy; drawer slides open toward the arms (-Y)
CABINET_Y = 0.11
DRAWER_TRAVEL = 0.12

ARM_MOUNTS = {
    "A": (-0.20, -0.305, TABLE_TOP_HEIGHT + 0.018),
    "B": (0.20, -0.305, TABLE_TOP_HEIGHT + 0.018),
}
ARM_ORIENTATIONS = {  # Euler XYZ quaternions are computed by the arm merge; these are yaw angles
    "A": 1.5707963267948966,  # arm A faces +Y (into the table)
    "B": 1.5707963267948966,  # arm B faces +Y (into the table)
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
# Each arm's zone is its half of the table within comfortable reach of its
# front-edge mount; the zones overlap in a center strip (classified "shared"),
# and the shared zone proper is the front-center lens both arms reach at
# working height (measured: 25/25 dual-reach at table+0.05, union at table+0.10).
SHARED_ZONE = (-0.06, 0.06, -0.30, -0.16)
ARM_A_ZONE = (-0.48, 0.02, -0.39, 0.05)
ARM_B_ZONE = (-0.02, 0.48, -0.39, 0.05)

PLACEMATS = {
    "placemat_1": (-0.06, -0.095, TABLE_TOP_HEIGHT),  # plate setting (arm A side)
    "placemat_2": (0.265, -0.065, TABLE_TOP_HEIGHT),  # mug setting (arm B side)
}
DRAWER_HANDLE = (CABINET_X, CABINET_Y - 0.18, TABLE_TOP_HEIGHT + 0.0575)
HOME_JOINTS = {
    "A": (0.0, -0.70, 0.80, 0.20, 0.0, 0.43),
    "B": (0.0, -0.70, 0.80, 0.20, 0.0, 0.43),
}  # per-arm safe pose: 5 joint angles (rad) + gripper aperture (0-1)

CONTROL_HZ = 25
PHYSICS_HZ = 500
