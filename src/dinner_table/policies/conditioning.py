"""Skill conditioning packed into observation.state. Native ACT (physicalai)
consumes state + images only, so conditioning is structural. This module is the
single source of truth for the state-vector layout; the teacher (data
generation) and the runtime (deployment) build it identically."""

from __future__ import annotations

import numpy as np

from dinner_table.contracts.geometry import DRAWER_HANDLE, JOINT_NAMES, PLACEMATS
from dinner_table.reasoning.schema import RelativeTarget

SKILLS = (
    "open_drawer",
    "close_drawer",
    "pick",
    "place",
    "handoff",
    "hold",
    "pour",
    "home",
    "retract",
)
OBJECTS = (
    "plate",
    "mug",
    "bottle",
    "spoon_1",
    "spoon_2",
    "fork_1",
    "fork_2",
    "drawer_top",
)
NO_OBJECT = "none"
CONDITIONING_OBJECTS = OBJECTS + (NO_OBJECT,)  # 9 entries

STATE_DIM = 35
# Layout (float32), exactly:
#   [0:10]  the 10 non-gripper joint positions (rad), gathered from JOINT_NAMES
#           order (A's 5 arm joints, then B's 5 arm joints)
#   [10]    A.gripper aperture, normalized 0..1
#   [11]    B.gripper aperture, normalized 0..1
#   [12]    active-arm one-hot: A
#   [13]    active-arm one-hot: B
#   [14:23] skill one-hot over SKILLS (9 slots)
#   [23:32] object one-hot over CONDITIONING_OBJECTS (9 slots; index 31 = "none")
#   [32:35] goal xyz (m, world frame); zeros for home/retract


def skill_index(skill: str) -> int:
    return SKILLS.index(skill)


def object_index(obj: str | None) -> int:
    return CONDITIONING_OBJECTS.index(obj if obj is not None else NO_OBJECT)


def build_state(
    joints: np.ndarray,  # (12,) full joint vector, JOINT_NAMES order
    skill: str,
    arm: str,
    object_name: str | None,
    goal_xyz: np.ndarray,  # (3,) meters, world frame
) -> np.ndarray:
    """Assemble the (35,) float32 observation.state vector."""
    if joints.shape != (12,):
        raise ValueError(f"joints must have shape (12,), got {joints.shape}")
    state = np.zeros(STATE_DIM, dtype=np.float32)
    non_gripper = [i for i, n in enumerate(JOINT_NAMES) if not n.endswith("gripper")]
    state[0:10] = joints[non_gripper]
    state[10] = joints[JOINT_NAMES.index("A.gripper")]
    state[11] = joints[JOINT_NAMES.index("B.gripper")]
    state[12 + (0 if arm == "A" else 1)] = 1.0
    state[14 + skill_index(skill)] = 1.0
    state[23 + object_index(object_name)] = 1.0
    state[32:35] = goal_xyz.astype(np.float32)
    return state


def goal_for_skill(
    skill: str,
    object_name: str | None,
    object_position: np.ndarray | None,  # (3,) perceived/GT object pose
    target: str | RelativeTarget | None,
    anchor_position: np.ndarray | None = None,  # (3,) perceived pose of a RelativeTarget's anchor
) -> np.ndarray:
    """Skill-dependent goal anchor (3,) - MUST be computed identically by the
    teacher during data generation and by the runtime at deployment. Rules:
      pick / hold / handoff : the object's current position
      place                 : the placement target position (named anchor, or
                               RelativeTarget = anchor pose + offset: left = -X,
                               right = +X (person facing +Y), beside = away
                               from table center along X; offset 0.14 m)
      pour                  : the mug's current position (bottle is the object)
      open_drawer / close_drawer : the drawer handle position
      home / retract        : zeros
    """
    if skill in ("pick", "hold", "handoff"):
        if object_position is None:
            raise ValueError(f"{skill} requires object_position")
        return np.asarray(object_position, dtype=np.float64)
    if skill == "place":
        if target is None:
            raise ValueError("place requires target")
        if not isinstance(target, str):  # RelativeTarget
            if anchor_position is None:
                raise ValueError("relative target requires anchor_position")
            if target.relation == "beside":
                side = 1.0 if anchor_position[0] >= 0.0 else -1.0
                return np.asarray(anchor_position, dtype=np.float64) + np.array(
                    [0.14 * side, 0.0, 0.0]
                )
            offset = {"left_of": (-0.14, 0.0, 0.0), "right_of": (0.14, 0.0, 0.0)}[target.relation]
            return np.asarray(anchor_position, dtype=np.float64) + np.asarray(offset)
        if target in PLACEMATS:
            return np.asarray(PLACEMATS[target], dtype=np.float64)
        if target == "drawer_tray":
            return np.asarray((0.0, 0.50, 0.42), dtype=np.float64)
        raise ValueError(f"unsupported place target {target}")
    if skill == "pour":
        if object_position is None:
            raise ValueError("pour requires the mug's position as object_position")
        return np.asarray(object_position, dtype=np.float64)
    if skill in ("open_drawer", "close_drawer"):
        return np.asarray(DRAWER_HANDLE, dtype=np.float64)
    return np.zeros(3)
