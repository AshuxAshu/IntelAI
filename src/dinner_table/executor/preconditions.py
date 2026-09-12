"""Pure precondition/postcondition predicates over perceived world state.

Perception enters as plain data (an ObjectPose3D map plus drawer state and
multiview verdicts), so every predicate is deterministic and unit-testable
without sim, models, or network. Zone claimability from the rule table is
enforced by the executor through workspace.ZoneClaims at execution time and
tested separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import HOME_JOINTS
from dinner_table.perception.interfaces import ObjectPose3D
from dinner_table.reasoning.schema import PreconditionReport, Step

PLACE_TOLERANCE_M = 0.02
HOME_TOLERANCE_RAD = 0.1


class PreconditionError(DinnerTableError):
    """Unknown skill passed to a predicate."""


@dataclass(frozen=True)
class WorldState:
    """Everything the predicates need about the perceived world.

    `poses` maps object names to perceived 3D poses; `multiview_ok` lists the
    objects that passed perception/depth_fusion.multiview_check; `joints`
    (arm -> 6-vector: 5 joint angles + gripper aperture) is needed only for
    home/retract postconditions.
    """

    poses: dict[str, ObjectPose3D]
    drawer_open: bool
    multiview_ok: frozenset[str] = frozenset()
    joints: dict[str, np.ndarray] | None = None


def _other(arm: str) -> str:
    return "B" if arm == "A" else "A"


def _pose(world: WorldState, name: str | None) -> ObjectPose3D | None:
    if name is None:
        return None
    return world.poses.get(name)


def check_precondition(step: Step, world: WorldState) -> PreconditionReport:
    """Precondition gate for one step (executor rule table)."""
    pose = _pose(world, step.object)
    if step.skill == "pick":
        if pose is None:
            return PreconditionReport(ok=False, reason=f"{step.object} not visible")
        if pose.held_by is not None:
            return PreconditionReport(
                ok=False, reason=f"{step.object} already held by arm {pose.held_by}"
            )
        if step.object not in world.multiview_ok:
            return PreconditionReport(ok=False, reason=f"{step.object} failed multiview check")
        return PreconditionReport(ok=True, reason="visible, unheld, multiview-verified")
    if step.skill == "place":
        if pose is None or pose.held_by != step.arm:
            return PreconditionReport(ok=False, reason=f"{step.object} not held by arm {step.arm}")
        return PreconditionReport(ok=True, reason="held by acting arm")
    if step.skill == "open_drawer":
        if world.drawer_open:
            return PreconditionReport(ok=False, reason="drawer already open")
        return PreconditionReport(ok=True, reason="drawer closed")
    if step.skill == "close_drawer":
        if not world.drawer_open:
            return PreconditionReport(ok=False, reason="drawer already closed")
        return PreconditionReport(ok=True, reason="drawer open")
    if step.skill in ("handoff", "hold"):
        if pose is None or pose.held_by != step.arm:
            return PreconditionReport(ok=False, reason=f"{step.object} not held by arm {step.arm}")
        return PreconditionReport(ok=True, reason="held by acting arm")
    if step.skill == "pour":
        if pose is None or pose.held_by != step.arm:
            return PreconditionReport(ok=False, reason=f"{step.object} not held by arm {step.arm}")
        mug = world.poses.get("mug")
        if mug is None or mug.held_by != _other(step.arm):
            return PreconditionReport(ok=False, reason="mug not held by the other arm")
        return PreconditionReport(ok=True, reason="bottle and mug held by opposite arms")
    if step.skill in ("home", "retract"):
        return PreconditionReport(ok=True, reason="no precondition")
    raise PreconditionError(f"unknown skill {step.skill!r}")


def check_postcondition(
    step: Step, world: WorldState, goal: np.ndarray | None = None
) -> PreconditionReport:
    """Postcondition gate for one step (executor rule table).

    `goal` is the step's resolved world-frame goal; required only for place.
    """
    pose = _pose(world, step.object)
    if step.skill == "pick":
        if pose is None or pose.held_by != step.arm:
            return PreconditionReport(ok=False, reason=f"{step.object} not held by arm {step.arm}")
        return PreconditionReport(ok=True, reason="object grasped")
    if step.skill == "place":
        if goal is None:
            return PreconditionReport(ok=False, reason="place postcondition requires the goal")
        if pose is None:
            return PreconditionReport(ok=False, reason=f"{step.object} not visible")
        if pose.held_by is not None:
            return PreconditionReport(ok=False, reason=f"{step.object} still held")
        distance = float(np.linalg.norm(pose.position - goal))
        if distance > PLACE_TOLERANCE_M:
            return PreconditionReport(
                ok=False, reason=f"placed {distance:.3f} m from goal (tolerance 2 cm)"
            )
        return PreconditionReport(ok=True, reason="placed within tolerance")
    if step.skill == "open_drawer":
        if not world.drawer_open:
            return PreconditionReport(ok=False, reason="drawer not open")
        return PreconditionReport(ok=True, reason="drawer open")
    if step.skill == "close_drawer":
        if world.drawer_open:
            return PreconditionReport(ok=False, reason="drawer still open")
        return PreconditionReport(ok=True, reason="drawer closed")
    if step.skill == "handoff":
        if pose is None or pose.held_by != _other(step.arm):
            return PreconditionReport(
                ok=False, reason=f"{step.object} not held by arm {_other(step.arm)}"
            )
        return PreconditionReport(ok=True, reason="object transferred")
    if step.skill == "hold":
        return PreconditionReport(ok=True, reason="hold ends with its parallel group")
    if step.skill == "pour":
        # Water level is not camera-observable: verification is delegated to the
        # VLM boundary check on the following step (honest spec).
        return PreconditionReport(ok=True, reason="delegated to VLM boundary check")
    if step.skill in ("home", "retract"):
        if world.joints is None or step.arm not in world.joints:
            return PreconditionReport(ok=False, reason=f"arm {step.arm} joints unavailable")
        current = np.asarray(world.joints[step.arm], dtype=np.float64)
        if current.shape != (6,):
            return PreconditionReport(ok=False, reason="joints must be a 6-vector")
        deviation = float(np.abs(current[:5] - np.asarray(HOME_JOINTS[step.arm])[:5]).max())
        if deviation > HOME_TOLERANCE_RAD:
            return PreconditionReport(
                ok=False, reason=f"arm {deviation:.3f} rad from home (tolerance 0.1)"
            )
        return PreconditionReport(ok=True, reason="arm at home pose")
    raise PreconditionError(f"unknown skill {step.skill!r}")
