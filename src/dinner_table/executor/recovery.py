"""Recovery policy: the S6 edge-case matrix as data plus small handlers.

classify() maps a failed step + report + world to one of the six matrix
failure modes; handle() dispatches to the Executor's request hooks (the
executor owns the actual state transitions). Boundary-gate escalation goes
through handle_boundary - the world has diverged from the plan's assumptions,
which is the object_moved mode. Every handled failure is recorded in the
executor's telemetry `recovered` field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dinner_table.executor.preconditions import WorldState
from dinner_table.reasoning.schema import PreconditionReport, Step

if TYPE_CHECKING:
    from dinner_table.executor.graph_executor import Executor

# The S6 matrix, as data: failure mode -> recovery behavior.
RECOVERY_MATRIX = {
    "missed_grasp": "re-grasp: re-issue the pick conditioning with the tracker-refreshed goal",
    "dropped": "re-perceive + re-pick, then continue forward",
    "object_moved": "VLM replan with bounded depth",
    "drawer_jammed": "wiggle: mark the step for the gate's oscillation override and retry",
    "fumble": "both arms retract, then re-pick and restart the group",
    "spilled": "re-pour the remainder",
}

MAX_RECOVERY_MODES = tuple(RECOVERY_MATRIX)


def classify(step: Step, failure: PreconditionReport, world: WorldState | None) -> str:
    """Map a failed step to one of the six matrix failure modes.

    The raw failure reason is preserved in telemetry alongside the mode, so
    pragmatic mappings (a failed gripper release classifies as missed_grasp -
    both re-attempt the step mechanically) stay auditable.
    """
    poses = world.poses if world is not None else {}
    if step.skill == "pick":
        if step.object is not None and step.object in poses:
            return "missed_grasp"
        return "dropped"  # not visible: re-perceive + re-pick
    if step.skill == "place":
        if "still held" in failure.reason:
            return "missed_grasp"  # release miss: re-attempt the step
        return "dropped"  # misplaced or invisible: re-perceive + re-pick + re-place
    if step.skill in ("open_drawer", "close_drawer"):
        return "drawer_jammed"
    if step.skill == "handoff":
        return "fumble"
    if step.skill == "pour":
        return "spilled"
    return "object_moved"


def handle(step: Step, failure: PreconditionReport, ex: Executor) -> None:
    """Dispatch a postcondition failure through the S6 matrix."""
    mode = classify(step, failure, ex.world)
    ex.record_recovery(mode)
    if mode == "missed_grasp":
        ex.request_retry(step)
    elif mode == "dropped":
        ex.request_rollback_to_pick(step)
    elif mode == "object_moved":
        ex.request_replan(f"step {step.id}: {failure.reason}")
    elif mode == "drawer_jammed":
        ex.request_wiggle(step)
    elif mode == "fumble":
        ex.request_retract_pause(step)
    elif mode == "spilled":
        ex.request_repour(step)


def handle_boundary(step: Step, failure: PreconditionReport, ex: Executor) -> None:
    """Boundary-gate escalation: after one re-perceive attempt the world still
    contradicts the plan's assumptions - replan (object_moved mode)."""
    ex.record_recovery("object_moved")
    ex.request_replan(f"boundary step {step.id}: {failure.reason}")
