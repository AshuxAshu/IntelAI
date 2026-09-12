"""Task-graph scheduling: parallel grouping, arm reassignment, goal checks.

group_steps turns a validated TaskGraph into ordered execution groups and
applies holder-chain-aware transformations to singleton steps (parallel-group
members are never reassigned - they are coordinated by design):

1. ARM REASSIGNMENT: a singleton pick whose object lies exclusively
   in the other arm's zone swaps to that arm; the swap then propagates through
   the hold chain - handoff sources and places follow the object's holder.
   Chain-following is equivalent to the spec's object-position rule for
   handoffs and places, and guarantees a pick/place pair can never end up on
   different arms.
2. PLACEMENT-GOAL FEASIBILITY: a place step's resolved goal (named anchor, or
   a RelativeTarget resolved from the anchor's perceived pose plus the
   conditioning offsets) is checked against the acting arm's zone. An
   exclusive-zone mismatch that reassignment cannot fix without breaking the
   hold chain is logged as a warning line for the HUD and the VLM replan path
   - never silently swapped.

Steps whose object or anchor is not yet visible are left unchanged for the
executor's re-perceive path.
"""

from __future__ import annotations

import numpy as np

from dinner_table.executor.workspace import zone_of
from dinner_table.perception.interfaces import ObjectPose3D
from dinner_table.policies.conditioning import goal_for_skill
from dinner_table.reasoning.schema import Step, TaskGraph


def _other(arm: str) -> str:
    return "B" if arm == "A" else "A"


def _object_position(step: Step, poses: dict[str, ObjectPose3D]) -> np.ndarray | None:
    """Perceived position of a step's object; None when not visible."""
    if step.object is None:
        return None
    pose = poses.get(step.object)
    return None if pose is None else pose.position


def _place_goal(step: Step, poses: dict[str, ObjectPose3D]) -> np.ndarray | None:
    """Resolved world-frame goal of a place step (named anchor, or a
    RelativeTarget via the anchor's perceived pose); None when the anchor is
    not visible. Mirrors conditioning.goal_for_skill exactly."""
    if isinstance(step.target, str):
        return goal_for_skill("place", step.object, None, step.target)
    anchor = poses.get(step.target.anchor)
    if anchor is None:
        return None
    return goal_for_skill("place", step.object, None, step.target, anchor_position=anchor.position)


def group_steps(
    graph: TaskGraph, poses: dict[str, ObjectPose3D]
) -> tuple[list[list[Step]], list[str]]:
    """Ordered execution groups plus HUD log lines.

    Parallel groups become single units placed at their first member's
    position; every other step is a singleton. Singleton picks may be
    reassigned to the other arm when their object lies exclusively in that
    arm's zone; handoffs and places then follow the resulting hold chain.
    """
    grouped: list[list[Step]] = []
    group_index: dict[int, int] = {}
    for step in graph.steps:
        if step.parallel_group is None:
            grouped.append([step])
        elif step.parallel_group in group_index:
            grouped[group_index[step.parallel_group]].append(step)
        else:
            group_index[step.parallel_group] = len(grouped)
            grouped.append([step])

    log: list[str] = []
    holder: dict[str, str] = {}

    def replace(group_i: int, step_i: int, update: dict[str, str]) -> Step:
        updated = grouped[group_i][step_i].model_copy(update=update)
        grouped[group_i][step_i] = updated
        return updated

    for group_i, group in enumerate(grouped):
        for step_i, step in enumerate(group):
            reassignable = step.parallel_group is None
            if step.skill == "pick":
                if reassignable:
                    position = _object_position(step, poses)
                    if position is not None and zone_of(position) == _other(step.arm):
                        step = replace(group_i, step_i, {"arm": _other(step.arm)})
                        log.append(
                            f"reassigned step {step.id} to arm {step.arm}: goal in {step.arm}"
                        )
                holder[step.object] = step.arm
            elif step.skill == "handoff":
                if reassignable:
                    source = holder.get(step.object, step.arm)
                    if source != step.arm:
                        step = replace(
                            group_i,
                            step_i,
                            {"arm": source, "target": f"hand_of_{_other(source)}"},
                        )
                        log.append(
                            f"reassigned step {step.id} to arm {step.arm}: follows object holder"
                        )
                holder[step.object] = _other(step.arm)
            elif step.skill == "place":
                if reassignable:
                    acting = holder.get(step.object, step.arm)
                    if acting != step.arm:
                        step = replace(group_i, step_i, {"arm": acting})
                        log.append(
                            f"reassigned step {step.id} to arm {acting}: follows object holder"
                        )
                    goal = _place_goal(step, poses)
                    if goal is not None:
                        goal_zone = zone_of(goal)
                        if goal_zone in ("A", "B") and goal_zone != step.arm:
                            log.append(
                                f"step {step.id}: placement goal in {goal_zone} "
                                f"unreachable by arm {step.arm} - left for recovery"
                            )
                holder.pop(step.object, None)  # released after placement
    return grouped, log
