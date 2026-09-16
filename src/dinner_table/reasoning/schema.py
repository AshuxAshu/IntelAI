"""Task-graph contract between the VLM tier and the executor tier."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

SkillName = Literal[
    "open_drawer",
    "close_drawer",
    "pick",
    "place",
    "handoff",
    "hold",
    "pour",
    "home",
    "retract",
]
ObjectName = Literal[
    "plate",
    "mug",
    "bottle",
    "spoon_1",
    "spoon_2",
    "fork_1",
    "fork_2",
    "drawer_top",
]
PlacementTarget = Literal[
    "placemat_1",
    "placemat_2",
    "fork_setting",
    "spoon_setting",
    "drawer_tray",
    "hand_of_A",
    "hand_of_B",
    "mug",
]
ArmName = Literal["A", "B"]


class RelativeTarget(BaseModel):
    """Placement defined relative to a *perceived* object, e.g. "beside the plate".
    World-frame semantics - a person sitting at the table facing the cabinet
    (+Y): left = -X, right = +X, beside = the anchor's side away from the
    table center along X, offset 0.14 m."""

    model_config = ConfigDict(extra="forbid")

    relation: Literal["left_of", "right_of", "beside"]
    anchor: Literal[
        "plate",
        "mug",
        "bottle",
        "spoon_1",
        "spoon_2",
        "fork_1",
        "fork_2",
    ]


SKILL_REQUIRES_OBJECT = {
    "open_drawer": True,
    "close_drawer": True,
    "pick": True,
    "place": True,
    "handoff": True,
    "hold": True,
    "pour": True,
    "home": False,
    "retract": False,
}
SKILL_REQUIRES_TARGET = {"place": True, "handoff": True, "pour": True}


class Step(BaseModel):
    """One skill invocation. `extra="forbid"` makes hallucinated keys a hard error."""

    model_config = ConfigDict(extra="forbid")

    id: int  # 1-based, unique, strictly increasing
    skill: SkillName
    arm: ArmName
    object: ObjectName | None = None
    target: PlacementTarget | RelativeTarget | None = None
    amount: float | None = None  # pour only: target fill fraction in (0, 1]
    parallel_group: int | None = None  # steps sharing a group id run concurrently


class TaskGraph(BaseModel):
    """Validated plan emitted by the VLM and consumed by the executor."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    instruction: str
    steps: list[Step]

    @model_validator(mode="after")
    def _check(self) -> TaskGraph:
        if len(self.steps) == 0:
            raise ValueError("task graph must contain at least one step")
        ids = [s.id for s in self.steps]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ValueError("step ids must be unique and strictly increasing")
        held: dict[str, str] = {}  # object -> holding arm, for dependency checks
        for s in self.steps:
            if SKILL_REQUIRES_OBJECT[s.skill] and s.object is None:
                raise ValueError(f"step {s.id}: skill {s.skill} requires object")
            if s.skill in SKILL_REQUIRES_TARGET and s.target is None:
                raise ValueError(f"step {s.id}: skill {s.skill} requires target")
            if s.skill == "pick":
                held[s.object] = s.arm
            if s.skill == "place" and held.get(s.object) != s.arm:
                raise ValueError(
                    f"step {s.id}: place requires an earlier pick of {s.object} by arm {s.arm}"
                )
            if s.skill == "handoff":
                if not isinstance(s.target, str) or s.target not in ("hand_of_A", "hand_of_B"):
                    raise ValueError("handoff target must be hand_of_A or hand_of_B")
                if s.target == f"hand_of_{s.arm}":
                    raise ValueError("handoff source and destination arms must differ")
                if held.get(s.object) != s.arm:
                    raise ValueError("handoff requires holding the object first")
                held[s.object] = "A" if s.target == "hand_of_A" else "B"
        groups: dict[int, list[Step]] = {}
        for s in self.steps:
            if s.parallel_group is not None:
                groups.setdefault(s.parallel_group, []).append(s)
        for gid, members in groups.items():
            if len(members) > 2:
                raise ValueError(f"parallel group {gid} has more than 2 steps")
            if len({m.arm for m in members}) != len(members):
                raise ValueError(f"parallel group {gid} must use different arms")
        for s in self.steps:
            if s.skill == "pour":
                members = groups.get(s.parallel_group, []) if s.parallel_group is not None else []
                holds = [m for m in members if m.skill == "hold" and m.object is not None]
                if not any(h.object == "mug" for h in holds):
                    raise ValueError("pour must be parallel with a hold of the mug")
        return self


class ObjectStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visible: bool
    where: str  # short natural-language position, e.g. "on table near placemat_1"
    held_by: ArmName | None = None


class SceneSummary(BaseModel):
    """Everything the VLM sees about the world at a decision point. Built by the
    executor from PerceptionSnapshot + task state; never contains raw floats."""

    model_config = ConfigDict(extra="forbid")

    instruction: str
    objects: dict[str, ObjectStatus]
    drawer_open: bool
    completed_step_ids: list[int]
    current_step_id: int | None = None


class PreconditionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    reason: str


class VlmDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anomaly: Literal[
        "object_moved",
        "object_missing",
        "object_dropped",
        "drawer_jammed",
        "grasp_lost",
        "none",
        "unknown",
    ]
    explanation: str
    suggested_action: Literal["retry_skill", "replan", "abort"]
