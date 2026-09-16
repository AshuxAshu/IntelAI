"""Privileged task-graph runner: the oracle that turns a plan into motion.

``run_graph`` executes a validated ``TaskGraph`` against a scene with ground
truth grounding, running steps in order and advancing steps that share a
``parallel_group`` as paired coroutines. Every tick is logged into an
``EpisodeLog`` whose schema the data engine, the taxonomy and the VQA factory
consume; images are never logged, because the data engine re-renders them
deterministically from ``(seed, action_history)``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

import numpy as np

from dinner_table.policies.conditioning import goal_for_skill
from dinner_table.reasoning.schema import RelativeTarget, Step, TaskGraph
from dinner_table.teacher.bimanual import Handoff, Hold, ParallelGroup, Pour
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.skills import (
    CloseDrawer,
    Home,
    OpenDrawer,
    Pick,
    Place,
    Retract,
    Skill,
)

logger = logging.getLogger(__name__)

STEP_TIMEOUT_S = 150.0
EPISODE_TIMEOUT_S = 900.0
MAX_ATTEMPTS = 3  # the first try plus the two retries the plan allows
OUTCOMES = ("success", "failed")


@dataclass
class StepRecord:
    """One executed graph step and how it ended."""

    step_id: int
    skill: str
    arm: str
    outcome: str
    attempts: int
    phase_at_failure: str | None = None
    failure_cause: str | None = None


@dataclass
class EpisodeLog:
    """Frozen episode record: what was asked, what ran, and every tick."""

    episode_id: str
    seed: int
    dr_profile: str
    instruction: str
    task_id: str
    steps: list[StepRecord] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    success: bool = False
    water_fraction: float | None = None

    def to_dict(self) -> dict:
        """Return the JSON-serializable form of the log."""
        return asdict(self)

    def action_history(self) -> np.ndarray:
        """Return the (T, 12) action sequence that reproduces the episode."""
        if len(self.frames) == 0:
            return np.zeros((0, 12), dtype=np.float64)
        return np.array([f["action"] for f in self.frames], dtype=np.float64)


class TeacherPolicyError(Exception):
    """Raised when a task graph cannot be turned into teacher skills."""


def _target_for_skill(step: Step):
    """Return the Place/Pour target in the form the skill expects."""
    if step.target is None:
        return None
    if isinstance(step.target, RelativeTarget):
        return {"anchor": step.target.anchor, "relation": step.target.relation}
    return step.target


def build_skill(step: Step) -> Skill:
    """Instantiate the teacher skill a graph step names."""
    if step.skill == "open_drawer":
        return OpenDrawer(step.arm)
    if step.skill == "close_drawer":
        return CloseDrawer(step.arm)
    if step.skill == "pick":
        return Pick(step.arm, step.object)
    if step.skill == "place":
        return Place(step.arm, step.object, _target_for_skill(step))
    if step.skill == "hold":
        return Hold(step.arm, step.object)
    if step.skill == "handoff":
        to_arm = "A" if step.target == "hand_of_A" else "B"
        return Handoff(step.object, step.arm, to_arm)
    if step.skill == "pour":
        amount = step.amount if step.amount is not None else 0.6
        return Pour(step.arm, step.object, step.target, amount)
    if step.skill == "home":
        return Home(step.arm)
    if step.skill == "retract":
        return Retract(step.arm)
    raise TeacherPolicyError(f"no teacher skill for graph skill {step.skill}")


def _goal_xyz(ctx: TeacherContext, step: Step) -> np.ndarray:
    """Goal anchor (3,) for the conditioning vector, from ground truth.

    Computed exactly as the runtime computes it from perception, so the
    student sees the same conditioning in training and at deployment.
    """
    object_position = None
    anchor_position = None
    if step.skill == "pour":
        object_position = ctx.object("mug")[0]
    elif step.object is not None and step.object != "drawer_top":
        object_position = ctx.object(step.object)[0]
    target = step.target
    if isinstance(target, RelativeTarget):
        anchor_position = ctx.object(target.anchor)[0]
    return goal_for_skill(step.skill, step.object, object_position, target, anchor_position)


def _record_frame(ctx: TeacherContext, log: EpisodeLog, action: np.ndarray,
                  step: Step, skill: Skill) -> None:
    joints = ctx.scene.qpos_12()
    log.frames.append({
        "tick": len(log.frames),
        "joints": [float(v) for v in joints],
        "action": [float(v) for v in action],
        "skill": step.skill,
        "phase": skill.phase,
        "goal_xyz": [float(v) for v in _goal_xyz(ctx, step)],
    })


def _recover(ctx: TeacherContext, step: Step, skill: Skill) -> Iterator[np.ndarray]:
    """Restore a step's precondition so the retry starts where the first try did.

    A skill that still holds its object resumes from wherever the carry
    stopped. A skill that needs the object in hand and has lost it — a place
    whose release verified badly, a pour that fumbled — re-picks it from
    wherever it came to rest; anything else simply parks the arm.
    """
    arm = skill.arm
    held = ctx.carrying.get(arm)
    if held == step.object:
        return
    if held is not None:
        return  # the arm holds something else: the step cannot be retried cleanly
    if step.skill in ("place", "pour", "hold", "handoff") and step.object is not None:
        yield from Home(arm).run(ctx)
        yield from Pick(arm, step.object).run(ctx)
        return
    yield from Home(arm).run(ctx)


def _run_members(ctx: TeacherContext, members: list[tuple[Step, Skill]]) -> Iterator[np.ndarray]:
    """Run one step, or a parallel group, as a single action stream."""
    if len(members) == 1:
        yield from members[0][1].run(ctx)
        return
    yield from ParallelGroup([skill for _, skill in members]).run(ctx)


def _ordered_members(unit: list[Step]) -> list[Step]:
    """Put the group's driving step first; a hold only accompanies it."""
    return sorted(unit, key=lambda step: step.skill == "hold")


def _grouped_steps(graph: TaskGraph) -> list[list[Step]]:
    """Order the graph's steps, collapsing each parallel group into one unit."""
    units: list[list[Step]] = []
    seen_groups: dict[int, list[Step]] = {}
    for step in graph.steps:
        if step.parallel_group is None:
            units.append([step])
            continue
        if step.parallel_group not in seen_groups:
            seen_groups[step.parallel_group] = []
            units.append(seen_groups[step.parallel_group])
        seen_groups[step.parallel_group].append(step)
    return units


def run_graph(scene, graph: TaskGraph, seed: int, dr_profile: str | None = None) -> EpisodeLog:
    """Execute a validated task graph with privileged grounding.

    Returns an ``EpisodeLog``; a step that fails all its attempts ends the
    episode with ``success=False`` and the failure attributed to that step.
    """
    ctx = TeacherContext(scene)
    log = EpisodeLog(
        episode_id=f"{graph.task_id}_{seed:06d}",
        seed=int(seed),
        dr_profile=dr_profile if dr_profile is not None else str(scene.dr_profile_name),
        instruction=graph.instruction,
        task_id=graph.task_id,
    )
    for unit in _grouped_steps(graph):
        outcome = "failed"
        attempts = 0
        phase_at_failure: str | None = None
        failure_cause: str | None = None
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            members = [(step, build_skill(step)) for step in _ordered_members(unit)]
            ctx.begin(STEP_TIMEOUT_S)
            try:
                for action in _run_members(ctx, members):
                    ctx.live.note_skill(members[0][1])
                    ctx.step(action)
                    _record_frame(ctx, log, action, members[0][0], members[0][1])
                outcome = "success"
                phase_at_failure = None
                failure_cause = None
                break
            except SkillFailed as exc:
                phase_at_failure = exc.phase
                failure_cause = exc.cause
                logger.info(
                    "step %s (%s) attempt %d failed at %s: %s",
                    unit[0].id, unit[0].skill, attempts, exc.phase, exc.cause,
                )
                if attempts >= MAX_ATTEMPTS:
                    break
                try:
                    ctx.begin(STEP_TIMEOUT_S)
                    for action in _recover(ctx, members[0][0], members[0][1]):
                        ctx.step(action)
                        _record_frame(ctx, log, action, members[0][0], members[0][1])
                except SkillFailed:
                    break
        for step in unit:
            log.steps.append(StepRecord(
                step_id=step.id,
                skill=step.skill,
                arm=step.arm,
                outcome=outcome,
                attempts=attempts,
                phase_at_failure=phase_at_failure,
                failure_cause=failure_cause,
            ))
        if outcome != "success":
            log.success = False
            log.water_fraction = float(scene.fill_fraction("mug"))
            return log
    log.success = True
    log.water_fraction = float(scene.fill_fraction("mug"))
    return log


def _step(step_id: int, skill: str, arm: str, **kwargs) -> Step:
    return Step(id=step_id, skill=skill, arm=arm, **kwargs)


def _canonical_graph() -> TaskGraph:
    """The problem statement's own command.

    Arm A cannot reach the bottle's side of the table, so the bottle travels
    to it through the relay handoff — the reassignment the command itself
    calls for.
    """
    return TaskGraph(
        task_id="canonical",
        instruction=(
            "Open the top drawer, pick up the plate with arm A, place it on the table, "
            "pick up the mug with arm B, pour water into the mug with arm A."
        ),
        steps=[
            _step(1, "open_drawer", "A", object="drawer_top"),
            _step(2, "pick", "A", object="plate"),
            _step(3, "place", "A", object="plate", target="placemat_1"),
            _step(4, "pick", "B", object="bottle"),
            _step(5, "handoff", "B", object="bottle", target="hand_of_A"),
            _step(6, "pick", "B", object="mug"),
            _step(7, "hold", "B", object="mug", parallel_group=1),
            _step(8, "pour", "A", object="bottle", target="mug", amount=0.6,
                  parallel_group=1),
            _step(9, "place", "A", object="bottle", target="placemat_2"),
            _step(10, "place", "B", object="mug", target="placemat_2"),
        ],
    )


def _full_graph() -> TaskGraph:
    """The full demonstration task: drawer, cutlery, settings, relay, pour."""
    return TaskGraph(
        task_id="full_dinner",
        instruction="Set the dinner table and pour a glass of water.",
        steps=[
            _step(1, "open_drawer", "A", object="drawer_top"),
            _step(2, "pick", "A", object="fork_1"),
            _step(3, "place", "A", object="fork_1", target="fork_setting"),
            _step(4, "pick", "A", object="spoon_1"),
            _step(5, "place", "A", object="spoon_1", target="spoon_setting"),
            _step(6, "pick", "A", object="plate"),
            _step(7, "place", "A", object="plate", target="placemat_1"),
            _step(8, "pick", "B", object="bottle"),
            _step(9, "handoff", "B", object="bottle", target="hand_of_A"),
            _step(10, "pick", "B", object="mug"),
            _step(11, "hold", "B", object="mug", parallel_group=1),
            _step(12, "pour", "A", object="bottle", target="mug", amount=0.6,
                  parallel_group=1),
            _step(13, "place", "A", object="bottle", target="placemat_2"),
            _step(14, "place", "B", object="mug", target="placemat_2"),
            _step(15, "close_drawer", "A", object="drawer_top"),
            _step(16, "home", "A"),
            _step(17, "home", "B"),
        ],
    )


def _pick_place_graph() -> TaskGraph:
    return TaskGraph(
        task_id="plate_setting",
        instruction="Put the plate on the left place setting.",
        steps=[
            _step(1, "pick", "A", object="plate"),
            _step(2, "place", "A", object="plate", target="placemat_1"),
            _step(3, "retract", "A"),
        ],
    )


def _mug_setting_graph() -> TaskGraph:
    return TaskGraph(
        task_id="mug_setting",
        instruction="Pick up the mug with arm B and set it down at its place setting.",
        steps=[
            _step(1, "pick", "B", object="mug"),
            _step(2, "place", "B", object="mug", target="placemat_2"),
            _step(3, "retract", "B"),
        ],
    )


def _cutlery_graph() -> TaskGraph:
    return TaskGraph(
        task_id="cutlery_setting",
        instruction="Open the drawer and lay a fork and a spoon beside the plate.",
        steps=[
            _step(1, "open_drawer", "A", object="drawer_top"),
            _step(2, "pick", "A", object="fork_1"),
            _step(3, "place", "A", object="fork_1", target="fork_setting"),
            _step(4, "pick", "A", object="spoon_1"),
            _step(5, "place", "A", object="spoon_1", target="spoon_setting"),
            _step(6, "close_drawer", "A", object="drawer_top"),
        ],
    )


def _handoff_graph() -> TaskGraph:
    return TaskGraph(
        task_id="bottle_handoff",
        instruction="Pick up the bottle with arm B and pass it to arm A.",
        steps=[
            _step(1, "pick", "B", object="bottle"),
            _step(2, "handoff", "B", object="bottle", target="hand_of_A"),
            _step(3, "place", "A", object="bottle", target="placemat_1"),
        ],
    )


def _pour_graph() -> TaskGraph:
    return TaskGraph(
        task_id="pour_water",
        instruction="Hold the mug with arm B and pour water into it with arm A.",
        steps=[
            _step(1, "pick", "B", object="bottle"),
            _step(2, "handoff", "B", object="bottle", target="hand_of_A"),
            _step(3, "pick", "B", object="mug"),
            _step(4, "hold", "B", object="mug", parallel_group=1),
            _step(5, "pour", "A", object="bottle", target="mug", amount=0.6,
                  parallel_group=1),
            _step(6, "place", "A", object="bottle", target="placemat_1"),
            _step(7, "place", "B", object="mug", target="placemat_2"),
        ],
    )


def _relational_graph() -> TaskGraph:
    return TaskGraph(
        task_id="relational_fork",
        instruction="Open the drawer and put a fork beside the plate.",
        steps=[
            _step(1, "open_drawer", "A", object="drawer_top"),
            _step(2, "pick", "A", object="fork_1"),
            _step(3, "place", "A", object="fork_1",
                  target=RelativeTarget(relation="beside", anchor="plate")),
        ],
    )


CANONICAL_GRAPHS: dict[str, TaskGraph] = {
    "canonical": _canonical_graph(),
    "full_dinner": _full_graph(),
    "plate_setting": _pick_place_graph(),
    "mug_setting": _mug_setting_graph(),
    "cutlery_setting": _cutlery_graph(),
    "bottle_handoff": _handoff_graph(),
    "pour_water": _pour_graph(),
    "relational_fork": _relational_graph(),
}
