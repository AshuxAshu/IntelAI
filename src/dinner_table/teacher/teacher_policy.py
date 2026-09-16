"""Privileged teacher oracle: run a validated task graph and log the episode.

``run_graph`` is the demonstration source for the data engine. It grounds every
step in ground-truth simulator state (never perception), resolves relative
placement targets through the same ``goal_for_skill`` the deployed runtime
conditions on, advances parallel groups as paired coroutines, and verifies each
step's postcondition before moving on. Failures are recorded with an
attributable phase and cause rather than raised.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from dinner_table.contracts.geometry import DRAWER_TRAVEL, HOME_JOINTS
from dinner_table.policies.conditioning import goal_for_skill
from dinner_table.reasoning.schema import RelativeTarget, Step, TaskGraph
from dinner_table.teacher.context import JAW_FORCE_MIN, SkillFailed, TeacherContext
from dinner_table.teacher.kinematics import arm_q
from dinner_table.teacher.skills import (
    CloseDrawer,
    Handoff,
    Hold,
    Home,
    OpenDrawer,
    ParallelGroup,
    Pick,
    Place,
    Pour,
    Retract,
)
from dinner_table.teacher.task_graphs import CANONICAL_GRAPHS

MAX_STEP_RETRIES = 2
STEP_TIMEOUT_S = 150.0
RECOVERY_TIMEOUT_S = 30.0
PLACE_XY_TOLERANCE_M = 0.012
HOME_TOLERANCE_RAD = 0.15

__all__ = ["CANONICAL_GRAPHS", "EpisodeLog", "FrameRecord", "StepRecord", "run_graph"]


@dataclass(frozen=True)
class StepRecord:
    """Outcome of one task-graph step, including its retries."""

    step_id: int
    skill: str
    arm: str
    outcome: str  # "success" or "failed"
    attempts: int
    phase_at_failure: str | None = None
    failure_cause: str | None = None


@dataclass(frozen=True)
class FrameRecord:
    """One control tick of the demonstration; images are re-rendered from the seed."""

    tick: int
    joints: list[float]
    action: list[float]
    skill: str
    phase: str
    goal_xyz: list[float]
    event: str = ""
    ctrl: list[float] = field(default_factory=list)
    sat: dict = field(default_factory=dict)
    step_id: int = -1


@dataclass
class EpisodeLog:
    """Frozen demonstration schema consumed by the data engine and the eval taxonomy."""

    episode_id: str
    seed: int
    dr_profile: str
    instruction: str
    task_id: str
    steps: list[StepRecord] = field(default_factory=list)
    frames: list[FrameRecord] = field(default_factory=list)
    success: bool = False
    water_fraction: float | None = None

    def to_dict(self) -> dict:
        """JSON-serializable view of the log."""
        return asdict(self)


def build_skill(step: Step):
    """Instantiate the teacher skill a validated step names."""
    if step.skill == "open_drawer":
        return OpenDrawer(step.arm)
    if step.skill == "close_drawer":
        return CloseDrawer(step.arm)
    if step.skill == "pick":
        return Pick(step.arm, step.object)
    if step.skill == "place":
        return Place(step.arm, step.object, step.target)
    if step.skill == "hold":
        return Hold(step.arm, step.object)
    if step.skill == "handoff":
        to_arm = "A" if step.target == "hand_of_A" else "B"
        return Handoff(step.object, step.arm, to_arm)
    if step.skill == "pour":
        amount = step.amount if step.amount is not None else 0.6
        return Pour(step.arm, step.object, str(step.target), amount)
    if step.skill == "home":
        return Home(step.arm)
    return Retract(step.arm)


def group_parallel(graph: TaskGraph) -> list[list[Step]]:
    """Execution units: parallel-group members travel together, in first-member order."""
    groups: list[list[Step]] = []
    index: dict[int, int] = {}
    for step in graph.steps:
        if step.parallel_group is None:
            groups.append([step])
        elif step.parallel_group in index:
            groups[index[step.parallel_group]].append(step)
        else:
            index[step.parallel_group] = len(groups)
            groups.append([step])
    return groups


def _primary(group: list[Step]) -> Step:
    """The step that drives a group; hold is the passive complement."""
    for step in group:
        if step.skill != "hold":
            return step
    return group[0]


def goal_of(ctx: TeacherContext, step: Step) -> np.ndarray:
    """Conditioning goal for a step, grounded in privileged ground truth.

    Mirrors the runtime's perceived-anchor resolution exactly: the only
    difference is that the poses are read from the simulator instead of the
    detector.
    """
    object_position = None
    if step.skill == "pour":
        # goal_for_skill's pour contract takes the receiving container's pose.
        object_position = ctx.object(str(step.target))[0]
    elif step.object is not None:
        object_position = ctx.object(step.object)[0]
    anchor_position = None
    if isinstance(step.target, RelativeTarget):
        anchor_position = ctx.object(step.target.anchor)[0]
    try:
        return goal_for_skill(
            step.skill, step.object, object_position, step.target, anchor_position=anchor_position
        )
    except ValueError:
        return np.zeros(3)


def place_target(ctx: TeacherContext, step: Step) -> np.ndarray:
    """World-frame landing spot a place step is judged against."""
    return Place(step.arm, step.object, step.target)._target_xyz(ctx)


def check_postcondition(ctx: TeacherContext, step: Step) -> tuple[bool, str]:
    """Verify a completed step against the same rules its skill verifies."""
    if step.skill == "open_drawer":
        if ctx.drawer_opening() < 0.88 * DRAWER_TRAVEL:
            return False, "jammed"
        return True, ""
    if step.skill == "close_drawer":
        if ctx.drawer_opening() > 0.12 * DRAWER_TRAVEL:
            return False, "jammed"
        return True, ""
    if step.skill in ("pick", "hold"):
        if ctx.carrying.get(step.arm) != step.object:
            return False, "missed_grasp"
        fixed, moving = ctx.finger_forces(step.arm, step.object)
        if min(fixed, moving) <= JAW_FORCE_MIN:
            return False, "missed_grasp"
        return True, ""
    if step.skill == "handoff":
        to_arm = "A" if step.target == "hand_of_A" else "B"
        if ctx.carrying.get(to_arm) != step.object:
            return False, "regrasp_missed"
        fixed, moving = ctx.finger_forces(to_arm, step.object)
        if min(fixed, moving) <= JAW_FORCE_MIN:
            return False, "regrasp_missed"
        return True, ""
    if step.skill == "place":
        target = place_target(ctx, step)
        position, _ = ctx.object(step.object)
        if float(np.linalg.norm(position[:2] - target[:2])) > PLACE_XY_TOLERANCE_M:
            return False, "misplaced"
        if ctx.object_support_force(step.object) <= 0.06:
            return False, "no_support"
        if ctx.carrying.get(step.arm) is not None:
            return False, "misplaced"
        return True, ""
    if step.skill == "pour":
        amount = step.amount if step.amount is not None else 0.6
        if ctx.fill_fraction(str(step.target)) < 0.8 * amount:
            return False, "spilled"
        return True, ""
    q = arm_q(ctx.data, step.arm)
    if float(np.max(np.abs(q - np.array(HOME_JOINTS[step.arm][:5])))) > HOME_TOLERANCE_RAD:
        return False, "not_parked"
    return True, ""


def _release_lost_grasps(ctx: TeacherContext, arms: tuple[str, ...]) -> None:
    """Drop bookkeeping for a grasp the jaws no longer hold, before a retry."""
    for arm in arms:
        held = ctx.carrying.get(arm)
        ctx.allowed[arm] = set()
        if held is None:
            continue
        fixed, moving = ctx.finger_forces(arm, held)
        if min(fixed, moving) <= JAW_FORCE_MIN:
            ctx.carrying[arm] = None


def _drive(
    ctx: TeacherContext, runner, log: EpisodeLog, group: list[Step], skill_name: str, perturber=None
) -> None:
    """Step physics through a skill's coroutine, recording one frame per tick."""
    primary = _primary(group)
    goal = goal_of(ctx, primary)
    if isinstance(runner, ParallelGroup):
        phase_source = runner.primary
    else:
        phase_source = runner
    for action in runner.run(ctx):
        phase = phase_source.phase
        if perturber is not None:
            fired = perturber.poll(ctx)
        else:
            fired = []
        tick = ctx.ticks()
        joints = [float(v) for v in ctx.scene.qpos_12()]
        commanded = [float(v) for v in np.asarray(action, dtype=np.float64)]
        sat = ctx.saturation_state()
        ctx.live.note_skill(phase_source)
        ctx.step(action)
        # NOTE: ctrl is captured post-step: the gripper saturation and drawer
        # servo rewrite data.ctrl during the substeps, and the replay needs the
        # final vector, not the pre-step action.
        log.frames.append(
            FrameRecord(
                tick=tick,
                joints=joints,
                action=commanded,
                skill=skill_name,
                phase=phase,
                goal_xyz=[float(v) for v in goal],
                event=",".join(ev.name for ev in fired),
                ctrl=[float(v) for v in ctx.data.ctrl],
                sat=sat,
                step_id=primary.id,
            )
        )


def run_graph(scene, graph: TaskGraph, seed: int, perturber=None) -> EpisodeLog:
    """Execute a validated task graph with privileged grounding; never raises."""
    ctx = TeacherContext(scene)
    ctx.begin(STEP_TIMEOUT_S)
    log = EpisodeLog(
        episode_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{graph.task_id}/{seed}")),
        seed=int(seed),
        dr_profile=str(scene.dr_profile_name),
        instruction=graph.instruction,
        task_id=graph.task_id,
    )
    completed = True
    for group in group_parallel(graph):
        primary = _primary(group)
        arms = tuple(sorted({step.arm for step in group}))
        record = None
        for attempt in range(1, MAX_STEP_RETRIES + 2):
            if len(group) == 1:
                runner = build_skill(primary)
                skill_name = primary.skill
            else:
                partner = next(step for step in group if step is not primary)
                runner = ParallelGroup(build_skill(primary), build_skill(partner))
                skill_name = f"{primary.skill}+{partner.skill}"
            # Re-snapshot the idle arm's hold from its live command: the
            # previous group may have left it holding something aloft, and the
            # begin-time snapshot would yank it back to its parked pose.
            ctx.latch_hold()
            ctx.extend_deadline(STEP_TIMEOUT_S)
            try:
                _drive(ctx, runner, log, group, skill_name, perturber)
                ok, cause = check_postcondition(ctx, primary)
                if not ok:
                    raise SkillFailed(primary.skill, "verify", cause)
            except SkillFailed as exc:
                # Keep the FIRST failure's attribution: once a grasp is gone the
                # retries fail on the missing precondition, which would mask the
                # cause that actually ended the step.
                if record is None:
                    phase, cause = exc.phase, exc.cause
                else:
                    phase, cause = record.phase_at_failure, record.failure_cause
                record = StepRecord(
                    primary.id, primary.skill, primary.arm, "failed", attempt, phase, cause
                )
                if attempt > MAX_STEP_RETRIES:
                    break
                if not _recover(ctx, log, arms, perturber, primary.id):
                    break
                continue
            record = StepRecord(primary.id, primary.skill, primary.arm, "success", attempt)
            break
        log.steps.append(record)
        if record.outcome == "failed":
            completed = False
            break
    log.success = completed
    log.water_fraction = ctx.fill_fraction("mug")
    return log


def _recover(
    ctx: TeacherContext, log: EpisodeLog, arms: tuple[str, ...], perturber=None, step_id: int = -1
) -> bool:
    """Park the acting arms for a retry; False when even the park fails."""
    _release_lost_grasps(ctx, arms)
    for arm in arms:
        if ctx.carrying.get(arm) is not None:
            continue  # a verified grasp is kept; the retry resumes from it
        ctx.extend_deadline(RECOVERY_TIMEOUT_S)
        home = Home(arm)
        try:
            for action in home.run(ctx):
                if perturber is not None:
                    fired = perturber.poll(ctx)
                else:
                    fired = []
                tick = ctx.ticks()
                joints = [float(v) for v in ctx.scene.qpos_12()]
                commanded = [float(v) for v in np.asarray(action, dtype=np.float64)]
                sat = ctx.saturation_state()
                ctx.step(action)
                log.frames.append(
                    FrameRecord(
                        tick=tick,
                        joints=joints,
                        action=commanded,
                        skill="home",
                        phase=home.phase,
                        goal_xyz=[0.0, 0.0, 0.0],
                        event=",".join(ev.name for ev in fired),
                        ctrl=[float(v) for v in ctx.data.ctrl],
                        sat=sat,
                        step_id=step_id,
                    )
                )
        except SkillFailed:
            return False
    return True
