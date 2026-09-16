"""Graph executor: the decision state machine between the VLM and the policy.

The Executor owns task-level decision logic only - no physics, no inference.
Per tick it consumes a PolicyObservation (fresh camera frames only when it
asked for them via needs_grounding()), refreshes its perceived WorldState
through the Detector + Tracker contracts, advances the current task-graph
group, gates pre/postconditions, and triggers recovery or bounded VLM
replans. SkillExecutorSource drives it: conditioning() feeds
conditioning.build_state, the ACT policy provides actions, gate() safety-clamps
them, and TaskAborted (raised on terminal aborts) is caught to end the episode.

Timing constants are ticks at 25 Hz: postcondition polling starts after
MIN_STEP_TICKS and repeats every POLL_EVERY_TICKS; a step times out at
STEP_TIMEOUT_TICKS; pour groups complete by POUR_DURATION_TICKS with water-level
verification delegated to the VLM boundary check (run off-thread through
CALL_RUNNER so a blocking call never stalls a tick).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import CONTROL_HZ
from dinner_table.executor.preconditions import WorldState, check_postcondition, check_precondition
from dinner_table.executor.recovery import handle, handle_boundary
from dinner_table.executor.scheduler import group_steps
from dinner_table.executor.workspace import ZoneClaims, zone_for_step, zone_of
from dinner_table.perception.interfaces import (
    OBJECT_LABELS,
    Detector,
    PerceptionSnapshot,
    Tracker,
)
from dinner_table.policies.conditioning import goal_for_skill
from dinner_table.reasoning.interfaces import VlmEngine
from dinner_table.reasoning.schema import (
    ObjectStatus,
    PreconditionReport,
    RelativeTarget,
    SceneSummary,
    Step,
    TaskGraph,
)

MIN_STEP_TICKS = 25
POLL_EVERY_TICKS = 12
STEP_TIMEOUT_TICKS = 200
POUR_DURATION_TICKS = 90
RETRACT_PAUSE_TICKS = 25
MAX_STEP_RETRIES = 2
MAX_REPLAN_DEPTH = 3
ZONE_WAIT_MAX_TICKS = 50
WIGGLE_AMPLITUDE_RAD = 0.05
WIGGLE_FREQUENCY_HZ = 1.5
VELOCITY_LIMIT_PER_TICK = 0.15
GRIPPER_VELOCITY_PER_TICK = 0.10
DRAWER_LABELS = ("drawer_open", "drawer_closed")
TASK_OBJECTS = tuple(name for name in OBJECT_LABELS if name not in DRAWER_LABELS)

# Provisional conservative joint limits (lo, hi) per joint, JOINT_NAMES order:
# 5 arm hinges + gripper aperture per arm. Replaced by calibration-derived
# limits when the scene wiring lands (same file owner).
_ARM_LIMITS = (
    (-1.66, 1.66),
    (-1.83, 1.83),
    (-1.83, 1.83),
    (-1.83, 1.83),
    (-1.75, 1.75),
    (0.0, 1.0),
)
SAFE_JOINT_LIMITS = _ARM_LIMITS + _ARM_LIMITS


class ExecutorError(DinnerTableError):
    """Invalid executor usage."""


class TaskAborted(DinnerTableError):
    """Terminal abort; consumers catch this and end the episode gracefully."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PolicyObservation:
    """One control tick's inputs for the executor's decision logic.

    Camera frames are present only on ticks where needs_grounding() was True
    before the tick (event-triggered perception)."""

    joints: np.ndarray  # (12,) current joint positions, JOINT_NAMES order
    overhead_rgb: np.ndarray | None = None  # (H, W, 3) uint8, or None
    overhead_depth: np.ndarray | None = None  # (H, W) float32 meters, or None
    timestamp: float = 0.0


def _thread_runner(fn, args, box: dict) -> None:
    def _run() -> None:
        try:
            box["result"] = fn(*args)
        except Exception as exc:  # noqa: BLE001 - worker records, never crashes the thread
            box["error"] = exc
        finally:
            box["done"] = True

    threading.Thread(target=_run, daemon=True).start()


def _inline_runner(fn, args, box: dict) -> None:
    try:
        box["result"] = fn(*args)
    except Exception as exc:  # noqa: BLE001 - recorded for the poller either way
        box["error"] = exc
    finally:
        box["done"] = True


# Injectable so tests run blocking VLM calls deterministically (synchronously).
CALL_RUNNER = _thread_runner


class Executor:
    """Task-graph state machine (frozen SkillExecutorSource-facing API)."""

    def __init__(self, vlm: VlmEngine, detector: Detector, tracker: Tracker) -> None:
        self._vlm = vlm
        self._detector = detector
        self._tracker = tracker
        self._mode = "default"
        self._graph: TaskGraph | None = None
        self._groups: list[list[Step]] | None = None
        self._sched_log: list[str] = []
        self._group_i = 0
        self._step_timer = 0
        self._tick = 0
        self._joints: np.ndarray | None = None
        self._world: WorldState | None = None
        self._latest_poses: dict = {}
        self._world_fresh = False
        self._want_frames = False
        self._boundary_fails = 0
        self._zone_wait = 0
        self._claims = ZoneClaims()
        self._claimed_arms: set[str] = set()
        self._retries: dict[int, int] = {}
        self._replan_depth = 0
        self._parse_failures = 0
        self._completed_ids: list[int] = []
        self._recovered: str | None = None
        self._postcondition: str | None = None
        self._vlm_pending = False
        self._vlm_call: str | None = None
        self._wiggle = False
        self._verify: dict | None = None
        self._resume_step: Step | None = None
        self._terminal_reason: str | None = None

    # ------------------------------------------------------------------ public

    @property
    def world(self) -> WorldState | None:
        """Latest perceived world state (recovery classification input)."""
        return self._world

    def adopt_graph(self, graph: TaskGraph) -> None:
        """Install a parsed plan; groups are computed lazily at the first
        boundary with fresh perception so reassignment sees real poses."""
        self._graph = graph
        self._groups = None
        self._group_i = 0
        self._retries = {}
        self._completed_ids = []
        self._boundary_fails = 0
        self._verify = None
        self._mode = "boundary"
        self._want_frames = True  # a fresh plan needs perception immediately

    def begin_default_skill(self) -> None:
        """Safe retract/home conditioning (pre-plan and replan-wait mode)."""
        self._mode = "default"
        self._want_frames = False

    def needs_grounding(self) -> bool:
        """True when the NEXT tick should carry fresh camera frames."""
        return self._want_frames

    def current_step(self) -> Step | None:
        if self._groups is None or self._mode not in ("running", "boundary"):
            return None
        if self._group_i >= len(self._groups):
            return None
        return self._primary(self._groups[self._group_i])

    def conditioning(self) -> tuple[str, str, str | None, np.ndarray]:
        """(skill, arm, object, goal_xyz) for the current tick."""
        if self._mode == "running" and self._groups is not None:
            primary = self._primary(self._groups[self._group_i])
            goal = self._goal_for(primary)
            if goal is not None:
                return (primary.skill, primary.arm, primary.object, goal)
            self._want_frames = True  # unresolvable goal: request perception
        return ("home", "A", None, np.zeros(3))

    def gate(self, action: np.ndarray) -> np.ndarray:
        """Safety gate: joint-limit clamp, velocity clamp, idle-arm hold,
        wiggle override for jammed drawer steps. Never returns None."""
        arr = np.asarray(action, dtype=np.float64).copy()
        if arr.shape != (12,):
            raise ExecutorError(f"action must have shape (12,), got {arr.shape}")
        for i, (lo, hi) in enumerate(SAFE_JOINT_LIMITS):
            arr[i] = min(max(arr[i], lo), hi)
        if self._joints is not None:
            for i in range(12):
                limit = GRIPPER_VELOCITY_PER_TICK if i % 6 == 5 else VELOCITY_LIMIT_PER_TICK
                arr[i] = min(max(arr[i], self._joints[i] - limit), self._joints[i] + limit)
                lo, hi = SAFE_JOINT_LIMITS[i]
                arr[i] = min(max(arr[i], lo), hi)
        active_arm = self._active_arm()
        if active_arm is not None and self._joints is not None:
            idle = "B" if active_arm == "A" else "A"
            base = 0 if idle == "A" else 6
            arr[base : base + 6] = self._joints[base : base + 6]
        if self._wiggle and active_arm is not None:
            base = 0 if active_arm == "A" else 6
            phase = 2.0 * math.pi * WIGGLE_FREQUENCY_HZ * self._step_timer / CONTROL_HZ
            for j in range(5):
                lo, hi = SAFE_JOINT_LIMITS[base + j]
                arr[base + j] = min(
                    max(arr[base + j] + WIGGLE_AMPLITUDE_RAD * math.sin(phase), lo), hi
                )
        return arr

    def hold_safe(self) -> np.ndarray:
        """Safe-hold action for disconnect/abort: current joints clamped to
        limits, or zeros (in-limits mid-pose) before the first tick."""
        if self._joints is None:
            return np.zeros(12, dtype=np.float64)
        return np.array(
            [min(max(v, lo), hi) for v, (lo, hi) in zip(self._joints, SAFE_JOINT_LIMITS)],
            dtype=np.float64,
        )

    def done(self) -> bool:
        return self._mode == "terminal"

    def state(self) -> dict:
        primary = self.current_step()
        goal = self._goal_for(primary) if primary is not None else None
        return {
            "tick": self._tick,
            "mode": self._mode,
            "active_skill": primary.skill
            if primary
            else ("home" if self._mode != "terminal" else None),
            "active_arm": primary.arm if primary else None,
            "parallel_group": primary.parallel_group if primary else None,
            "vlm_pending": self._vlm_pending,
            "vlm_call": self._vlm_call,
            "goal_xyz": goal.tolist() if goal is not None else None,
            "postcondition": self._postcondition,
            "recovered": self._recovered,
            "retries_left": {
                step_id: MAX_STEP_RETRIES - used for step_id, used in self._retries.items()
            },
            "group_index": self._group_i if self._groups is not None else None,
            "replan_depth": self._replan_depth,
            "done": self._mode == "terminal",
            "terminal_reason": self._terminal_reason,
            "wiggle": self._wiggle,
        }

    def tick(self, obs: PolicyObservation) -> None:
        """One control tick of decision logic. Raises TaskAborted on terminal
        aborts (state is recorded first; consumers catch and end the episode)."""
        self._tick += 1
        self._recovered = None
        self._postcondition = None
        self._vlm_call = None
        joints = np.asarray(obs.joints, dtype=np.float64)
        if joints.shape != (12,):
            raise ExecutorError(f"joints must have shape (12,), got {joints.shape}")
        self._joints = joints
        if obs.overhead_rgb is not None and obs.overhead_depth is not None:
            self._refresh_world(obs)
        if self._mode == "replanning":
            self._poll_replan()
        elif self._mode == "boundary":
            self._boundary_step()
        elif self._mode == "running":
            self._running_step()
        elif self._mode == "retract_pause":
            self._retract_step()
        self._vlm_pending = self._mode == "replanning" or self._verify is not None
        self._want_frames = (self._mode == "boundary" and self._verify is None) or (
            self._mode == "running" and self._poll_due(self._step_timer + 1)
        )

    # ----------------------------------------------------- recovery request API

    def record_recovery(self, mode: str) -> None:
        self._recovered = mode

    def request_retry(self, step: Step) -> None:
        if self._retries_exhausted(step):
            self.request_replan(f"retries exhausted for step {step.id}")
            return
        self._retries[step.id] = self._retries.get(step.id, 0) + 1
        self._step_timer = 0

    def request_rollback_to_pick(self, step: Step) -> None:
        if self._retries_exhausted(step):
            self.request_replan(f"retries exhausted for step {step.id}")
            return
        self._retries[step.id] = self._retries.get(step.id, 0) + 1
        self._rollback_to_pick_of(step.object)

    def request_replan(self, reason: str) -> None:
        if self._replan_depth >= MAX_REPLAN_DEPTH:
            self._abort(f"replan_depth_exceeded ({reason})")
            return
        if self._graph is None:
            self._abort(f"replan_without_graph ({reason})")
            return
        self._replan_depth += 1
        self._vlm_call = "parse"
        self._vlm.submit_parse(self._graph.instruction, self._summary())
        self._mode = "replanning"

    def request_wiggle(self, step: Step) -> None:
        if self._retries_exhausted(step):
            self.request_replan(f"retries exhausted for step {step.id}")
            return
        self._retries[step.id] = self._retries.get(step.id, 0) + 1
        self._wiggle = True
        self._step_timer = 0

    def request_retract_pause(self, step: Step) -> None:
        if self._retries_exhausted(step):
            self.request_replan(f"retries exhausted for step {step.id}")
            return
        self._retries[step.id] = self._retries.get(step.id, 0) + 1
        self._resume_step = step
        self._step_timer = 0
        self._mode = "retract_pause"

    def request_repour(self, step: Step) -> None:
        if self._retries_exhausted(step):
            self.request_replan(f"retries exhausted for step {step.id}")
            return
        self._retries[step.id] = self._retries.get(step.id, 0) + 1
        if self._groups is not None:
            for gi, group in enumerate(self._groups):
                if any(member.id == step.id for member in group):
                    self._group_i = gi
                    break
        self._verify = None
        self._boundary_fails = 0
        self._mode = "boundary"

    # ------------------------------------------------------------- internals

    def _retries_exhausted(self, step: Step) -> bool:
        return self._retries.get(step.id, 0) >= MAX_STEP_RETRIES

    def _primary(self, group: list[Step]) -> Step:
        """The group's primary step drives conditioning; hold is the passive
        complement, so the first non-hold member is primary."""
        for step in group:
            if step.skill != "hold":
                return step
        return group[0]

    def _active_arm(self) -> str | None:
        """Arm whose joints follow the policy; None means both arms follow."""
        if self._mode != "running" or self._groups is None:
            return None
        primary = self._primary(self._groups[self._group_i])
        if primary.skill in ("home", "retract"):
            return None
        return primary.arm

    def _poll_due(self, timer: int) -> bool:
        return timer >= MIN_STEP_TICKS and (timer - MIN_STEP_TICKS) % POLL_EVERY_TICKS == 0

    def _abort(self, reason: str) -> None:
        self._mode = "terminal"
        self._terminal_reason = f"aborted:{reason}"
        for arm in self._claimed_arms:
            self._claims.release(arm)
        self._claimed_arms = set()
        raise TaskAborted(reason)

    def _complete(self) -> None:
        self._mode = "terminal"
        self._terminal_reason = "completed"

    def _finish_group(self, pour_verify: Step | None = None) -> None:
        for arm in self._claimed_arms:
            self._claims.release(arm)
        self._claimed_arms = set()
        self._wiggle = False
        if self._groups is not None and self._group_i < len(self._groups):
            self._completed_ids.extend(step.id for step in self._groups[self._group_i])
        self._group_i += 1
        if pour_verify is not None:
            self._verify = {"step": pour_verify, "submitted": False, "box": None}
        if self._group_i >= (len(self._groups) if self._groups else 0) and self._verify is None:
            self._complete()
        else:
            self._mode = "boundary"
            self._boundary_fails = 0

    def _rollback_to_pick_of(self, obj: str | None) -> None:
        if self._groups is not None and obj is not None:
            for gi in range(self._group_i - 1, -1, -1):
                if any(m.skill == "pick" and m.object == obj for m in self._groups[gi]):
                    self._group_i = gi
                    self._boundary_fails = 0
                    self._mode = "boundary"
                    return
        self.request_replan(f"no pick of {obj} to roll back to")

    def _goal_for(self, step: Step) -> np.ndarray | None:
        obj_pos = None
        if step.object is not None and self._world is not None:
            pose = self._world.poses.get(step.object)
            obj_pos = pose.position if pose is not None else None
        anchor_pos = None
        if isinstance(step.target, RelativeTarget) and self._world is not None:
            pose = self._world.poses.get(step.target.anchor)
            anchor_pos = pose.position if pose is not None else None
        try:
            return goal_for_skill(step.skill, step.object, obj_pos, step.target, anchor_pos)
        except ValueError:
            return None

    def _refresh_world(self, obs: PolicyObservation) -> None:
        detections = self._detector.detect(obs.overhead_rgb)
        labels = {d.label for d in detections}
        if "drawer_open" in labels:
            drawer_open = True
        elif "drawer_closed" in labels:
            drawer_open = False
        else:
            drawer_open = self._world.drawer_open if self._world is not None else False
        snapshot = PerceptionSnapshot(
            overhead_rgb=obs.overhead_rgb,
            overhead_depth=obs.overhead_depth,
            detections=tuple(detections),
            timestamp=obs.timestamp,
        )
        grounded = {}
        for name in TASK_OBJECTS:
            pose = self._detector.ground(snapshot, name)
            if pose is not None:
                grounded[name] = pose
        filtered = self._tracker.update(grounded)
        self._latest_poses = filtered
        fused = self._tracker.fuse_held(
            filtered, {"A": float(self._joints[10]), "B": float(self._joints[11])}
        )
        # Grounded implies confidently visible (the Detector contract); the
        # stricter depth_fusion multiview check wires in at runtime integration.
        self._world = WorldState(
            poses=fused,
            drawer_open=drawer_open,
            multiview_ok=frozenset(grounded),
            joints={"A": self._joints[0:6].copy(), "B": self._joints[6:12].copy()},
        )
        self._world_fresh = True

    def _summary(self) -> SceneSummary:
        poses = self._world.poses if self._world is not None else {}
        drawer_open = self._world.drawer_open if self._world is not None else False
        objects = {}
        for name in TASK_OBJECTS:
            pose = poses.get(name)
            if pose is None:
                objects[name] = ObjectStatus(visible=False, where="not visible", held_by=None)
            else:
                objects[name] = ObjectStatus(
                    visible=True, where=_where_words(pose.position), held_by=pose.held_by
                )
        return SceneSummary(
            instruction=self._graph.instruction if self._graph else "",
            objects=objects,
            drawer_open=drawer_open,
            completed_step_ids=list(self._completed_ids),
            current_step_id=(self.current_step().id if self.current_step() is not None else None),
        )

    def _poll_replan(self) -> None:
        if not self._vlm.ready:
            return
        graph = self._vlm.result()
        if graph is None:
            self._parse_failures += 1
            if self._parse_failures >= 2:
                self._abort(f"parse_failed: {self._vlm.last_error() or 'no result'}")
                return
            self._vlm.submit_parse(self._graph.instruction, self._summary())
            return
        self.adopt_graph(graph)

    def _verify_step(self) -> None:
        if not self._verify["submitted"]:
            upcoming = self.current_step()
            step_id = upcoming.id if upcoming is not None else self._verify["step"].id
            box = {"done": False, "result": None, "error": None}
            self._vlm_call = "verify"
            CALL_RUNNER(self._vlm.check_preconditions, (self._graph, step_id, self._summary()), box)
            self._verify.update(submitted=True, box=box)
            return
        box = self._verify["box"]
        if not box["done"]:
            return
        if box["error"] is not None:
            report = PreconditionReport(ok=False, reason=f"vlm check error: {box['error']}")
        else:
            report = box["result"]
        step = self._verify["step"]
        self._verify = None
        if not report.ok:
            handle(step, report, self)

    def _boundary_step(self) -> None:
        if self._verify is not None:
            self._verify_step()
            if self._verify is not None or self._mode != "boundary":
                return
        if self._groups is None:
            if not self._world_fresh:
                return
            self._groups, self._sched_log = group_steps(self._graph, dict(self._latest_poses))
            self._group_i = 0
        if self._group_i >= len(self._groups):
            self._complete()
            return
        group = self._groups[self._group_i]
        if not self._world_fresh:
            return
        failures = []
        for step in group:
            report = check_precondition(step, self._world)
            if not report.ok:
                failures.append((step, report))
        if failures:
            self._boundary_fails += 1
            if self._boundary_fails >= 2:
                recovery_step, recovery_report = failures[0]
                handle_boundary(recovery_step, recovery_report, self)
            return
        self._boundary_fails = 0
        if len(group) == 1:
            claims_ok = self._claims.claim(group[0].arm, zone_for_step(group[0]))
        else:
            # A coordinated parallel group (hold+pour) shares its zones by
            # design: claim once under the first member's arm so uncoordinated
            # groups are still excluded while members never block each other.
            first_arm = group[0].arm
            claims_ok = all(
                self._claims.claim(first_arm, zone)
                for zone in {zone_for_step(step) for step in group}
            )
        if not claims_ok:
            self._zone_wait += 1
            if self._zone_wait > ZONE_WAIT_MAX_TICKS:
                self._abort("zone_deadlock")
                return
            return
        self._zone_wait = 0
        self._claimed_arms = {step.arm for step in group}
        self._step_timer = 0
        self._world_fresh = False
        self._mode = "running"

    def _running_step(self) -> None:
        group = self._groups[self._group_i]
        primary = self._primary(group)
        self._step_timer += 1
        if any(step.skill == "pour" for step in group):
            if self._step_timer >= POUR_DURATION_TICKS:
                self._postcondition = "pass"  # duration completed; VLM verifies next
                pour_member = next(step for step in group if step.skill == "pour")
                self._finish_group(pour_verify=pour_member)
            return
        if self._step_timer >= STEP_TIMEOUT_TICKS:
            self._postcondition = "fail:timeout"
            handle(primary, PreconditionReport(ok=False, reason="step timeout"), self)
            return
        if self._poll_due(self._step_timer) and self._world_fresh:
            goal = self._goal_for(primary) if primary.skill == "place" else None
            report = check_postcondition(primary, self._world, goal=goal)
            if report.ok:
                self._postcondition = "pass"
                self._finish_group()
            else:
                self._postcondition = f"fail:{report.reason}"
                handle(primary, report, self)

    def _retract_step(self) -> None:
        self._step_timer += 1
        if self._step_timer >= RETRACT_PAUSE_TICKS and self._resume_step is not None:
            step = self._resume_step
            self._resume_step = None
            self._step_timer = 0
            self._rollback_to_pick_of(step.object)


def _where_words(position: np.ndarray) -> str:
    """Coarse natural-language position for the VLM's SceneSummary."""
    zone = zone_of(position)
    if zone == "A":
        return "on the right side of the table"
    if zone == "B":
        return "on the left side of the table"
    if zone == "shared":
        return "near the center of the table"
    if position[1] > 0.4:
        return "in or near the drawer"
    return "beyond the arms' reach"
