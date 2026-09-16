"""SkillExecutorSource: the executor + VLM + perception as a runtime action source.

Structural ActionSource (connect / update / disconnect) per the §4 Phase 5
sketch — the seam Intel's runtime designates for custom skill-switching logic.
No physicalai import here on purpose: the class is duck-typed by the runtime
via the dinner_demo.yaml class_path, and stays importable (and testable) on
hosts without the Intel stack.

Protocol rules honored: update() ALWAYS returns a (12,) action within joint
limits (never None), and the VLM runs off-thread — update() only polls ready().
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from dinner_table.contracts.geometry import POLICY_CAMERA_NAMES
from dinner_table.executor.graph_executor import (
    TASK_OBJECTS,
    Executor,
    PolicyObservation,
    TaskAborted,
)
from dinner_table.perception.interfaces import Detector
from dinner_table.policies.conditioning import build_state
from dinner_table.reasoning.interfaces import VlmEngine
from dinner_table.reasoning.schema import ObjectStatus, SceneSummary
from dinner_table.runtime.telemetry import TickRecord, blank_stages

# Executor-internal VLM call names translated to the §10.7 vocabulary.
_VLM_CALL_MAP = {"verify": "precondition"}


def _empty_summary(instruction: str) -> SceneSummary:
    """Pre-perception summary for the connect-time parse: nothing yet visible."""
    return SceneSummary(
        instruction=instruction,
        objects={name: ObjectStatus(visible=False, where="not visible") for name in TASK_OBJECTS},
        drawer_open=False,
        completed_step_ids=[],
        current_step_id=None,
    )


def _frame_array(frame: Any) -> np.ndarray:
    data = frame.data if hasattr(frame, "data") else frame
    return np.asarray(data)


def _bus_task(bus: Any) -> str | None:
    if bus is None:
        return None
    task = getattr(bus, "task", None)
    if isinstance(task, str) and task:
        return task
    if isinstance(bus, dict):
        task = bus.get("task")
        if isinstance(task, str) and task:
            return task
    return None


class SkillExecutorSource:
    """Hosts Executor + VLM + detector + ACT policy inside the runtime tick."""

    def __init__(
        self,
        act: Any,
        vlm: VlmEngine,
        detector: Detector,
        executor: Executor,
        task: str | None = None,
        episode: str = "",
        telemetry: Any | None = None,
    ) -> None:
        self._act = act
        self._vlm = vlm
        self._detector = detector
        self._executor = executor
        self._task = task
        self._episode = episode
        self._telemetry = telemetry
        self._parse_outstanding = False
        self._terminal_reason: str | None = None
        self._safe_action = np.zeros(12, dtype=np.float64)

    # ------------------------------------------------------------- properties

    @property
    def terminal_reason(self) -> str | None:
        """Set once the executor aborts; latched for the rest of the episode."""
        return self._terminal_reason

    # ------------------------------------------------------- ActionSource API

    def connect(self, *, bus: Any = None, session_id: str = "") -> None:
        """Reset the policy, fire the background parse, begin the default skill."""
        task = self._task or _bus_task(bus)
        if not task:
            raise ValueError(
                "no session instruction: pass task= or provide bus.task (runtime "
                "config 'task' field, read by the demo wiring)"
            )
        self._task = task
        if not self._episode:
            self._episode = session_id or "episode-0"
        self._terminal_reason = None
        self._parse_outstanding = False
        self._act.reset()
        self._vlm.submit_parse(task, _empty_summary(task))
        self._parse_outstanding = True
        self._executor.begin_default_skill()

    def update(
        self,
        robot_state: Any,
        camera_frames: dict[str, Any],
        step: int,
    ) -> np.ndarray:
        """One 25 Hz tick: poll VLM → tick executor → policy action → gate."""
        joints = np.asarray(robot_state.joint_positions, dtype=np.float64)
        if joints.shape != (12,):
            raise ValueError(f"joints must have shape (12,), got {joints.shape}")
        frames = {name: _frame_array(frame) for name, frame in camera_frames.items()}

        if self._terminal_reason is not None:
            return self._emit_hold(joints, step, repeat=True)

        vlm_call: str | None = None
        if self._parse_outstanding and self._vlm.ready:
            self._parse_outstanding = False
            graph = self._vlm.result()
            if graph is not None:
                self._executor.adopt_graph(graph)
                vlm_call = "parse"

        want_frames = self._executor.needs_grounding()
        overhead_rgb = frames.get("overhead") if want_frames else None
        overhead_depth = frames.get("overhead_depth") if want_frames else None
        detections: list[dict] = []
        if want_frames and overhead_rgb is not None:
            # Telemetry/HUD copy of the event-triggered detection. The executor
            # independently detects inside tick() for control (its frozen API
            # takes frames, not detections), so grounding ticks pay one extra
            # YOLO inference (~1 in 12 ticks plus plan boundaries).
            detect_t0 = time.perf_counter()
            for det in self._detector.detect(overhead_rgb):
                detections.append(
                    {
                        "label": det.label,
                        "xyxy": [float(v) for v in det.xyxy],
                        "confidence": float(det.confidence),
                    }
                )
            detect_ms = (time.perf_counter() - detect_t0) * 1000.0
        else:
            detect_ms = 0.0

        executor_t0 = time.perf_counter()
        try:
            self._executor.tick(
                PolicyObservation(
                    joints=joints,
                    overhead_rgb=overhead_rgb,
                    overhead_depth=overhead_depth,
                    timestamp=time.monotonic(),
                )
            )
        except TaskAborted as exc:
            self._terminal_reason = exc.reason
            self._safe_action = self._executor.gate(joints)
            return self._emit_hold(joints, step, repeat=False)
        executor_ms = (time.perf_counter() - executor_t0) * 1000.0

        skill, arm, obj, goal = self._executor.conditioning()
        state = build_state(joints=joints, skill=skill, arm=arm, object_name=obj, goal_xyz=goal)
        obs = {"state": state}
        for name in POLICY_CAMERA_NAMES:
            if name in frames:
                obs[f"images.{name}"] = frames[name]
        policy_t0 = time.perf_counter()
        action = np.asarray(self._act.select_action(obs), dtype=np.float64)
        policy_ms = (time.perf_counter() - policy_t0) * 1000.0
        if action.shape != (12,):
            raise ValueError(f"policy must return shape (12,), got {action.shape}")
        gated = self._executor.gate(action)

        ex_state = self._executor.state()
        if vlm_call is None:
            vlm_call = _VLM_CALL_MAP.get(ex_state["vlm_call"], ex_state["vlm_call"])
        stage_ms = blank_stages()
        stage_ms["detect"] = detect_ms
        stage_ms["policy"] = policy_ms
        stage_ms["executor"] = executor_ms
        record = TickRecord(
            timestamp=time.monotonic(),
            episode=self._episode,
            step=int(step),
            stage_ms=stage_ms,
            active_skill=ex_state["active_skill"],
            active_arm=ex_state["active_arm"],
            parallel_group=ex_state["parallel_group"],
            vlm_call=vlm_call,
            vlm_pending=bool(ex_state["vlm_pending"]) or self._parse_outstanding,
            detections=detections,
            goal_xyz=ex_state["goal_xyz"],
            action=[float(v) for v in gated],
            postcondition=ex_state["postcondition"],
            recovered=ex_state["recovered"],
        )
        self._emit(record)
        return gated

    def disconnect(self) -> None:
        """Park the executor in its safe hold; the runtime stops ticking after."""
        self._safe_action = self._executor.hold_safe()

    # --------------------------------------------------------------- internals

    def _emit_hold(self, joints: np.ndarray, step: int, repeat: bool) -> np.ndarray:
        """Latched post-abort path: safe hold action + telemetry every tick."""
        held = self._executor.gate(self._safe_action)
        record = TickRecord(
            timestamp=time.monotonic(),
            episode=self._episode,
            step=int(step),
            stage_ms=blank_stages(),
            active_skill=None,
            active_arm=None,
            parallel_group=None,
            vlm_call=None,
            vlm_pending=False,
            detections=[],
            goal_xyz=None,
            action=[float(v) for v in held],
            postcondition=f"fail:aborted:{self._terminal_reason}",
            recovered=None if repeat else f"abort:{self._terminal_reason}",
        )
        self._emit(record)
        return held

    def _emit(self, record: TickRecord) -> None:
        sink = self._telemetry
        if sink is None:
            return
        if hasattr(sink, "attach"):
            sink.attach(record)
        else:
            sink(record)


TelemetrySink = Callable[[TickRecord], None]
