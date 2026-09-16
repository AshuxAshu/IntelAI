"""B11: SkillExecutorSource + telemetry + HUD.

No physicalai import anywhere in this file: the source is structural (the
runtime duck-types connect/update/disconnect), so these tests run on any host.
Device-latency budgets re-verify on the i7 in B12's system-mode test; here the
latency test asserts record completeness plus a host-independent ceiling.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from dinner_table.contracts.geometry import HOME_JOINTS, POLICY_CAMERA_NAMES
from dinner_table.executor.graph_executor import (
    SAFE_JOINT_LIMITS,
    Executor,
    TaskAborted,
)
from dinner_table.perception.interfaces import Detection, ObjectPose3D
from dinner_table.reasoning.schema import (
    PreconditionReport,
    Step,
    TaskGraph,
    VlmDiagnosis,
)
from dinner_table.runtime.hud import render_hud
from dinner_table.runtime.skill_source import SkillExecutorSource
from dinner_table.runtime.telemetry import (
    STAGE_KEYS,
    HudTelemetryCallback,
    TickRecord,
    blank_stages,
)

HOME_12 = np.array(HOME_JOINTS["A"] + HOME_JOINTS["B"], dtype=np.float64)


def _graph(*skills: str) -> TaskGraph:
    return TaskGraph(
        task_id="t",
        instruction="go home",
        steps=[Step(id=i + 1, skill=skill, arm="A") for i, skill in enumerate(skills)],
    )


class FakeAct:
    """ACT stand-in asserting the PolicySource input convention."""

    def __init__(self) -> None:
        self.resets = 0
        self.calls = 0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, obs: dict) -> np.ndarray:
        self.calls += 1
        assert set(obs) == {"state"} | {f"images.{c}" for c in POLICY_CAMERA_NAMES}
        assert obs["state"].shape == (35,)
        return np.zeros(12)


class FakeVlm:
    """VlmEngine double: background parse resolving after `delay_s`. Not a stub
    of Dev A's engines — a timing double for the source's own poll behavior."""

    def __init__(self, graph: TaskGraph | None = None, delay_s: float = 0.0) -> None:
        self._graph = graph if graph is not None else _graph("home")
        self._delay_s = delay_s
        self._ready = False
        self.submits: list[str] = []

    def submit_parse(self, instruction: str, summary) -> None:
        self.submits.append(instruction)
        self._ready = False

        def _run() -> None:
            time.sleep(self._delay_s)
            self._ready = True

        threading.Thread(target=_run, daemon=True).start()

    @property
    def ready(self) -> bool:
        return self._ready

    def result(self):
        return self._graph if self._ready else None

    def last_error(self) -> str | None:
        return None

    def check_preconditions(self, graph, step_id, summary) -> PreconditionReport:
        return PreconditionReport(ok=True, reason="fake")

    def diagnose(self, graph, step_id, summary) -> VlmDiagnosis:
        return VlmDiagnosis(anomaly="none", explanation="fake", suggested_action="retry_skill")


class FakeDetector:
    def __init__(self, poses: dict[str, ObjectPose3D] | None = None) -> None:
        self._poses = poses or {}
        self.detect_calls = 0

    def detect(self, overhead_rgb: np.ndarray) -> list[Detection]:
        self.detect_calls += 1
        return [Detection(label="plate", xyxy=(10.0, 10.0, 60.0, 60.0), confidence=0.9)]

    def ground(self, snapshot, target: str):
        return self._poses.get(target)


class FakeTracker:
    def update(self, poses: dict) -> dict:
        return dict(poses)

    def fuse_held(self, poses: dict, apertures: dict) -> dict:
        return dict(poses)


class FakeRobotState:
    def __init__(self, joints: np.ndarray | None = None) -> None:
        self.joint_positions = HOME_12.copy() if joints is None else joints


def _frames(depth: bool = True) -> dict[str, np.ndarray]:
    frames = {name: np.zeros((64, 64, 3), dtype=np.uint8) for name in POLICY_CAMERA_NAMES}
    if depth:
        frames["overhead_depth"] = np.full((64, 64), 0.8, dtype=np.float32)
    return frames


def _source(
    vlm: FakeVlm | None = None, telemetry=None
) -> tuple[SkillExecutorSource, FakeAct, Executor]:
    vlm = vlm if vlm is not None else FakeVlm(delay_s=3600.0)  # never resolves
    act = FakeAct()
    det = FakeDetector()
    ex = Executor(vlm, det, FakeTracker())
    src = SkillExecutorSource(act, vlm, det, ex, task="go home", telemetry=telemetry)
    return src, act, ex


def _in_limits(action: np.ndarray) -> bool:
    return all(lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(action, SAFE_JOINT_LIMITS))


# ------------------------------------------------------------------ B11 tests


def test_action_source_protocol() -> None:
    src, act, _ = _source()
    assert callable(src.connect) and callable(src.update) and callable(src.disconnect)
    src.connect(session_id="s1")
    assert act.resets == 1
    state = FakeRobotState()
    for step in range(10):
        action = src.update(state, _frames(), step)
        assert action is not None
        assert action.shape == (12,)
        assert _in_limits(action)
    assert act.calls == 10
    src.disconnect()


def test_latency_budget() -> None:
    records: list[TickRecord] = []
    src, _, _ = _source(telemetry=records.append)
    src.connect(session_id="s1")
    state = FakeRobotState()
    for step in range(100):
        src.update(state, _frames(), step)
    assert len(records) == 100
    for record in records:
        assert set(record.stage_ms) == set(STAGE_KEYS)
    p95 = float(np.percentile([r.stage_ms["executor"] for r in records], 95))
    assert p95 < 50.0  # host-independent ceiling; device budgets re-verify on i7


def test_vlm_never_blocks() -> None:
    vlm = FakeVlm(delay_s=3.0)
    records: list[TickRecord] = []
    src, _, ex = _source(vlm=vlm, telemetry=records.append)
    src.connect(session_id="s1")
    state = FakeRobotState()
    modes: list[str] = []
    t0 = time.perf_counter()
    for step in range(75):
        src.update(state, _frames(), step)
        modes.append(ex.state()["mode"])
    elapsed = time.perf_counter() - t0
    assert elapsed < 75 * 0.05
    assert all(r.vlm_pending for r in records)  # parse outstanding throughout
    assert all(m == "default" for m in modes)  # safe default skill meanwhile
    deadline = time.monotonic() + 10.0
    while not vlm.ready and time.monotonic() < deadline:
        time.sleep(0.05)
    assert vlm.ready
    for step in range(75, 100):
        src.update(state, _frames(), step)
        modes.append(ex.state()["mode"])
    assert any(m in ("boundary", "running", "terminal") for m in modes[75:])


def test_hud_render() -> None:
    records = []
    for step in range(50):
        stages = blank_stages()
        stages["policy"] = 11.0
        stages["executor"] = 0.8
        records.append(
            TickRecord(
                timestamp=float(step),
                episode="e",
                step=step,
                stage_ms=stages,
                active_skill="pick" if step < 25 else "place",
                active_arm="A",
                parallel_group=None,
                vlm_call=None,
                vlm_pending=False,
                detections=(
                    [{"label": "plate", "xyxy": [10, 10, 60, 60], "confidence": 0.9}]
                    if step == 49
                    else []
                ),
                goal_xyz=[0.1, -0.2, 0.05],
                action=[0.0] * 12,
                postcondition="pass" if step == 49 else None,
                recovered=None,
            )
        )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    first = render_hud(frame, records, devices={"policy": "GPU"})
    second = render_hud(frame, records, devices={"policy": "GPU"})
    assert first.shape == frame.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)  # deterministic from records alone
    assert np.any(first != frame)  # overlays actually drawn
    assert np.array_equal(render_hud(frame, []), frame)  # no records: passthrough


# ------------------------------------------------------------- abort handling


def test_task_aborted_latches_safe_hold() -> None:
    records: list[TickRecord] = []
    src, _, ex = _source(telemetry=records.append)
    src.connect(session_id="s1")
    state = FakeRobotState()
    src.update(state, _frames(), 0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ex, "tick", _boom(ex))
        action = src.update(state, _frames(), 1)
    assert src.terminal_reason == "boom"
    assert action.shape == (12,) and _in_limits(action)
    for step in (2, 3):
        latched = src.update(state, _frames(), step)
        assert np.array_equal(latched, action)
    assert records[1].postcondition == "fail:aborted:boom"
    assert records[1].recovered == "abort:boom"
    assert records[2].postcondition == "fail:aborted:boom"


def _boom(ex: Executor):
    def _raise(obs) -> None:
        raise TaskAborted("boom")

    return _raise


def test_hold_safe() -> None:
    src, _, ex = _source()
    assert np.array_equal(ex.hold_safe(), np.zeros(12))  # before the first tick
    src.connect(session_id="s1")
    src.update(FakeRobotState(), _frames(), 0)
    assert np.allclose(ex.hold_safe(), HOME_12)  # clamps to the live joints


# ------------------------------------------------------------------ telemetry


class _Tick:
    def __init__(self, step: int, stage_ms: dict) -> None:
        self.step = step
        self.stage_ms = stage_ms


class _Inference:
    def __init__(self, step: int, model: str, latency_ms: float) -> None:
        self.step = step
        self.model = model
        self.latency_ms = latency_ms


def test_telemetry_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "ticks.jsonl"
    with HudTelemetryCallback(path) as cb:
        for step in range(3):
            stages = blank_stages()
            stages["executor"] = 1.0
            cb.attach(
                TickRecord(
                    timestamp=1.0 + step,
                    episode="e",
                    step=step,
                    stage_ms=stages,
                    active_skill="home",
                    active_arm="A",
                    parallel_group=None,
                    vlm_call=None,
                    vlm_pending=False,
                    detections=[],
                    goal_xyz=None,
                    action=[0.0] * 12,
                    postcondition=None,
                    recovered=None,
                )
            )
        cb.on_inference(_Inference(1, "detector", 2.5))
        for step in range(3):
            cb.on_tick(_Tick(step, {"physics": 3.0, "render": 5.0}))
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3  # exactly one line per tick
    merged = TickRecord.from_json(lines[1])
    assert merged.step == 1
    assert merged.stage_ms["detect"] == 2.5
    assert merged.stage_ms["physics"] == 3.0
    assert merged.stage_ms["executor"] == 1.0  # source stages preserved


def test_callback_ring_buffer_serves_hud() -> None:
    cb = HudTelemetryCallback()
    for step in range(5):
        record = TickRecord(
            timestamp=0.0,
            episode="e",
            step=step,
            stage_ms=blank_stages(),
            active_skill="home",
            active_arm="A",
            parallel_group=None,
            vlm_call=None,
            vlm_pending=False,
            detections=[],
            goal_xyz=None,
            action=[0.0] * 12,
            postcondition=None,
            recovered=None,
        )
        cb.attach(record)
    assert [r.step for r in cb.recent(2)] == [3, 4]
    cb.on_tick(object())  # unfamiliar events never raise
    cb.on_inference(object())
    cb.close()


def test_package_exports() -> None:
    import dinner_table.runtime as rt

    assert rt.SkillExecutorSource is SkillExecutorSource
    assert rt.HudTelemetryCallback is HudTelemetryCallback
    assert callable(rt.render_hud)
    try:
        import importlib.util

        has_physicalai = importlib.util.find_spec("physicalai") is not None
    except (ImportError, ValueError):
        has_physicalai = False
    if not has_physicalai:
        with pytest.raises(ImportError, match="Intel stack"):
            rt.MuJoCoCamera  # noqa: B018 - lazy adapter must fail loudly here
