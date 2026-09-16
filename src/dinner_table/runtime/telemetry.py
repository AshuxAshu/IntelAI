"""Per-tick telemetry: the single source of truth for HUD, benchmark, and eval.

One JSON line per control tick. SkillExecutorSource builds the record (it owns
the skill/arm/detection/action context a runtime callback cannot see) and hands
it to HudTelemetryCallback, which merges the runtime-reported stages
(physics/render, plus detector inference latency) and appends the merged line.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

# Stage keys every record carries, even when a stage did not run that tick
# (0.0 = not measured / not run — never missing, so consumers need no guards).
STAGE_KEYS = ("physics", "render", "detect", "policy", "executor")


@dataclass
class TickRecord:
    """Exactly the §10.7 schema. JSON-serializable via to_json()."""

    timestamp: float  # time.monotonic()
    episode: str  # episode id
    step: int  # control tick index
    stage_ms: dict[str, float]  # {"physics": …, "render": …, "detect": …, …}
    active_skill: str | None
    active_arm: str | None
    parallel_group: int | None
    vlm_call: str | None  # "parse" | "precondition" | "diagnose" | None this tick
    vlm_pending: bool
    detections: list[dict]  # [{"label": …, "xyxy": […], "confidence": …}, …]
    goal_xyz: list[float] | None
    action: list[float]  # 12 targets
    postcondition: str | None  # "pass" | "fail:<reason>" | None
    recovered: str | None  # recovery name when triggered this tick

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> TickRecord:
        return cls(**json.loads(line))


def blank_stages() -> dict[str, float]:
    """Zeroed stage map with every key present."""
    return {key: 0.0 for key in STAGE_KEYS}


class HudTelemetryCallback:
    """RobotRuntime callback: JSONL telemetry + HUD ring buffer.

    Wiring: SkillExecutorSource calls attach() once per tick; the runtime calls
    on_tick()/on_inference() with its own events. on_tick() merges the
    runtime-reported stages into the matching record and appends the merged
    line, so the file holds exactly one line per tick. Event shapes are read
    defensively (getattr with defaults) — the callback never raises on an
    unfamiliar event, it just merges what it recognizes.
    """

    def __init__(self, jsonl_path: str | Path | None = None, ring_size: int = 150) -> None:
        self._path = Path(jsonl_path) if jsonl_path is not None else None
        self._ring: deque[TickRecord] = deque(maxlen=max(1, int(ring_size)))
        self._by_step: dict[int, TickRecord] = {}
        self._flushed_through = -1
        self._file = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Persistent append handle (closed in close()); a with-block would
            # close it at the end of __init__.
            self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    # ------------------------------------------------------------ source side

    def attach(self, record: TickRecord) -> None:
        """Buffer one source-built record (called once per tick)."""
        for key in STAGE_KEYS:
            record.stage_ms.setdefault(key, 0.0)
        self._ring.append(record)
        self._by_step[record.step] = record

    # ----------------------------------------------------------- runtime side

    def on_tick(self, event: object) -> None:
        """Merge runtime stages for this tick, then append the merged line."""
        step = _event_step(event)
        stages = _event_stages(event)
        if step is not None and step in self._by_step:
            self._by_step[step].stage_ms.update(stages)
        self.flush_through(step)

    def on_inference(self, event: object) -> None:
        """Fold one model-inference latency into the matching record."""
        step = _event_step(event)
        name = getattr(event, "model", getattr(event, "stage", getattr(event, "name", None)))
        ms = _event_ms(event)
        if step is None or name is None or ms is None:
            return
        record = self._by_step.get(step)
        if record is None:
            return
        key = str(name).lower()
        key = {
            "detection": "detect",
            "detector": "detect",
            "act": "policy",
            "policy": "policy",
            "render": "render",
            "physics": "physics",
        }.get(key, key)
        if key in STAGE_KEYS:
            record.stage_ms[key] = ms

    # ------------------------------------------------------------------ sinks

    def recent(self, n: int) -> list[TickRecord]:
        """Newest-first ring buffer slice for the HUD (oldest-first order)."""
        return list(self._ring)[-max(0, int(n)) :]

    def flush_through(self, step: int | None) -> None:
        """Append every buffered record up to `step` not yet written."""
        if self._file is None:
            return
        if step is None:
            ordered = sorted(self._by_step.values(), key=lambda r: r.step)
        else:
            ordered = [
                r for r in sorted(self._by_step.values(), key=lambda r: r.step) if r.step <= step
            ]
        for record in ordered:
            if record.step > self._flushed_through:
                self._file.write(record.to_json() + "\n")
                self._flushed_through = record.step
        self._file.flush()

    def flush(self) -> None:
        """Write every buffered record (for hosts with no runtime on_tick)."""
        self.flush_through(None)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _event_step(event: object) -> int | None:
    for attr in ("step", "tick", "tick_index"):
        value = getattr(event, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _event_ms(event: object) -> float | None:
    for attr in ("latency_ms", "duration_ms", "inference_ms"):
        value = getattr(event, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    seconds = getattr(event, "duration_s", getattr(event, "latency_s", None))
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        return float(seconds) * 1000.0
    return None


def _event_stages(event: object) -> dict[str, float]:
    stages = getattr(event, "stage_ms", None)
    if isinstance(stages, dict):
        out = {}
        for key, value in stages.items():
            if key in STAGE_KEYS and isinstance(value, (int, float)):
                out[key] = float(value)
        return out
    single = getattr(event, "stage", None)
    ms = _event_ms(event)
    if isinstance(single, str) and single in STAGE_KEYS and ms is not None:
        return {single: ms}
    return {}


def now() -> float:
    return time.monotonic()
