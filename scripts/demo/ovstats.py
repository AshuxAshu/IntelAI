"""Live OpenVINO telemetry for the demonstration overlay.

Runs the project's real OpenVINO perception engine (``OpenvinoDetector``, the
YOLO stand-in on the OpenVINO runtime) against the live overhead camera frame
during an episode and records per-inference wall-clock latency, so the overlay
shows measurements taken in the loop rather than numbers copied from a report.

The detector is the perception path only: it does NOT drive the arms (the
teacher skills do). The overlay labels it as such. ACT reference figures are
carried separately from the committed benchmark run, and are attributed.

Failure is non-fatal by design: if OpenVINO or the stand-in weights are absent,
``LiveOpenvino`` degrades to an "unavailable" panel and the video still renders.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

YOLO_ONNX = Path("artifacts/standin/yolo11n.onnx")
YOLO_NAMES = Path("artifacts/standin/yolo11n_names.json")

# Reference ACT figures from the committed benchmark run on this host
# (bench/results_13th-gen-intel-r-core-tm-i7-13620h_20260913-060133.md).
# These are a recorded measurement, not live numbers; the overlay says so.
ACT_REFERENCE = [
    {"label": "FP32", "device": "CPU", "p50_ms": 17.61, "ips": 56.4},
    {"label": "FP16", "device": "CPU", "p50_ms": 18.50, "ips": 53.6},
    {"label": "INT8", "device": "CPU", "p50_ms": 6.11, "ips": 159.8},
]


@dataclass
class LiveOpenvino:
    """Sampled OpenVINO detector latencies plus host utilisation."""

    device: str = "CPU"
    label: str = "yolo11n"
    available: bool = False
    error: str = ""
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=240))
    n_inferences: int = 0
    n_detections: int = 0
    last_labels: str = ""
    cpu_pct: float = 0.0
    last_rss_mb: float = 0.0
    _detector: object = None
    _psutil: object = None
    _proc: object = None

    def __post_init__(self) -> None:
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process()
        except Exception:  # noqa: BLE001 - telemetry is optional
            self._psutil = None
        try:
            from dinner_table.runtime.engines import OpenvinoDetector

            names = json.loads(YOLO_NAMES.read_text(encoding="utf-8"))
            self._detector = OpenvinoDetector(YOLO_ONNX, names, self.device)
            self.available = True
        except Exception as exc:  # noqa: BLE001 - a missing stack must not break rendering
            self.error = f"{type(exc).__name__}: {exc}"[:120]

    def sample(self, frame_rgb: np.ndarray) -> None:
        """Run one real inference and record its latency; never raises."""
        if self._proc is not None:
            try:
                # First call primes the counter and reads 0.0; harmless.
                self.cpu_pct = float(self._proc.cpu_percent(interval=None)) / max(
                    1, self._psutil.cpu_count(logical=True) or 1
                )
                self.last_rss_mb = self._proc.memory_info().rss / (1024 * 1024)
            except (AttributeError, OSError):
                # psutil can lose the process (containers, races); telemetry is
                # optional and must never break a render, so keep the last value.
                pass
        if not self.available:
            return
        try:
            t0 = time.perf_counter()
            dets = self._detector.detect(frame_rgb)
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            self.n_inferences += 1
            self.n_detections += len(dets)
            if dets:
                top = sorted(dets, key=lambda d: -d.confidence)[:3]
                self.last_labels = ", ".join(f"{d.label} {d.confidence:.2f}" for d in top)
            else:
                self.last_labels = "no COCO-class detections"
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.error = f"{type(exc).__name__}: {exc}"[:120]

    # ---- derived statistics, computed from the recorded samples ----

    @property
    def last_ms(self) -> float | None:
        return self.latencies_ms[-1] if self.latencies_ms else None

    @property
    def p50_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return float(np.percentile(np.asarray(self.latencies_ms), 50))

    @property
    def p95_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return float(np.percentile(np.asarray(self.latencies_ms), 95))

    @property
    def ips(self) -> float | None:
        p50 = self.p50_ms
        return None if not p50 else 1000.0 / p50

    def series(self) -> np.ndarray:
        return np.asarray(self.latencies_ms, dtype=np.float64)
