"""Background telemetry sampling for benchmarks and the system mode.

TelemetrySampler runs a 100 ms loop on a daemon thread and records, per
sample: per-core CPU utilization (psutil), iGPU utilization (sysfs
gpu_busy_percent, falling back to an intel_gpu_top -J subprocess), NPU
utilization (xpu-smi when present), and package power (RAPL energy deltas).
Every interface is optional: when one is missing or denied, the sample
records None for it instead of failing the sampler.
"""

from __future__ import annotations

import glob
import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Self

import psutil

SAMPLE_INTERVAL_S = 0.1
_GPU_BUSY_GLOB = "/sys/class/drm/card*/gpu_busy_percent"
_RAPL_ENERGY_GLOB = "/sys/class/powercap/intel-rapl:*/energy_uj"
_RAPL_MAX_GLOB = "/sys/class/powercap/intel-rapl:{index}/max_energy_range_uj"


@dataclass(frozen=True)
class TelemetrySample:
    """One telemetry tick; None fields mark unavailable interfaces."""

    timestamp: float
    cpu_util_per_core: list[float | None]
    igpu_util: float | None
    npu_util: float | None
    package_power_w: float | None


def rss_mb() -> float:
    """Current process RSS in MiB (benchmark memory-delta measurements)."""
    return psutil.Process().memory_info().rss / 2**20


def _read_float(path: str) -> float | None:
    try:
        with open(path, encoding="ascii") as handle:
            return float(handle.read().strip())
    except (OSError, ValueError):
        return None


class _IgpuReader:
    """iGPU utilization from sysfs, or an intel_gpu_top -J subprocess."""

    def __init__(self) -> None:
        self._sysfs_paths = sorted(glob.glob(_GPU_BUSY_GLOB))
        self._proc: subprocess.Popen | None = None
        if not self._sysfs_paths and shutil.which("intel_gpu_top") is not None:
            # NOTE: the plan sketches "intel_gpu_top -l json"; -l is plain text
            # upstream, -J is the json stream, so the adaptation lives here.
            try:
                self._proc = subprocess.Popen(
                    ["intel_gpu_top", "-J", "-s", "100"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except OSError:
                self._proc = None

    def read(self) -> float | None:
        """Percent busy of the first interface that answers, else None."""
        for path in self._sysfs_paths:
            value = _read_float(path)
            if value is not None:
                return value
        return self._read_from_process()

    def _read_from_process(self) -> float | None:
        if self._proc is None or self._proc.stdout is None:
            return None
        line = self._proc.stdout.readline()
        if not line:
            return None
        try:
            payload = json.loads(line)
            engines = payload.get("engines", {})
            for name in ("Render", "render", "RCS", "rcs"):
                busy = engines.get(name, {}).get("busy")
                if isinstance(busy, list) and len(busy) == 2 and busy[1]:
                    return 100.0 * busy[0] / busy[1]
        except (ValueError, AttributeError, TypeError):
            return None
        return None

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


class _NpuReader:
    """NPU utilization via xpu-smi when the tool exists; None otherwise."""

    def __init__(self) -> None:
        self._available = shutil.which("xpu-smi") is not None

    def read(self) -> float | None:
        if not self._available:
            return None
        try:
            proc = subprocess.run(
                ["xpu-smi", "dump", "-m", "0", "-j"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            return self._extract_utilization(json.loads(proc.stdout))
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        return None

    @staticmethod
    def _extract_utilization(payload: object) -> float | None:
        """First numeric utilization value found anywhere in the dump."""
        if isinstance(payload, dict):
            for key, value in payload.items():
                if "utilization" in key.lower() and isinstance(value, (int, float)):
                    return float(value)
                found = _NpuReader._extract_utilization(value)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = _NpuReader._extract_utilization(item)
                if found is not None:
                    return found
        return None


class _RaplReader:
    """Package power from RAPL energy counters (handles counter wraparound)."""

    def __init__(self) -> None:
        self._energy_paths = [
            path
            for path in sorted(glob.glob(_RAPL_ENERGY_GLOB))
            if path.rsplit("/", 2)[-2].count(":") == 1
        ]
        self._previous_time: float | None = None
        self._previous_uj: list[int] | None = None

    def read(self, now: float) -> float | None:
        """Watts across all package domains; None until two reads exist."""
        current: list[int] = []
        for path in self._energy_paths:
            try:
                with open(path, encoding="ascii") as handle:
                    current.append(int(handle.read().strip()))
            except (OSError, ValueError):
                return None
        if self._previous_uj is None or self._previous_time is None or now <= self._previous_time:
            self._previous_time, self._previous_uj = now, current
            return None
        delta_uj = 0.0
        for index, (prev, cur) in enumerate(zip(self._previous_uj, current)):
            step = cur - prev
            if step < 0:
                maximum = _read_float(_RAPL_MAX_GLOB.format(index=index))
                if maximum is None:
                    return None
                step += maximum + 1.0
            delta_uj += step
        watts = delta_uj / 1e6 / (now - self._previous_time)
        self._previous_time, self._previous_uj = now, current
        return watts


class TelemetrySampler:
    """Start/stop background sampling; stop() returns the collected samples."""

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._samples: list[TelemetrySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._igpu = _IgpuReader()
        self._npu = _NpuReader()
        self._rapl = _RaplReader()

    def start(self) -> None:
        """Begin sampling on the background thread (idempotent)."""
        if self._thread is not None:
            return
        psutil.cpu_percent(percpu=True, interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[TelemetrySample]:
        """Stop sampling and return every collected sample."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=3.0)
            self._thread = None
        self._igpu.close()
        return list(self._samples)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._take_sample())
            self._stop.wait(self._interval_s)

    def _take_sample(self) -> TelemetrySample:
        now = time.monotonic()
        cpu = list(psutil.cpu_percent(percpu=True))
        return TelemetrySample(
            timestamp=now,
            cpu_util_per_core=cpu,
            igpu_util=self._igpu.read(),
            npu_util=self._npu.read(),
            package_power_w=self._rapl.read(now),
        )
