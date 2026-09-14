"""Host hardware banner for benchmark reports.

probe_host collects everything a benchmark record needs for reproducibility:
OpenVINO devices with per-device properties, CPU model, power profile,
RAM, the exact Intel-stack package versions, and the NPU availability flag.
Every field is best-effort: a missing interface records None instead of
raising, so a benchmark can run on any host and still print its banner.
"""

from __future__ import annotations

import importlib.metadata as md
import logging
import platform
import subprocess
from datetime import UTC, datetime

import psutil

logger = logging.getLogger(__name__)

DEVICE_PROPERTY_KEYS = (
    "FULL_DEVICE_NAME",
    "DEVICE_TYPE",
    "DEVICE_ARCHITECTURE",
    "OPTIMIZATION_CAPABILITIES",
    "GPU_EXECUTION_UNITS_COUNT",
    "GPU_DEVICE_TOTAL_MEM_SIZE",
    "RANGE_FOR_STREAMS",
)
BANNER_FIELDS = (
    "cpu",
    "devices",
    "device_properties",
    "power_profile",
    "ram_total_gb",
    "physicalai",
    "physicalai_train",
    "openvino",
    "nncf",
    "npu_available",
    "os",
    "timestamp_utc",
)


def _json_safe(value: object) -> object:
    """Convert an OpenVINO property value into a JSON-serializable one."""
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _ov_devices() -> list[str]:
    """OpenVINO available devices; empty when openvino is unusable."""
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception:  # noqa: BLE001 - the banner must never fail the probe
        return []


def _device_properties(devices: list[str]) -> dict[str, dict]:
    """Best-effort per-device property snapshot (unreadable keys are skipped)."""
    import openvino as ov

    core = ov.Core()
    properties: dict[str, dict] = {}
    for device in devices:
        entry: dict[str, object] = {}
        for key in DEVICE_PROPERTY_KEYS:
            try:
                entry[key] = _json_safe(core.get_property(device, key))
            except Exception:  # noqa: BLE001 - property support varies per device
                logger.debug("device %s does not expose %s", device, key)
                continue
        properties[device] = entry
    return properties


def _cpu_model() -> str | None:
    """CPU model string from /proc/cpuinfo, with a portable fallback."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    name = platform.processor() or platform.machine()
    return name or None


def _power_profile() -> str | None:
    """Active power profile via powerprofilesctl; None when unavailable."""
    try:
        proc = subprocess.run(
            ["powerprofilesctl", "get"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    profile = proc.stdout.strip()
    return profile or None


def _package_version(name: str) -> str | None:
    """Installed version of a package; None when not installed."""
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def probe_host() -> dict:
    """The hardware banner stored with every benchmark report."""
    devices = _ov_devices()
    device_properties: dict[str, dict] = {}
    if devices:
        try:
            device_properties = _device_properties(devices)
        except Exception:  # noqa: BLE001 - openvino present but properties unreadable
            device_properties = {}
    return {
        "cpu": _cpu_model(),
        "devices": devices,
        "device_properties": device_properties,
        "power_profile": _power_profile(),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "physicalai": _package_version("physicalai"),
        "physicalai_train": _package_version("physicalai-train"),
        "openvino": _package_version("openvino"),
        "nncf": _package_version("nncf"),
        "npu_available": "NPU" in devices,
        "os": platform.platform(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
