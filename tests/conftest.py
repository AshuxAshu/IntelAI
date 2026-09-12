"""Root conftest - owned by Dev B, never modified (rule R2 applies to it too)."""

from __future__ import annotations

import importlib.util

import pytest


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _ov_devices() -> tuple[str, ...]:
    try:
        import openvino as ov

        return tuple(ov.Core().available_devices)
    except Exception:  # noqa: BLE001 - device probe must never break conftest import on any host
        return ()


DEVICES = _ov_devices()
requires_openvino = pytest.mark.skipif(not _has("openvino"), reason="openvino not installed")
requires_physicalai = pytest.mark.skipif(not _has("physicalai"), reason="physicalai not installed")
requires_ov_gpu = pytest.mark.skipif("GPU" not in DEVICES, reason="no OpenVINO GPU device")
requires_npu = pytest.mark.skipif("NPU" not in DEVICES, reason="no OpenVINO NPU device")


@pytest.fixture
def artifacts_dir(tmp_path):
    out = tmp_path / "artifacts"
    out.mkdir()
    return out
