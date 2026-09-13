"""NPU preflight tests: graceful headless behavior and on-device feasibility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "npu_preflight.py"
NPU_DEVICES = ("NPU", "NPUW:CPU,NPU")


def _ov_devices() -> tuple[str, ...]:
    try:
        import openvino as ov

        return tuple(ov.Core().available_devices)
    except Exception:  # noqa: BLE001 - no openvino means no NPU either way
        return ()


def _run_preflight(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    out = tmp_path / "npu_preflight.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), *extra],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc, json.loads(out.read_text(encoding="utf-8"))


class TestPreflightHeadless:
    @pytest.mark.slow
    def test_cpu_only_host_records_graceful_failures(self, tmp_path):
        _proc, record = _run_preflight(tmp_path, "--models", "yolo")
        devices = _ov_devices()
        assert record["npu_available"] == ("NPU" in devices)
        if "NPU" in devices:
            return
        for model in record["models"]:
            if "skipped" in model:
                continue
            for device in NPU_DEVICES:
                assert model["results"][device]["compiles"] is False
                assert model["results"][device].get("error")

    @pytest.mark.slow
    def test_every_audited_model_has_both_npu_devices(self, tmp_path):
        _proc, record = _run_preflight(tmp_path)
        assert record["models"]
        for model in record["models"]:
            if "skipped" in model:
                continue
            assert set(model["results"]) == set(NPU_DEVICES)
            assert isinstance(model["unsupported_ops_on_npu"], (list, type(None)))


class TestPreflightOnNpu:
    @pytest.mark.npu
    @pytest.mark.skipif("NPU" not in _ov_devices(), reason="no OpenVINO NPU device")
    def test_lists_per_model_feasibility(self, tmp_path):
        _proc, record = _run_preflight(tmp_path, "--models", "yolo")
        assert record["npu_available"] is True
        for model in record["models"]:
            if "skipped" in model:
                continue
            for device in NPU_DEVICES:
                assert isinstance(model["results"][device]["compiles"], bool)
            assert isinstance(model["unsupported_ops_on_npu"], list)
