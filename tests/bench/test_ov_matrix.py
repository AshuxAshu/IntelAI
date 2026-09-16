"""Optimization-matrix tests: reporting, gates, and device resolution.

The end-to-end matrix needs a checkpoint and the full Intel stack, so the
expensive path is a nightly smoke; the fast tests pin the pieces that decide
what a reader sees - the table's device columns and dash policy, the parity
verdicts, and the JSON/CSV/markdown record shape.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dinner_table.bench import ov_matrix
from dinner_table.bench.ov_matrix import (
    COLUMNS,
    DASH,
    FP16_REL_ERROR_LIMIT,
    FP32_REL_ERROR_LIMIT,
    INT8_NORMALIZED_MSE_LIMIT,
    MatrixError,
    main,
    parity_verdict,
    render_markdown,
    resolve_matrix_devices,
    write_outputs,
)

requires_openvino = pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None, reason="openvino not installed"
)


def _record() -> dict:
    """A hand-built record with one measured, one absent, and one failed cell."""
    return {
        "schema": "ov-matrix/1.0",
        "host": {"cpu": "Test CPU", "os": "Test OS", "devices": ["CPU", "GPU.0"], "openvino": "2026.1"},
        "devices": {"CPU": "CPU", "iGPU": "GPU.0", "NPU": None},
        "checkpoint": "/tmp/fake.ckpt",
        "notes": ["NPU is not available on this host"],
        "rows": [
            {
                "label": "OpenVINO FP32",
                "backend": "openvino",
                "precision": "FP32",
                "variant": "fp32",
                "model_size_bytes": 68 * 2**20,
                "weights": "FP32",
                "activations": "FP32",
                "rel_error_vs_fp32": 0.0,
                "normalized_mse_vs_fp32": 0.0,
                "rel_error_vs_pytorch": 1e-6,
                "cells": {
                    "CPU": {"device_id": "CPU", "latency_ms": {"p50": 17.6, "p90": 19.0, "p95": 19.7, "p99": 22.1}, "throughput_ips": 56.4, "cold_compile_s": 0.38, "rss_delta_mb": 348.6, "error": ""},
                    "iGPU": {"device_id": "GPU.0", "latency_ms": {"p50": 20.4, "p90": 22.2, "p95": 24.1, "p99": 27.5}, "throughput_ips": 49.1, "cold_compile_s": 0.67, "rss_delta_mb": 165.5, "error": ""},
                    "NPU": {"device_id": None, "latency_ms": None, "throughput_ips": None, "cold_compile_s": None, "rss_delta_mb": None, "error": "absent"},
                },
            },
            {
                "label": "OpenVINO INT8 (NNCF)",
                "backend": "openvino",
                "precision": "INT8",
                "variant": "int8_ptq",
                "model_size_bytes": 17 * 2**20,
                "weights": "INT8",
                "activations": "FP32",
                "rel_error_vs_fp32": 3e-2,
                "normalized_mse_vs_fp32": 1e-3,
                "rel_error_vs_pytorch": 3e-2,
                "cells": {
                    "CPU": {"device_id": "CPU", "latency_ms": {"p50": 7.4, "p90": 8.0, "p95": 8.4, "p99": 9.1}, "throughput_ips": 135.0, "cold_compile_s": 0.52, "rss_delta_mb": 126.2, "error": ""},
                    "iGPU": {"device_id": "GPU.0", "latency_ms": None, "throughput_ips": None, "cold_compile_s": None, "rss_delta_mb": None, "error": "compile failed"},
                    "NPU": {"device_id": None, "latency_ms": None, "throughput_ips": None, "cold_compile_s": None, "rss_delta_mb": None, "error": "absent"},
                },
            },
        ],
        "closed_loop": None,
    }


class TestParityVerdicts:
    @pytest.mark.fast
    def test_fp32_and_fp16_gate_on_relative_error(self):
        assert parity_verdict("fp32", FP32_REL_ERROR_LIMIT / 10, 0.0)[0]
        assert not parity_verdict("fp32", FP32_REL_ERROR_LIMIT * 10, 0.0)[0]
        assert parity_verdict("fp16", FP16_REL_ERROR_LIMIT / 10, 0.0)[0]
        assert not parity_verdict("fp16", FP16_REL_ERROR_LIMIT * 10, 0.0)[0]

    @pytest.mark.fast
    def test_int8_gates_on_normalized_mse(self):
        assert parity_verdict("int8_ptq", 1.0, INT8_NORMALIZED_MSE_LIMIT / 10)[0]
        assert not parity_verdict("int8_ptq", 1.0, INT8_NORMALIZED_MSE_LIMIT * 10)[0]
        assert parity_verdict("int8_weights", 1.0, INT8_NORMALIZED_MSE_LIMIT / 10)[0]


class TestMarkdown:
    @pytest.mark.fast
    def test_table_has_the_requested_shape(self):
        text = render_markdown(_record())
        header = "| ACT policy | CPU | iGPU | NPU | Action error vs FP32 |"
        assert header in text
        assert f"| OpenVINO FP32 | 17.60 ms | 20.40 ms | {DASH} | reference |" in text
        assert f"| OpenVINO INT8 (NNCF) | 7.40 ms | {DASH} | {DASH} | 3.000e-02 |" in text

    @pytest.mark.fast
    def test_absent_and_failed_cells_are_distinguishable(self):
        text = render_markdown(_record())
        detail = text.split("## Detail", 1)[1]
        assert "compile failed" not in detail  # errors are not printed as timings
        assert DASH in detail

    @pytest.mark.fast
    def test_notes_and_gates_are_carried(self):
        text = render_markdown(_record())
        assert "NPU is not available on this host" in text
        assert "## Parity gates" in text
        assert "reference" in text


class TestOutputs:
    @pytest.mark.fast
    def test_writes_json_csv_markdown(self, tmp_path):
        paths = write_outputs(_record(), tmp_path, "stem")
        names = sorted(path.name for path in paths)
        assert names == ["stem.csv", "stem.json", "stem.md"]
        record = json.loads((tmp_path / "stem.json").read_text(encoding="utf-8"))
        assert record["schema"] == "ov-matrix/1.0"
        rows = (tmp_path / "stem.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 1 + 2 * len(COLUMNS)
        assert rows[0].startswith("row,backend,precision,variant,device,device_id")


class TestDeviceResolution:
    @pytest.mark.fast
    @requires_openvino
    def test_intel_igpu_wins_the_igpu_column(self):
        columns, extras = resolve_matrix_devices()
        assert set(columns) == set(COLUMNS)
        assert columns["CPU"] is not None, "every OpenVINO host exposes CPU"
        for device in extras:
            assert device != columns["iGPU"]
        if "NPU" not in _available():
            assert columns["NPU"] is None

    @pytest.mark.fast
    @requires_openvino
    def test_nvidia_gpu_is_not_the_igpu_column(self):
        columns, _extras = resolve_matrix_devices()
        for device in _available():
            name = _device_name(device)
            if device.split(".")[0] == "GPU" and "NVIDIA" in name:
                assert columns["iGPU"] != device


class TestClosedLoopSeam:
    @pytest.mark.fast
    def test_policy_closed_loop_is_an_explicit_error(self):
        with pytest.raises(MatrixError, match="policy in the simulation loop"):
            ov_matrix.closed_loop_success()


class TestCli:
    @pytest.mark.fast
    def test_policy_closed_loop_fails_fast(self, tmp_path, capsys):
        # Must not provision or benchmark before rejecting an impossible mode.
        assert main(["--closed-loop", "policy", "--out", str(tmp_path), "--no-auto-train"]) == 2
        assert "not implemented" in capsys.readouterr().out

    @pytest.mark.fast
    def test_rejects_nonpositive_iters(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--iters", "0", "--out", str(tmp_path), "--no-auto-train"])


class TestMeasurementPath:
    """The real InferenceModel wiring, timed cheaply on the committed export."""

    @pytest.mark.nightly
    @requires_openvino
    def test_rung_timing_reports_positive_p50(self):
        export_dir = Path("artifacts/act_public/fp16")
        if not (export_dir / "manifest.json").is_file():
            pytest.skip("no exported ACT rung; run scripts/benchmark_openvino.py first")
        pytest.importorskip("physicalai")
        result = ov_matrix._bench_rung(export_dir, "CPU", iters=5, warmup=2, seed=0)
        assert result["latency_ms"]["p50"] > 0
        assert result["num_iters"] == 5
        assert result["cold_compile_s"] >= 0


def _available() -> list[str]:
    import openvino as ov

    return list(ov.Core().available_devices)


def _device_name(device: str) -> str:
    import openvino as ov

    try:
        return str(ov.Core().get_property(device, "FULL_DEVICE_NAME"))
    except Exception:  # noqa: BLE001 - a missing name is not a test failure
        return ""
