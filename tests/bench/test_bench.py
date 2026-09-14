"""Benchmark tests: schema validation, host probe, and the CI smoke run."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import jsonschema
import pytest

from dinner_table.bench.device_probe import BANNER_FIELDS, probe_host

BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"
SMOKE_LIMIT_S = 90.0

EXAMPLE_RECORD = {
    "schema": "bench/1.2",
    "host": {
        "cpu": "Intel Core Ultra 7 258V",
        "devices": ["CPU", "GPU", "NPU"],
        "power_profile": "balanced",
        "openvino": "2025.1",
    },
    "runs": [
        {
            "model": "act",
            "device": "GPU",
            "precision": "INT8",
            "latency_ms": {"p50": 2.8, "p90": 3.1, "p95": 3.6, "p99": 4.4},
            "throughput_ips": 341.2,
            "cold_compile_s": 8.2,
            "weights": "INT8",
            "activations": "INT8",
            "rss_delta_mb": 96,
            "parity": {"closed_loop_success_delta_pp": -1.0, "verdict": "keep"},
        },
        {
            "model": "vlm",
            "device": "NPUW:CPU,NPU",
            "precision": "INT8",
            "prefill_tok_s": 180.4,
            "decode_tok_s": 24.1,
            "e2e_128tok_s": 6.1,
        },
    ],
    "system": {
        "control_fps_target": 25.0,
        "control_fps_achieved": 25.0,
        "stage_ms_avg": {
            "physics": 3.1,
            "render": 4.2,
            "detect": 0.9,
            "policy": 3.0,
            "executor": 0.2,
        },
        "power_w_avg": 28.4,
        "igpu_util_avg": 0.41,
        "npu_util_avg": 0.18,
        "success_rate": 0.9,
        "seeds": 10,
        "retries_per_skill_avg": 0.21,
    },
}


def _validate(record: dict) -> None:
    schema = json.loads((BENCH_DIR / "schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(instance=record, schema=schema)


def _train_stack_available() -> bool:
    try:
        return importlib.util.find_spec("physicalai.train") is not None
    except (ImportError, ValueError):
        return False


class TestBenchSchema:
    @pytest.mark.fast
    def test_example_record_validates(self):
        _validate(EXAMPLE_RECORD)

    @pytest.mark.fast
    def test_example_record_missing_field_rejected(self):
        record = json.loads(json.dumps(EXAMPLE_RECORD))
        del record["runs"][0]["latency_ms"]["p99"]
        with pytest.raises(jsonschema.ValidationError):
            _validate(record)

    @pytest.mark.fast
    def test_committed_results_validate(self):
        records = sorted(BENCH_DIR.glob("results_*.json"))
        if not records:
            pytest.skip("no committed benchmark results yet")
        for path in records:
            _validate(json.loads(path.read_text(encoding="utf-8")))


class TestDeviceProbe:
    @pytest.mark.fast
    def test_banner_has_every_field(self):
        banner = probe_host()
        for field in BANNER_FIELDS:
            assert field in banner, f"probe banner is missing {field}"
        assert banner["ram_total_gb"] > 0
        assert banner["npu_available"] == ("NPU" in banner["devices"])

    @pytest.mark.fast
    def test_devices_mirror_openvino(self):
        if importlib.util.find_spec("openvino") is None:
            pytest.skip("openvino not installed")
        import openvino as ov

        banner = probe_host()
        assert banner["devices"] == list(ov.Core().available_devices)
        assert "CPU" in banner["devices"]


@pytest.fixture(scope="module")
def public_act_export(tmp_path_factory):
    """The public ACT export, trained and exported on demand when absent."""
    export_dir = Path("artifacts/act_public/fp16")
    if (export_dir / "manifest.json").is_file():
        return export_dir
    if not _train_stack_available():
        pytest.skip("no public ACT export and no training stack to build one")
    from dinner_table.policies.export_quantize import export_openvino
    from dinner_table.policies.studio_train import run as train_run

    root = tmp_path_factory.mktemp("bench_smoke")
    config = root / "pusht_act.yaml"
    config.write_text(
        "model:\n"
        "  class_path: physicalai.policies.ACT\n"
        "  init_args: {chunk_size: 25, n_action_steps: 25, image_size: [224, 224], use_vae: true, optimizer_lr: 2.5e-4}\n"
        "data:\n"
        "  class_path: physicalai.data.lerobot.LeRobotDataModule\n"
        "  init_args:\n"
        "    repo_id: lerobot/pusht\n"
        "    data_format: physicalai\n"
        "    train_batch_size: 8\n"
        "    episodes: [0, 1, 2]\n"
        "    video_backend: pyav\n"
        "trainer:\n"
        "  accelerator: cpu\n"
        "  devices: 1\n"
        "  max_steps: 50\n",
        encoding="utf-8",
    )
    meta_path = train_run(
        config,
        overrides=["--trainer.default_root_dir", "artifacts/act_public/experiments"],
        smoke=True,
        runs_dir="artifacts/act_public/runs",
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checkpoints = sorted(Path(meta["experiment_dir"]).glob("checkpoints/*.ckpt"))
    assert checkpoints, "the training run produced no checkpoint"
    export_openvino(checkpoints[0], export_dir)
    return export_dir


class TestBenchSmoke:
    @pytest.mark.nightly
    @pytest.mark.openvino
    def test_smoke_run_under_90s(self, public_act_export, tmp_path):
        command = [
            sys.executable,
            "-m",
            "dinner_table.bench.bench_intel",
            "--models",
            "act",
            "--devices",
            "CPU",
            "--precisions",
            "fp16",
            "--mode",
            "micro",
            "--iters",
            "5",
            "--out",
            str(tmp_path),
        ]
        start = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert elapsed < SMOKE_LIMIT_S, f"smoke run took {elapsed:.1f} s"
        results = sorted(tmp_path.glob("results_*.json"))
        assert len(results) == 1
        record = json.loads(results[0].read_text(encoding="utf-8"))
        _validate(record)
        assert record["runs"][0]["model"] == "act"
        assert record["runs"][0]["latency_ms"]["p50"] > 0

    @pytest.mark.slow
    def test_system_mode_clear_error(self, tmp_path):
        command = [
            sys.executable,
            "-m",
            "dinner_table.bench.bench_intel",
            "--mode",
            "system",
            "--out",
            str(tmp_path),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        assert proc.returncode != 0
        assert "system mode is not implemented" in (proc.stdout + proc.stderr)
