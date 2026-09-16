"""OpenVINO optimization matrix: one command in, one table out.

Rows are policy variants (PyTorch FP32, OpenVINO FP32, FP16, INT8 PTQ,
INT8 weights); columns are the Intel devices this host actually exposes
(CPU, iGPU, NPU) plus the column that keeps the speedups honest: each rung's
action-chunk error against the FP32 reference on identical inputs. A row's
latency is only meaningful next to its quality number - the gates in
docs/OPTIMIZATION.md exist so a faster row that silently changes behaviour
cannot ship.

Everything is provisioned on demand from a single checkpoint, and that
checkpoint is trained on demand too (the public stand-in) when none is
supplied, so a fresh clone reproduces the table with one command. Devices the
host does not expose are reported as absent, never skipped silently.

The closed-loop task-success column ("mug placed over N seeds") needs a policy
in the simulation loop, which does not exist yet - see closed_loop_success's
seam. Until it does, teacher_oracle_success measures the system end to end with
the privileged teacher driving it; that number is precision-independent and is
labelled as such wherever it appears.

# NOTE: this module runs past the ~300-line guideline - one cohesive matrix
# builder plus its provisioning and parity paths; splitting would create
# artificial seams between steps that share one checkpoint and one device list.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov

from dinner_table.bench.device_probe import probe_host
from dinner_table.bench.telemetry_sampler import rss_mb
from dinner_table.config import DinnerTableError
from dinner_table.policies.export_quantize import (
    RUNG_FP16,
    RUNG_FP32,
    RUNG_INT8_PTQ,
    RUNG_INT8_WEIGHTS,
    compress_int8_weights,
    export_fp32_reference,
    export_openvino,
    ptq_int8,
    random_inputs,
    read_ir,
)

SCHEMA_VERSION = "ov-matrix/1.0"
DEFAULT_OUT = Path("bench/ov_matrix")
DEFAULT_ITERS = 200
DEFAULT_WARMUP = 10
PARITY_SAMPLES = 100
CALIBRATION_SAMPLES = 300
STANDIN_CONFIG = Path("configs/physicalai/act_standin.yaml")
INTEL_GPU_PREFIX = "Intel"

# The parity gates the plan sets (docs/OPTIMIZATION.md): a row above its limit
# is reported as failing, not averaged away.
FP32_REL_ERROR_LIMIT = 1e-3
FP16_REL_ERROR_LIMIT = 5e-3
INT8_NORMALIZED_MSE_LIMIT = 5e-3

# (label, kind, rung directory name, precision, variant)
ROWS: tuple[tuple[str, str, str | None, str, str], ...] = (
    ("PyTorch FP32", "pytorch", None, "FP32", "pytorch"),
    ("OpenVINO FP32", "openvino", RUNG_FP32, "FP32", "fp32"),
    ("OpenVINO FP16", "openvino", RUNG_FP16, "FP16", "fp16"),
    ("OpenVINO INT8 (NNCF)", "openvino", RUNG_INT8_PTQ, "INT8", "int8_ptq"),
    ("OpenVINO INT8 weights", "openvino", RUNG_INT8_WEIGHTS, "INT8", "int8_weights"),
)
COLUMNS = ("CPU", "iGPU", "NPU")
DASH = "\u2013"  # the en dash the table uses for an absent device

logger = logging.getLogger(__name__)


class MatrixError(DinnerTableError):
    """Matrix provisioning or measurement failure the user must act on."""


@dataclass
class Cell:
    """One row x device measurement (or the failure that replaced it)."""

    device: str
    device_id: str | None
    latency_ms: dict[str, float] | None = None
    throughput_ips: float | None = None
    cold_compile_s: float | None = None
    rss_delta_mb: float | None = None
    num_iters: int | None = None
    error: str = ""


@dataclass
class Row:
    """One policy variant: per-device cells plus its quality against FP32."""

    label: str
    backend: str
    precision: str
    variant: str
    model_size_bytes: int | None = None
    cells: dict[str, Cell] = field(default_factory=dict)
    rel_error_vs_fp32: float | None = None
    normalized_mse_vs_fp32: float | None = None
    rel_error_vs_pytorch: float | None = None
    weights: str | None = None
    activations: str | None = None


def _dummy_inputs() -> Iterator[dict]:
    """Infinite empty-sample stream for callables that take no real input."""
    while True:
        yield {}


def _percentiles(times_s: list[float]) -> dict[str, float]:
    """p50/p90/p95/p99 in milliseconds from raw per-call seconds."""
    array = np.asarray(times_s, dtype=np.float64) * 1000.0
    values = np.percentile(array, [50, 90, 95, 99])
    return {name: round(float(value), 3) for name, value in zip(("p50", "p90", "p95", "p99"), values)}


def _timed(fn: Callable[[dict], object], inputs: Iterator[dict], iters: int, warmup: int) -> dict:
    """Timed loop built on physicalai's InferenceLatencyBenchmark.

    Mirrors bench_intel's methodology so a number from the matrix and a number
    from `make bench` are the same measurement under the same warmup policy.
    """
    from physicalai.benchmark.performance import InferenceLatencyBenchmark

    times: list[float] = []

    def recorded(sample: dict) -> object:
        start = time.perf_counter()
        output = fn(sample)
        times.append(time.perf_counter() - start)
        return output

    bench = InferenceLatencyBenchmark(max_iters=iters, warmup_iters=warmup)
    metrics = bench.run(recorded, inputs)
    measured = times[warmup:]
    if not measured:
        raise MatrixError("the latency benchmark produced no measured iterations")
    if metrics["num_iters"] != len(measured):
        raise MatrixError(
            f"recording mismatch: {metrics['num_iters']} benchmark iterations, "
            f"{len(measured)} recorded"
        )
    return {
        "latency_ms": _percentiles(measured),
        "throughput_ips": round(len(measured) / sum(measured), 2),
        "num_iters": len(measured),
    }


# --------------------------------------------------------------------------
# Provisioning
# --------------------------------------------------------------------------


def discover_checkpoint(explicit: Path | None) -> Path | None:
    """Explicit checkpoint, else the newest one under the known run dirs."""
    if explicit is not None:
        if not explicit.is_file():
            raise MatrixError(f"checkpoint not found: {explicit}")
        return explicit
    candidates: list[Path] = []
    for root in (Path("experiments"), Path("runs"), Path("artifacts")):
        candidates.extend(root.rglob("checkpoints/*.ckpt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def train_standin(out_root: Path) -> Path:
    """50-step public-dataset training run, so a fresh clone has a checkpoint.

    This is the stand-in (lerobot/pusht), not the dinner-table policy: it exists
    to make the matrix reproducible before our own dataset lands, and every
    report says so.
    """
    from dinner_table.policies.studio_train import run as train_run

    if not STANDIN_CONFIG.is_file():
        raise MatrixError(f"stand-in training config missing: {STANDIN_CONFIG}")
    logger.info("no checkpoint found; training the public stand-in (50 steps)")
    meta_path = train_run(
        STANDIN_CONFIG,
        overrides=["--trainer.default_root_dir", str(out_root / "experiments")],
        smoke=True,
        runs_dir=out_root / "runs",
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checkpoints = sorted(Path(meta["experiment_dir"]).glob("checkpoints/*.ckpt"))
    if not checkpoints:
        raise MatrixError("the stand-in training run produced no checkpoint")
    return checkpoints[-1]


def provision_ladder(
    ckpt: Path, root: Path, rungs: tuple[str, ...], force: bool = False, seed: int = 0
) -> dict[str, Path]:
    """Export every requested rung from one checkpoint (cached per rung.json).

    Returns rung name -> export directory. The export helpers return the rung's
    own rung.json path, which is not what callers need, so the directory is
    taken from the loop rather than the helper's return value.
    """
    provisions = {
        RUNG_FP16: lambda out: export_openvino(ckpt, out),
        RUNG_FP32: lambda out: export_fp32_reference(ckpt, out),
        RUNG_INT8_WEIGHTS: lambda out: compress_int8_weights(_seeded_fp32(ckpt, out)),
        RUNG_INT8_PTQ: lambda out: ptq_int8(
            _seeded_fp32(ckpt, out), random_inputs(out, CALIBRATION_SAMPLES, seed=seed)
        ),
    }
    provisioned: dict[str, Path] = {}
    for rung in rungs:
        out = root / rung
        if (out / "rung.json").is_file() and not force:
            logger.info("rung %s: reusing %s", rung, out)
            provisioned[rung] = out
            continue
        start = time.perf_counter()
        provisions[rung](out)
        provisioned[rung] = out
        logger.info("rung %s: provisioned in %.1f s", rung, time.perf_counter() - start)
    return provisioned


def _seeded_fp32(ckpt: Path, out: Path) -> Path:
    """FP32 reference export, the base the quantization rungs are built on."""
    export_fp32_reference(ckpt, out)
    return out


def _ir_path(export_dir: Path) -> Path:
    """IR path from the manifest's own artifact table (never a guessed filename)."""
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    path = export_dir / manifest["model"]["artifacts"]["openvino"]
    if not path.is_file():
        raise MatrixError(f"manifest artifact {path} is missing")
    return path


def model_size_bytes(export_dir: Path) -> int:
    """On-disk size of the exported IR (.xml + .bin)."""
    return sum(path.stat().st_size for path in export_dir.iterdir() if path.suffix in (".xml", ".bin"))


# --------------------------------------------------------------------------
# Device resolution
# --------------------------------------------------------------------------


def resolve_matrix_devices() -> tuple[dict[str, str | None], list[str]]:
    """Map the table's columns to real device ids, plus any extra devices.

    iGPU is the first Intel GPU OpenVINO exposes (Core Ultra and the i7 both
    report it; a discrete NVIDIA GPU is deliberately not a column). Missing
    columns resolve to None so the report can print an explicit dash.
    """
    try:
        available = list(ov.Core().available_devices)
    except Exception as exc:
        raise MatrixError(f"OpenVINO could not list devices: {exc}") from exc
    core = ov.Core()
    columns: dict[str, str | None] = {"CPU": None, "iGPU": None, "NPU": None}
    extras: list[str] = []
    for device in available:
        base = device.split(".")[0]
        if base == "CPU":
            columns["CPU"] = device
        elif base == "GPU":
            name = ""
            try:
                name = str(core.get_property(device, "FULL_DEVICE_NAME"))
            except Exception:  # noqa: BLE001 - an unreadable name is not a rejection
                logger.debug("device %s exposes no FULL_DEVICE_NAME", device)
            if INTEL_GPU_PREFIX in name:
                if columns["iGPU"] is None:
                    columns["iGPU"] = device
                else:
                    extras.append(device)
            else:
                extras.append(device)
        elif base == "NPU":
            columns["NPU"] = device
        else:
            extras.append(device)
    return columns, extras


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _bench_rung(rung_dir: Path, device: str, iters: int, warmup: int, seed: int) -> dict:
    """One export rung on one device through physicalai's InferenceModel."""
    from physicalai.benchmark.performance.input_sources import RandomInputSource
    from physicalai.inference import InferenceModel

    rss_before = rss_mb()
    start = time.perf_counter()
    model = InferenceModel(rung_dir, device=device)
    cold_compile_s = round(time.perf_counter() - start, 2)
    source = RandomInputSource(
        model.input_features, seed=seed, num_samples=iters + warmup + 8
    )
    stats = _timed(model, source, iters, warmup)
    return {
        **stats,
        "cold_compile_s": cold_compile_s,
        "rss_delta_mb": round(rss_mb() - rss_before, 1),
    }


def _bench_pytorch(ckpt: Path, reference_rung: Path, iters: int, warmup: int, seed: int) -> dict:
    """PyTorch on the CPU - the row the OpenVINO speedups are quoted against.

    Deliberately CPU-only: the requested table quotes PyTorch FP32 against the
    OpenVINO CPU/iGPU/NPU cells, and the project's torch comes from the CPU
    wheel index regardless of host.
    """
    import torch
    from physicalai.policies import ACT

    rss_before = rss_mb()
    start = time.perf_counter()
    policy = ACT.load_from_checkpoint(str(ckpt))
    policy.eval()
    cold_compile_s = round(time.perf_counter() - start, 2)
    shapes = _input_shapes(reference_rung)
    rng = np.random.default_rng(seed)
    batch = {
        name: torch.from_numpy(
            rng.uniform(0.0, 1.0, shape).astype(np.float32)
            if len(shape) > 2
            else rng.standard_normal(shape).astype(np.float32)
        )
        for name, shape in shapes.items()
    }
    with torch.no_grad():
        stats = _timed(lambda _sample: policy.model(batch), _dummy_inputs(), iters, warmup)
    return {**stats, "cold_compile_s": cold_compile_s, "rss_delta_mb": round(rss_mb() - rss_before, 1)}


def _input_shapes(reference_rung: Path) -> dict[str, tuple[int, ...]]:
    """Manifest-shaped inputs of the exported model (static after tracing)."""
    model = read_ir(_ir_path(reference_rung))
    return {
        inp.get_any_name(): tuple(dim.get_length() for dim in inp.partial_shape)
        for inp in model.inputs
    }


def action_chunks(export_dir: Path, inputs: list[dict], device: str) -> list[np.ndarray]:
    """The rung's action chunk for each sample, in order."""
    from physicalai.inference import InferenceModel

    model = InferenceModel(export_dir, device=device)
    return [np.asarray(model.predict_action_chunk(sample), dtype=np.float64) for sample in inputs]


def torch_action_chunks(ckpt: Path, inputs: list[dict]) -> list[np.ndarray]:
    """The checkpoint's action chunk for each sample, in order."""
    import torch
    from physicalai.policies import ACT

    policy = ACT.load_from_checkpoint(str(ckpt))
    policy.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for sample in inputs:
            batch = {name: torch.from_numpy(np.asarray(array)) for name, array in sample.items()}
            out = policy.model(batch)
            actions = out[0] if isinstance(out, tuple) else out
            chunks.append(np.asarray(actions.numpy()[0], dtype=np.float64))
    return chunks


def _max_relative_error(observed: list[np.ndarray], reference: list[np.ndarray]) -> float:
    worst = 0.0
    for actual, expected in zip(observed, reference):
        error = np.linalg.norm(actual - expected) / (np.linalg.norm(expected) + 1e-12)
        worst = max(worst, float(error))
    return worst


def _normalized_mse(observed: list[np.ndarray], reference: list[np.ndarray]) -> float:
    """Mean squared action error normalized by the reference magnitude."""
    squared = 0.0
    scale = 0.0
    for actual, expected in zip(observed, reference):
        squared += float(((actual - expected) ** 2).sum())
        scale += float((expected**2).sum())
    return squared / (scale + 1e-12)


def parity_verdict(variant: str, rel_error: float, normalized_mse: float) -> tuple[bool, str]:
    """Pass/fail for one row against the gate its precision has to meet."""
    if variant == "fp32":
        passed = rel_error < FP32_REL_ERROR_LIMIT
        return passed, f"rel err {rel_error:.2e} < {FP32_REL_ERROR_LIMIT:g}"
    if variant == "fp16":
        passed = rel_error < FP16_REL_ERROR_LIMIT
        return passed, f"rel err {rel_error:.2e} < {FP16_REL_ERROR_LIMIT:g}"
    passed = normalized_mse < INT8_NORMALIZED_MSE_LIMIT
    return passed, f"norm MSE {normalized_mse:.2e} < {INT8_NORMALIZED_MSE_LIMIT:g}"


# --------------------------------------------------------------------------
# Closed loop
# --------------------------------------------------------------------------


def teacher_oracle_success(seeds: list[int], profile: str, graph: str = "mug_setting") -> dict:
    """End-to-end mug success over N seeds with the privileged teacher driving.

    Precision never enters this loop: the teacher emits joint targets and the
    sim executes them, so every row would read the same rate. It is recorded as
    a system smoke measurement, never as quantization evidence.
    """
    from dinner_table.eval.teacher_eval import evaluate

    report = evaluate([graph], seeds, profile)
    entry = report["graphs"][graph]
    return {
        "driver": "teacher-oracle",
        "precision_dependent": False,
        "graph": graph,
        "profile": profile,
        "seeds": len(seeds),
        "successes": sum(episode["success"] for episode in entry["episodes"]),
        "success_rate": round(entry["success_rate"], 4),
        "first_attempt_rate": round(entry["first_attempt_rate"], 4),
        "per_seed": [
            {"seed": episode["seed"], "success": episode["success"]} for episode in entry["episodes"]
        ],
    }


def closed_loop_success(*_args, **_kwargs) -> dict:
    """Seam for the learned-policy closed-loop success column.

    The real column needs an ACT checkpoint for the dinner scene plus a policy
    in the loop; neither exists yet (the teacher skills are privileged, and the
    exported stand-in's state input is not our 35-dim conditioning vector).
    When that harness lands it plugs in here and the matrix gains a genuine
    precision-dependent success column - do not fake it with the teacher.
    """
    raise MatrixError(
        "closed-loop success needs a policy in the simulation loop, which is not "
        "implemented: the demo runs are driven by privileged teacher skills, not a "
        "learned policy. Use --closed-loop teacher for the system smoke number, or "
        "--closed-loop none to omit the column."
    )


# --------------------------------------------------------------------------
# Matrix assembly
# --------------------------------------------------------------------------


def build_matrix(args: argparse.Namespace) -> dict:
    """Provision, measure every cell, compute parity, and assemble the record."""
    columns, extras = resolve_matrix_devices()
    notes: list[str] = []
    for name, device in columns.items():
        if device is None:
            notes.append(f"{name} is not available on this host")
    notes.append(
        "the action-parity column is each rung against the OpenVINO FP32 reference on "
        "identical inputs; it is the quantization-quality check, not a task-success rate"
    )

    ckpt = discover_checkpoint(args.ckpt)
    if ckpt is None:
        if not args.auto_train:
            raise MatrixError(
                "no checkpoint found; pass --ckpt PATH or drop --no-auto-train to train the "
                "public stand-in"
            )
        ckpt = train_standin(args.out)
    else:
        logger.info("checkpoint: %s", ckpt)
    notes.append(f"checkpoint: {ckpt}")

    rungs = tuple(rung for _, kind, rung, _, _ in ROWS if kind == "openvino" and rung)
    provisioned = provision_ladder(ckpt, args.exports, rungs, force=args.force, seed=args.seed)

    shapes = _input_shapes(provisioned[RUNG_FP32])
    inputs = random_inputs(provisioned[RUNG_FP32], PARITY_SAMPLES, seed=args.seed)
    notes.append(f"inputs: {', '.join(f'{name}{list(shape)}' for name, shape in shapes.items())}")

    reference = action_chunks(provisioned[RUNG_FP32], inputs, "CPU")
    try:
        torch_reference = torch_action_chunks(ckpt, inputs)
    except Exception as exc:  # noqa: BLE001 - framework parity is best-effort
        logger.warning("pytorch parity unavailable: %s", exc)
        torch_reference = None

    rows: list[Row] = []
    for label, kind, rung, precision, variant in ROWS:
        row = Row(label=label, backend=kind, precision=precision, variant=variant)
        if kind == "pytorch":
            row.rel_error_vs_fp32 = _max_relative_error(torch_reference, reference) if torch_reference else None
            row.normalized_mse_vs_fp32 = _normalized_mse(torch_reference, reference) if torch_reference else None
            for column in COLUMNS:
                device = columns[column]
                if column != "CPU":
                    # PyTorch here is the CPU/XPU baseline; the CPU wheel has no XPU.
                    row.cells[column] = Cell(device=column, device_id=device, error="not measured")
                    continue
                row.cells[column] = _measure(
                    lambda: _bench_pytorch(ckpt, provisioned[RUNG_FP32], args.iters, args.warmup, args.seed),
                    column,
                    device,
                )
            rows.append(row)
            continue

        assert rung is not None
        export_dir = provisioned[rung]
        row.model_size_bytes = model_size_bytes(export_dir)
        chunks = action_chunks(export_dir, inputs, "CPU")
        row.rel_error_vs_fp32 = _max_relative_error(chunks, reference)
        row.normalized_mse_vs_fp32 = _normalized_mse(chunks, reference)
        if torch_reference is not None:
            row.rel_error_vs_pytorch = _max_relative_error(chunks, torch_reference)
        weights, activations = _ir_precisions(export_dir)
        row.weights, row.activations = weights, activations
        for column in COLUMNS:
            device = columns[column]
            if device is None:
                row.cells[column] = Cell(device=column, device_id=None, error="absent")
                continue
            row.cells[column] = _measure(
                lambda d=device, r=export_dir: _bench_rung(r, d, args.iters, args.warmup, args.seed),
                column,
                device,
            )
        rows.append(row)

    record: dict = {
        "schema": SCHEMA_VERSION,
        "host": probe_host(),
        "devices": columns,
        "extra_devices": extras,
        "checkpoint": str(ckpt),
        "notes": notes,
        "rows": [_row_dict(row) for row in rows],
        "closed_loop": None,
    }
    if args.closed_loop == "teacher":
        record["closed_loop"] = teacher_oracle_success(list(range(args.seeds)), args.profile)
    elif args.closed_loop == "policy":
        closed_loop_success()
    return record


def _measure(bench: Callable[[], dict], column: str, device: str | None) -> Cell:
    """Run one cell; a broken device must not kill the rest of the matrix."""
    start = time.perf_counter()
    try:
        result = bench()
    except Exception as exc:  # noqa: BLE001 - cell failures are recorded, not raised
        logger.warning("%s cell failed: %s", column, exc)
        return Cell(device=column, device_id=device, error=str(exc)[:300])
    logger.info("%s cell done in %.1f s", column, time.perf_counter() - start)
    return Cell(device=column, device_id=device, **result)


def _row_dict(row: Row) -> dict:
    return {
        "label": row.label,
        "backend": row.backend,
        "precision": row.precision,
        "variant": row.variant,
        "model_size_bytes": row.model_size_bytes,
        "weights": row.weights,
        "activations": row.activations,
        "rel_error_vs_fp32": row.rel_error_vs_fp32,
        "normalized_mse_vs_fp32": row.normalized_mse_vs_fp32,
        "rel_error_vs_pytorch": row.rel_error_vs_pytorch,
        "cells": {
            name: {
                "device_id": cell.device_id,
                "latency_ms": cell.latency_ms,
                "throughput_ips": cell.throughput_ips,
                "cold_compile_s": cell.cold_compile_s,
                "rss_delta_mb": cell.rss_delta_mb,
                "num_iters": cell.num_iters,
                "error": cell.error,
            }
            for name, cell in row.cells.items()
        },
    }


def _ir_precisions(export_dir: Path) -> tuple[str, str]:
    """Weights/activations precision read from the IR's own element types."""
    model = read_ir(_ir_path(export_dir))
    names = {
        ov.Type.f32: "FP32",
        ov.Type.f16: "FP16",
        ov.Type.i8: "INT8",
        ov.Type.u8: "UINT8",
    }
    totals: dict[str, int] = {}
    for constant in model.get_ops():
        if constant.get_type_name() != "Constant":
            continue
        try:
            size = int(np.prod(constant.output(0).get_shape())) * constant.output(0).get_element_type().size
        except Exception:  # noqa: BLE001 - a dynamic shape contributes no weight bytes
            logger.debug("constant with a dynamic shape skipped in the precision scan")
            continue
        label = names.get(constant.output(0).get_element_type())
        if label:
            totals[label] = totals.get(label, 0) + max(size, 0)
    weights = max(totals, key=totals.get) if totals else "FP32"
    return weights, "FP32"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _cell_ms(row: dict, column: str) -> str:
    cell = row["cells"].get(column, {})
    latency = cell.get("latency_ms")
    if latency:
        return f"{latency['p50']:.2f} ms"
    return DASH


def render_markdown(record: dict) -> str:
    """The headline table plus the supporting detail the numbers need."""
    host = record["host"]
    rows = record["rows"]
    closed = record.get("closed_loop")
    header = ["ACT policy", *COLUMNS, "Action error vs FP32"]
    lines = ["# OpenVINO optimization matrix", ""]
    lines.append(f"- host cpu: {host.get('cpu')}")
    lines.append(f"- os: {host.get('os')}")
    lines.append(
        f"- openvino {host.get('openvino')} | physicalai {host.get('physicalai')} "
        f"| nncf {host.get('nncf')}"
    )
    lines.append(f"- devices: {', '.join(host.get('devices', []))}")
    lines.append(f"- checkpoint: {record.get('checkpoint')}")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        error = row.get("rel_error_vs_fp32")
        quality = "reference" if row["variant"] == "fp32" else _sci(error)
        cells = [_cell_ms(row, column) for column in COLUMNS]
        lines.append("| " + " | ".join([row["label"], *cells, quality]) + " |")
    if closed and closed.get("driver") == "teacher-oracle":
        lines.append("")
        lines.append(
            f"Teacher-oracle mug success (system smoke, precision-independent): "
            f"{closed['successes']}/{closed['seeds']} seeds "
            f"({closed['success_rate']:.0%}, graph {closed['graph']})."
        )
    lines += ["", "## Detail", ""]
    lines.append(
        "| row | device | p50 ms | p95 ms | throughput ips | cold compile s | model MB | rss MB | weights | acts |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        size = row.get("model_size_bytes")
        model_mb = f"{size / 2**20:.1f}" if size else DASH
        for column in COLUMNS:
            cell = row["cells"].get(column, {})
            latency = cell.get("latency_ms") or {}
            lines.append(
                f"| {row['label']} | {column} | {_fmt(latency.get('p50'))} | {_fmt(latency.get('p95'))} "
                f"| {_fmt(cell.get('throughput_ips'))} | {_fmt(cell.get('cold_compile_s'))} "
                f"| {model_mb} | {_fmt(cell.get('rss_delta_mb'))} "
                f"| {_fmt(row.get('weights'))} | {_fmt(row.get('activations'))} |"
            )
    lines += ["", "## Parity gates", ""]
    lines.append("| row | rel err vs FP32 | norm MSE vs FP32 | rel err vs PyTorch | verdict |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in rows:
        rel = row.get("rel_error_vs_fp32")
        mse = row.get("normalized_mse_vs_fp32")
        if row["variant"] == "fp32":
            verdict = "reference"
        elif row["variant"] == "pytorch":
            verdict = "n/a (baseline)"
        elif rel is None or mse is None:
            verdict = "not measured"
        else:
            passed, detail = parity_verdict(row["variant"], rel, mse)
            verdict = f"{'pass' if passed else 'FAIL'} ({detail})"
        lines.append(
            f"| {row['label']} | {_sci(rel)} | {_sci(mse)} | {_sci(row.get('rel_error_vs_pytorch'))} "
            f"| {verdict} |"
        )
    lines += ["", "## Notes", ""]
    lines.extend(f"- {note}" for note in record.get("notes", []))
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    return DASH if value is None else str(value)


def _sci(value: object) -> str:
    """Scientific notation for the parity columns (a 1e-7 error must not read as 0)."""
    return DASH if value is None else f"{float(value):.3e}"


def write_outputs(record: dict, out_dir: Path, stem: str) -> list[Path]:
    """JSON (the record), CSV (flattened cells), and markdown (the table)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    paths.append(json_path)
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "row,backend,precision,variant,device,device_id,p50_ms,p90_ms,p95_ms,p99_ms,"
            "throughput_ips,cold_compile_s,rss_delta_mb,rel_error_vs_fp32,"
            "normalized_mse_vs_fp32,rel_error_vs_pytorch,error\n"
        )
        for row in record["rows"]:
            for column, cell in row["cells"].items():
                latency = cell.get("latency_ms") or {}
                fields = [
                    row["label"], row["backend"], row["precision"], row["variant"], column,
                    cell.get("device_id") or "", _csv(latency.get("p50")), _csv(latency.get("p90")),
                    _csv(latency.get("p95")), _csv(latency.get("p99")),
                    _csv(cell.get("throughput_ips")), _csv(cell.get("cold_compile_s")),
                    _csv(cell.get("rss_delta_mb")), _csv(row.get("rel_error_vs_fp32")),
                    _csv(row.get("normalized_mse_vs_fp32")), _csv(row.get("rel_error_vs_pytorch")),
                    str(cell.get("error") or "").replace(",", ";").replace("\n", " "),
                ]
                handle.write(",".join(fields) + "\n")
    paths.append(csv_path)
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(render_markdown(record), encoding="utf-8")
    paths.append(md_path)
    return paths


def _csv(value: object) -> str:
    return "" if value is None else str(value)


def _host_slug(cpu: str | None) -> str:
    tokens = re.findall(r"[a-z0-9]+", (cpu or "unknown-host").lower())
    return "-".join(tokens) or "unknown-host"


def _print_matrix(record: dict) -> None:
    """The table, on stdout, in the same shape as the markdown."""
    header = ["ACT policy", *COLUMNS, "Action error vs FP32"]
    print("| " + " | ".join(header) + " |")
    for row in record["rows"]:
        error = row.get("rel_error_vs_fp32")
        quality = "reference" if row["variant"] == "fp32" else _sci(error)
        print("| " + " | ".join([row["label"], *[_cell_ms(row, c) for c in COLUMNS], quality]) + " |")


def main(argv: list[str] | None = None) -> int:
    """CLI entry: provision, measure, and write the matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None, type=Path, help="trained ACT checkpoint (.ckpt)")
    parser.add_argument("--exports", default=Path("artifacts/act_public"), type=Path,
                        help="root for the exported rungs")
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path, help="report directory")
    parser.add_argument("--iters", default=DEFAULT_ITERS, type=int, help="timed calls per cell")
    parser.add_argument("--warmup", default=DEFAULT_WARMUP, type=int, help="warmup calls per cell")
    parser.add_argument("--seed", default=0, type=int, help="input/calibration seed")
    parser.add_argument("--seeds", default=10, type=int, help="closed-loop episode seeds")
    parser.add_argument("--profile", default="dr_train", help="scene profile for closed loop")
    parser.add_argument("--closed-loop", choices=["none", "teacher", "policy"], default="none",
                        help="policy is not implemented; teacher is a precision-independent smoke")
    parser.add_argument("--force", action="store_true", help="re-provision rungs that already exist")
    parser.add_argument("--no-auto-train", dest="auto_train", action="store_false",
                        help="fail instead of training the public stand-in when no checkpoint exists")
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iters < 1:
        parser.error("--iters and --warmup must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.closed_loop == "policy":
        # Fail before spending minutes benchmarking a run that cannot finish.
        print("error: closed-loop policy mode is not implemented (see --closed-loop teacher)")
        return 2
    try:
        record = build_matrix(args)
    except MatrixError as exc:
        print(f"error: {exc}")
        return 2
    stem = f"ov_matrix_{_host_slug(record['host'].get('cpu'))}_{time.strftime('%Y%m%d-%H%M%S')}"
    paths = write_outputs(record, args.out, stem)
    _print_matrix(record)
    print(f"results: {paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
