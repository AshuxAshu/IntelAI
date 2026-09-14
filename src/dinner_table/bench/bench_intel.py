"""Intel benchmark CLI: the deliverable-3 latency and throughput harness.

Micro mode sweeps model x device x precision on the public stand-in artifacts:
the ACT policy through physicalai's InferenceModel (its InferenceLatencyBenchmark
drives the timed loop), the YOLO detector through our OpenvinoDetector, and the
VLM timed manually (prefill/decode tok/s plus a fixed 128-token end-to-end run).
PyTorch CPU / PyTorch-XPU comparison rows are added when a torch checkpoint is
available. Results are schema-versioned JSON plus CSV, markdown, and charts via
report.py. System mode (full headless episodes) arrives with the evaluation
harness and raises a clear error until then.

# NOTE: this module runs past the ~300-line guideline - one cohesive CLI plus
# its three model benches; splitting would create artificial seams.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import openvino as ov

from dinner_table.bench.device_probe import probe_host
from dinner_table.bench.report import render
from dinner_table.bench.telemetry_sampler import rss_mb
from dinner_table.config import DinnerTableError

SCHEMA_VERSION = "bench/1.2"
ACT_EXPORT_ROOT = Path("artifacts/act_public")
PRECISION_RUNGS = {"fp32": "fp32_reference", "fp16": "fp16", "int8": "int8_ptq"}
PRECISION_LABELS = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
DEFAULT_ITERS = 200
DEFAULT_WARMUP = 10
AUTO_DEVICES = ("CPU", "GPU", "NPU")
VLM_E2E_TOKENS = 128
VLM_MAX_REPS = 5
VLM_SYSTEM = "You are a helpful robot assistant."
VLM_USER = "Describe what you see in this image in one sentence."
_TYPE_NAMES = {
    ov.Type.f32: "FP32",
    ov.Type.f16: "FP16",
    ov.Type.i8: "INT8",
    ov.Type.u8: "UINT8",
    ov.Type.i4: "INT4",
    ov.Type.u4: "UINT4",
}

logger = logging.getLogger(__name__)


class BenchError(DinnerTableError):
    """Benchmark misuse or environment failure."""


def _dummy_inputs() -> Iterator[dict]:
    """Infinite empty-sample stream for callables that take no real input."""
    while True:
        yield {}


def _percentiles(times_s: list[float]) -> dict[str, float]:
    """p50/p90/p95/p99 in milliseconds from raw per-call seconds."""
    array = np.asarray(times_s, dtype=np.float64) * 1000.0
    values = np.percentile(array, [50, 90, 95, 99])
    return {
        "p50": round(float(values[0]), 3),
        "p90": round(float(values[1]), 3),
        "p95": round(float(values[2]), 3),
        "p99": round(float(values[3]), 3),
    }


class _RecordingCall:
    """Wraps a callable so per-call wall times are recorded next to it."""

    def __init__(self, fn: Callable[[dict], object]) -> None:
        self._fn = fn
        self.times: list[float] = []

    def __call__(self, sample: dict) -> object:
        start = time.perf_counter()
        output = self._fn(sample)
        self.times.append(time.perf_counter() - start)
        return output


def _latency_benchmark(
    fn: Callable[[dict], object], inputs: Iterable[dict], iters: int, warmup: int
) -> dict:
    """Timed loop driven by physicalai's InferenceLatencyBenchmark over fn."""
    from physicalai.benchmark.performance import InferenceLatencyBenchmark

    recording = _RecordingCall(fn)
    bench = InferenceLatencyBenchmark(max_iters=iters, warmup_iters=warmup)
    metrics = bench.run(recording, inputs)
    measured = recording.times[warmup:]
    if not measured:
        raise BenchError("the latency benchmark produced no measured iterations")
    if metrics["num_iters"] != len(measured):
        raise BenchError(
            f"recording mismatch: {metrics['num_iters']} benchmark iterations, "
            f"{len(measured)} recorded"
        )
    return {
        "latency_ms": _percentiles(measured),
        "throughput_ips": round(len(measured) / sum(measured), 2),
        "num_iters": len(measured),
    }


def _dominant_type(model: ov.Model, constants: bool) -> str:
    """Dominant element type by tensor bytes among weights or compute ops."""
    totals: dict[str, int] = {}
    for op in model.get_ops():
        kind = op.get_type_name()
        if constants and kind != "Constant":
            continue
        if not constants and kind in ("Constant", "Parameter", "Result", "Convert"):
            continue
        output = op.output(0)
        label = _TYPE_NAMES.get(output.get_element_type())
        if label is None:
            continue
        try:
            size = max(int(np.prod(output.get_shape())) * output.get_element_type().size, 1)
        except Exception:  # noqa: BLE001 - dynamic shapes count as size 1
            size = 1
        if constants and size < 1024:
            continue
        totals[label] = totals.get(label, 0) + size
    if not totals:
        return "FP32"
    return max(totals, key=totals.get)


def _ir_precisions(ir_path: Path) -> tuple[str, str]:
    """(weights, activations) precision read from the IR: NNCF rt_info first,
    then the dominant element types of the graph itself."""
    model = ov.Core().read_model(str(ir_path))
    # NOTE: subscripting a missing rt_info key returns an invalid OVAny instead
    # of raising, and `in` on that object segfaults - membership is checked on
    # the parent map before every subscript.
    rt = model.get_rt_info()
    if "nncf" in rt:
        nncf = rt["nncf"]
        if "quantization" in nncf:
            return "INT8", "INT8"
        if "weight_compression" in nncf:
            return "INT8", _dominant_type(model, constants=False)
    else:
        logger.debug("no nncf rt_info on %s; falling back to the graph scan", ir_path)
    return _dominant_type(model, constants=True), _dominant_type(model, constants=False)


def _rung_ir_path(export_dir: Path) -> Path:
    """IR path from an export directory's own manifest artifact table."""
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    return export_dir / manifest["model"]["artifacts"]["openvino"]


def _bench_act(rung_dir: Path, device: str, precision: str, args: argparse.Namespace) -> dict:
    """One ACT rung on one device through physicalai's InferenceModel."""
    from physicalai.benchmark.performance.input_sources import RandomInputSource
    from physicalai.inference import InferenceModel

    rss_before = rss_mb()
    start = time.perf_counter()
    model = InferenceModel(rung_dir, device=device)
    cold_compile_s = round(time.perf_counter() - start, 2)
    source = RandomInputSource(
        model.input_features, seed=args.seed, num_samples=args.iters + args.warmup + 8
    )
    stats = _latency_benchmark(model, source, args.iters, args.warmup)
    weights, activations = _ir_precisions(_rung_ir_path(rung_dir))
    return {
        "model": "act",
        "backend": "openvino",
        "device": device,
        "precision": precision,
        **stats,
        "cold_compile_s": cold_compile_s,
        "weights": weights,
        "activations": activations,
        "rss_delta_mb": round(rss_mb() - rss_before, 1),
    }


def _bench_yolo(device: str, cache_dir: Path, args: argparse.Namespace) -> dict:
    """The stand-in YOLO detector through our OpenvinoDetector."""
    from dinner_table.runtime.engines import OpenvinoDetector, fetch_standin_yolo

    onnx_path, names = fetch_standin_yolo()
    rng = np.random.default_rng(args.seed)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    rss_before = rss_mb()
    start = time.perf_counter()
    detector = OpenvinoDetector(Path(onnx_path), names, device, cache_dir=cache_dir)
    cold_compile_s = round(time.perf_counter() - start, 2)
    stats = _latency_benchmark(
        lambda _sample: detector.detect(frame), _dummy_inputs(), args.iters, args.warmup
    )
    weights, activations = _ir_precisions(Path(onnx_path))
    precision = "INT8" if weights in ("INT8", "UINT8") else weights
    return {
        "model": "yolo",
        "backend": "openvino",
        "device": device,
        "precision": precision,
        **stats,
        "cold_compile_s": cold_compile_s,
        "weights": weights,
        "activations": activations,
        "rss_delta_mb": round(rss_mb() - rss_before, 1),
    }


def _bench_vlm(device: str, args: argparse.Namespace) -> dict:
    """The stand-in VLM: prefill/decode tok/s plus a fixed 128-token run."""
    import openvino_genai as genai

    from dinner_table.runtime.engines import fetch_standin_vlm

    model_dir = Path(fetch_standin_vlm())
    rss_before = rss_mb()
    start = time.perf_counter()
    pipe = genai.VLMPipeline(str(model_dir), device)
    cold_compile_s = round(time.perf_counter() - start, 2)
    tokenizer = pipe.get_tokenizer()
    messages = [
        {"role": "system", "content": VLM_SYSTEM},
        {"role": "user", "content": VLM_USER},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    rng = np.random.default_rng(args.seed)
    image = ov.Tensor(np.ascontiguousarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)))
    pipe.generate(
        prompt, images=[image], generation_config=genai.GenerationConfig(max_new_tokens=1)
    )
    config = genai.GenerationConfig(max_new_tokens=VLM_E2E_TOKENS, ignore_eos=True)
    reps = max(1, min(args.iters, VLM_MAX_REPS))
    e2e_times: list[float] = []
    prefill: list[float] = []
    decode: list[float] = []
    tokens_generated = 0
    for _ in range(reps):
        start = time.perf_counter()
        output = pipe.generate(prompt, images=[image], generation_config=config)
        e2e_times.append(time.perf_counter() - start)
        metrics = output.perf_metrics
        ttft_s = metrics.get_ttft().mean / 1000.0
        tpot_s = metrics.get_tpot().mean / 1000.0
        if ttft_s > 0:
            prefill.append(metrics.get_num_input_tokens() / ttft_s)
        if tpot_s > 0:
            decode.append(1.0 / tpot_s)
        tokens_generated = metrics.get_num_generated_tokens()
    weights, activations = _ir_precisions(model_dir / "openvino_language_model.xml")
    precision = "INT8" if weights in ("INT8", "UINT8") else weights
    del pipe
    gc.collect()
    return {
        "model": "vlm",
        "backend": "openvino",
        "device": device,
        "precision": precision,
        "latency_ms": _percentiles(e2e_times),
        "prefill_tok_s": round(float(np.median(prefill)), 2) if prefill else None,
        "decode_tok_s": round(float(np.median(decode)), 2) if decode else None,
        "e2e_128tok_s": round(float(np.median(e2e_times)), 3),
        "tokens_generated": tokens_generated,
        "reps": reps,
        "cold_compile_s": cold_compile_s,
        "weights": weights,
        "activations": activations,
        "rss_delta_mb": round(rss_mb() - rss_before, 1),
    }


def _bench_torch_act(ckpt: Path, reference_rung: Path, args: argparse.Namespace) -> list[dict]:
    """PyTorch CPU (and XPU when importable) comparison rows for the ACT policy."""
    import torch
    from physicalai.policies import ACT

    rss_before = rss_mb()
    start = time.perf_counter()
    policy = ACT.load_from_checkpoint(str(ckpt))
    policy.eval()
    cold_compile_s = round(time.perf_counter() - start, 2)
    model_ir = ov.Core().read_model(str(_rung_ir_path(reference_rung)))
    rng = np.random.default_rng(args.seed)
    shapes = {
        inp.get_any_name(): tuple(dim.get_length() for dim in inp.partial_shape)
        for inp in model_ir.inputs
    }
    rows: list[dict] = []
    backends = [("pytorch", "cpu", torch.device("cpu"))]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        backends.append(("pytorch-xpu", "xpu", torch.device("xpu")))
    for backend, device_name, device in backends:
        policy.model.to(device)
        batch = {
            name: torch.from_numpy(
                rng.uniform(0.0, 1.0, shape).astype(np.float32)
                if len(shape) > 2
                else rng.standard_normal(shape).astype(np.float32)
            ).to(device)
            for name, shape in shapes.items()
        }
        with torch.no_grad():

            def _forward(_sample: dict, _batch: dict = batch) -> object:
                return policy.model(_batch)

            stats = _latency_benchmark(_forward, _dummy_inputs(), args.iters, args.warmup)
        rows.append(
            {
                "model": "act",
                "backend": backend,
                "device": device_name,
                "precision": "FP32",
                **stats,
                "cold_compile_s": cold_compile_s,
                "weights": "FP32",
                "activations": "FP32",
                "rss_delta_mb": round(rss_mb() - rss_before, 1),
            }
        )
    policy.model.to("cpu")
    return rows


def _guarded(bench: Callable[[], dict], fallback: dict) -> dict:
    """Run one model x device x precision cell; record failures as error rows."""
    try:
        return bench()
    except Exception as exc:  # noqa: BLE001 - a broken cell must not kill the sweep
        logger.warning("benchmark cell failed: %s", exc)
        return {**fallback, "error": str(exc)[:300]}


def _discover_torch_ckpt(explicit: Path | None) -> Path | None:
    """Explicit checkpoint, else the newest one under the run metadata dirs."""
    if explicit is not None:
        if not explicit.is_file():
            raise BenchError(f"torch checkpoint not found: {explicit}")
        return explicit
    meta_paths: list[Path] = []
    for runs_dir in (Path("runs"), ACT_EXPORT_ROOT / "runs"):
        meta_paths.extend(sorted(runs_dir.glob("*/meta.json")))
    candidates: list[Path] = []
    for meta_path in meta_paths:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        experiment_dir = meta.get("experiment_dir")
        if experiment_dir:
            candidates.extend(sorted(Path(experiment_dir).glob("checkpoints/*.ckpt")))
    if Path("experiments").is_dir():
        candidates.extend(sorted(Path("experiments").rglob("checkpoints/*.ckpt")))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def micro_benchmarks(args: argparse.Namespace) -> list[dict]:
    """Every model x device x precision micro-benchmark cell."""
    runs: list[dict] = []
    cache_dir = args.out / ".ov_cache"
    if "act" in args.models:
        rungs = {
            label: ACT_EXPORT_ROOT / PRECISION_RUNGS[label]
            for label in args.precisions
            if (ACT_EXPORT_ROOT / PRECISION_RUNGS[label]).is_dir()
        }
        if not rungs:
            raise BenchError(
                f"no ACT export rungs under {ACT_EXPORT_ROOT}; generate them with "
                "python -m dinner_table.policies.export_quantize"
            )
        for rung, rung_dir in rungs.items():
            precision = PRECISION_LABELS[rung]
            for device in args.devices:
                runs.append(
                    _guarded(
                        lambda r=rung_dir, d=device, p=precision: _bench_act(r, d, p, args),
                        {"model": "act", "device": device, "precision": precision},
                    )
                )
        ckpt = _discover_torch_ckpt(args.ckpt)
        if ckpt is not None:
            logger.info("pytorch comparison checkpoint: %s", ckpt)
            runs.extend(_bench_torch_act(ckpt, next(iter(rungs.values())), args))
        else:
            logger.info(
                "pytorch comparison skipped: no checkpoint found (pass --ckpt or train via "
                "python -m dinner_table.policies.studio_train)"
            )
    if "yolo" in args.models:
        for device in args.devices:
            runs.append(
                _guarded(
                    lambda d=device: _bench_yolo(d, cache_dir, args),
                    {"model": "yolo", "device": device},
                )
            )
    if "vlm" in args.models:
        for device in args.devices:
            runs.append(
                _guarded(lambda d=device: _bench_vlm(d, args), {"model": "vlm", "device": device})
            )
    return runs


def resolve_devices(spec: str) -> list[str]:
    """'auto' maps to the available CPU/NPU plus every Intel GPU (multi-GPU
    hosts name them GPU.0, GPU.1, ...; non-Intel GPUs stay opt-in via the
    explicit device list); any other spec is the verbatim comma list."""
    if spec == "auto":
        core = ov.Core()
        devices: list[str] = []
        for device in core.available_devices:
            base = device.split(".")[0]
            if base not in AUTO_DEVICES:
                continue
            if base == "GPU":
                try:
                    name = str(core.get_property(device, "FULL_DEVICE_NAME"))
                except Exception:  # noqa: BLE001 - unreadable name keeps the device
                    name = ""
                if "Intel" not in name:
                    continue
            devices.append(device)
        return devices
    devices = [device.strip() for device in spec.split(",") if device.strip()]
    if not devices:
        raise BenchError("no benchmark devices requested")
    return devices


def resolve_precisions(spec: str) -> list[str]:
    """Requested ACT precisions in ladder order."""
    if spec == "all":
        return ["fp32", "fp16", "int8"]
    if spec not in PRECISION_RUNGS:
        raise BenchError(f"unknown precision {spec!r} (choose fp32, fp16, int8, all)")
    return [spec]


def resolve_seeds(spec: str) -> list[int]:
    """Seed list from '0-9' or '0,1,2' (system mode episodes)."""
    match = re.fullmatch(r"(\d+)-(\d+)", spec)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            raise BenchError(f"seed range {spec!r} is empty")
        return list(range(start, end + 1))
    try:
        seeds = [int(part) for part in spec.split(",") if part.strip()]
    except ValueError as exc:
        raise BenchError(f"cannot parse seeds {spec!r}: {exc}") from exc
    if not seeds:
        raise BenchError("no seeds requested")
    return seeds


def _system_not_implemented() -> None:
    raise BenchError(
        "system mode is not implemented yet: it runs full headless episodes through the "
        "evaluation harness, which lands later in the development sequence; use "
        "--mode micro for the model benchmarks"
    )


def _host_slug(cpu: str | None) -> str:
    """Filename-safe host identifier from the CPU model."""
    if not cpu:
        return "unknown-host"
    tokens = re.findall(r"[a-z0-9]+", cpu.lower())
    return "-".join(tokens) if tokens else "unknown-host"


def _print_summary(runs: list[dict]) -> None:
    """One plain line per run cell (no colors, no decorative separators)."""
    for run in runs:
        if "error" in run:
            print(f"{run['model']} {run['device']} failed: {run['error']}")
        elif run["model"] == "vlm":
            print(
                f"vlm {run['device']} prefill {run['prefill_tok_s']} tok/s "
                f"decode {run['decode_tok_s']} tok/s e2e128 {run['e2e_128tok_s']} s"
            )
        else:
            latency = run["latency_ms"]
            print(
                f"{run['model']} {run['device']} {run['precision']} {run.get('backend', 'openvino')} "
                f"p50 {latency['p50']} ms p95 {latency['p95']} ms "
                f"throughput {run['throughput_ips']} ips"
            )


def main() -> None:
    """CLI entry: run the benchmark sweep and write the report set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=["act", "yolo", "vlm", "all"], default="all")
    parser.add_argument("--devices", default="auto", help="comma list, or 'auto'")
    parser.add_argument("--precisions", choices=["fp32", "fp16", "int8", "all"], default="all")
    parser.add_argument("--mode", choices=["micro", "system", "full"], default="micro")
    parser.add_argument("--seeds", default="0-9", help="episode seeds, e.g. 0-9 or 0,3,7")
    parser.add_argument("--out", default="bench", type=Path)
    parser.add_argument("--iters", default=DEFAULT_ITERS, type=int, help="timed calls per cell")
    parser.add_argument("--warmup", default=DEFAULT_WARMUP, type=int, help="warmup calls per cell")
    parser.add_argument("--seed", default=0, type=int, help="input sampling seed")
    parser.add_argument("--ckpt", default=None, type=Path, help="torch ACT checkpoint")
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.mode == "system":
        _system_not_implemented()
    args.models = ["act", "yolo", "vlm"] if args.models == "all" else [args.models]
    args.precisions = resolve_precisions(args.precisions)
    args.devices = resolve_devices(args.devices)
    resolve_seeds(args.seeds)
    probe = probe_host()
    print(f"host: {probe['cpu']} devices: {probe['devices']} npu: {probe['npu_available']}")
    runs = micro_benchmarks(args)
    args.out.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%Y%m%d-%H%M%S")
    stem = f"results_{_host_slug(probe['cpu'])}_{date}"
    results = {"schema": SCHEMA_VERSION, "host": probe, "runs": runs}
    json_path = args.out / f"{stem}.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    render(results, args.out, stem)
    _print_summary(runs)
    print(f"results: {json_path}")
    if args.mode == "full":
        _system_not_implemented()


if __name__ == "__main__":
    main()
