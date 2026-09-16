"""Export and quantization ladder for the ACT policy (the Studio-to-runtime contract).

Four rungs over one checkpoint, each re-saved over the same manifest so the
physicalai runtime stays unaware of precision changes:
  1. export_openvino       - Studio default export (FP16-compressed IR + manifest)
  2. export_fp32_reference - the same export with FP32 weights (the parity baseline)
  3. compress_int8_weights - NNCF INT8 symmetric weight compression, with an
     asymmetric fallback when the CPU plugin rejects the symmetric graph
  4. ptq_int8              - NNCF full post-training quantization on calibration
     tensors (project dataset when available, else the public stand-in)
Every rung writes export_dir/rung.json (rung name, file hashes, model size) for
the benchmark and OPTIMIZATION.md tables.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import io
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import openvino as ov
from nncf import CompressWeightsMode, compress_weights, quantize
from nncf import Dataset as NncfDataset
from physicalai.export.backends import ExportParameters
from physicalai.inference.manifest import Manifest
from physicalai.policies import ACT

from dinner_table.config import DinnerTableError

logger = logging.getLogger(__name__)

RUNG_FP16 = "fp16"
RUNG_FP32 = "fp32_reference"
RUNG_INT8_WEIGHTS = "int8_weights"
RUNG_INT8_PTQ = "int8_ptq"
CALIBRATION_SUBSET_SIZE = 300


class ExportError(DinnerTableError):
    """Export or quantization failure."""


class FP32ReferenceACT(ACT):
    """ACT exporting FP32 weights through the Studio extra_export_args override point.

    The class name leaks into the manifest's policy name and artifact filename,
    so it must stay InferenceModel-safe (no leading underscore, alphanumeric)."""

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        args = dict(super().extra_export_args)
        args["openvino"] = dataclasses.replace(args["openvino"], compress_to_fp16=False)
        return args


def export_openvino(ckpt: Path, out_dir: Path) -> Path:
    """Rung 1: Studio default OpenVINO export (FP16-compressed IR + manifest)."""
    policy = ACT.load_from_checkpoint(str(ckpt))
    policy.export(out_dir, backend="openvino")
    return _write_rung(out_dir, RUNG_FP16)


def export_fp32_reference(ckpt: Path, out_dir: Path) -> Path:
    """Rung 2: the same export with compress_to_fp16=False (the parity baseline)."""
    policy = FP32ReferenceACT.load_from_checkpoint(str(ckpt))
    policy.export(out_dir, backend="openvino")
    return _write_rung(out_dir, RUNG_FP32)


def compress_int8_weights(export_dir: Path) -> Path:
    """Rung 3: INT8 symmetric weight compression, re-saved over the same manifest.

    # NOTE: the Studio compress_weights_openvino_int8_sym post-export hook does
    # not exist in physicalai 0.1.1; nncf.compress_weights is the equivalent.
    # INT8_SYM graphs of this policy are rejected by the OpenVINO CPU plugin
    # (they compile on GPU), so the rung verifies CPU compilation and falls
    # back to INT8_ASYM on a freshly read model (nncf mutates its input model,
    # so a re-compression of the same object would be doubly compressed).
    """
    ir_path = _ir_path(export_dir)
    mode = CompressWeightsMode.INT8_SYM
    compressed = compress_weights(read_ir(ir_path), mode=mode)
    if not _compiles_on_cpu(compressed):
        logger.warning("INT8_SYM graph rejected by the CPU plugin; retrying with INT8_ASYM")
        mode = CompressWeightsMode.INT8_ASYM
        compressed = compress_weights(read_ir(ir_path), mode=mode)
    if not _compiles_on_cpu(compressed):
        raise ExportError("INT8 weight compression produced a model the CPU plugin cannot compile")
    _save_ir(compressed, ir_path)
    return _write_rung(export_dir, RUNG_INT8_WEIGHTS, extra={"mode": mode.value})


def ptq_int8(export_dir: Path, calibration_samples: Iterable[Mapping[str, np.ndarray]]) -> Path:
    """Rung 4: NNCF full post-training quantization, re-saved over the same manifest."""
    samples = list(calibration_samples)
    if not samples:
        raise ExportError("post-training quantization requires at least one calibration sample")
    ir_path = _ir_path(export_dir)
    quantized = quantize(
        read_ir(ir_path),
        NncfDataset(samples),
        subset_size=min(CALIBRATION_SUBSET_SIZE, len(samples)),
    )
    _save_ir(quantized, ir_path)
    return _write_rung(export_dir, RUNG_INT8_PTQ, extra={"calibration_samples": len(samples)})


def random_inputs(export_dir: Path, n: int, seed: int = 0) -> list[dict[str, np.ndarray]]:
    """Seeded public stand-in inputs matching the exported model's actual inputs.

    Image tensors are drawn uniform in [0, 1], state tensors from a standard
    normal; shapes come from the compiled IR (fully static after tracing). Used
    for PTQ calibration and parity fixtures when the project dataset is
    unavailable.
    """
    model = read_ir(_ir_path(export_dir))
    rng = np.random.default_rng(seed)
    samples: list[dict[str, np.ndarray]] = []
    for _ in range(n):
        sample: dict[str, np.ndarray] = {}
        for inp in model.inputs:
            shape = tuple(dim.get_length() for dim in inp.partial_shape)
            name = inp.get_any_name()
            if len(shape) > 2:
                sample[name] = rng.uniform(0.0, 1.0, shape).astype(np.float32)
            else:
                sample[name] = rng.standard_normal(shape).astype(np.float32)
        samples.append(sample)
    return samples


def _ir_path(export_dir: Path) -> Path:
    """IR path from the manifest's own artifact table (never a guessed filename)."""
    manifest = Manifest.load(export_dir / "manifest.json")
    artifact = manifest.model.artifacts.get("openvino")
    if artifact is None:
        raise ExportError(f"manifest under {export_dir} declares no openvino artifact")
    path = export_dir / artifact
    if not path.is_file():
        raise ExportError(f"manifest artifact {path} is missing")
    return path


def read_ir(ir_path: Path) -> ov.Model:
    """Read an IR with its weights held in memory rather than memory-mapped.

    # NOTE: ov.Core().read_model(path) memory-maps the weights .bin. That is
    # harmless on Linux, where a mapped file can still be replaced, but on
    # Windows a mapped file is locked: the quantization rungs read the IR and
    # then save the compressed model back over the same path, and the
    # os.replace under that mapping fails with WinError 32 ("used by another
    # process"). Routing both halves through BytesIO keeps the weights on the
    # heap and leaves no handle open on the files being replaced.
    """
    if not ir_path.is_file():
        raise ExportError(f"IR not found: {ir_path}")
    xml_bytes = ir_path.read_bytes()
    weights_path = ir_path.with_suffix(".bin")
    if not weights_path.is_file():
        # A weights-free IR carries everything in the XML; one that references
        # weights without a .bin is corrupt, and OpenVINO's own complaint
        # ("Incorrect weights in bin file!") does not say which file is missing.
        try:
            return ov.Core().read_model(io.BytesIO(xml_bytes))
        except RuntimeError as exc:
            raise ExportError(
                f"{ir_path} has no sibling {weights_path.name} and the XML alone does not "
                f"deserialize: {exc}"
            ) from exc
    return ov.Core().read_model(io.BytesIO(xml_bytes), io.BytesIO(weights_path.read_bytes()))


def _compiles_on_cpu(model: ov.Model) -> bool:
    try:
        ov.Core().compile_model(model, "CPU")
    except RuntimeError:
        return False
    return True


def _replace_with_retry(source: Path, destination: Path, attempts: int = 5) -> None:
    """os.replace with a short backoff for Windows' transient file locks.

    Windows indexers and antivirus probes can hold a freshly written file for a
    moment after it closes; the move is legal a retry later. A failure on the
    final attempt propagates unchanged, so a genuine error is never masked.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            gc.collect()
            time.sleep(0.25 * (attempt + 1))


def _save_ir(model: ov.Model, ir_path: Path) -> None:
    """Save through a sibling temp file plus atomic replace.

    Overwriting the mapped files in place crashes the process, and a crash
    mid-write would leave a truncated IR, so both halves are written aside and
    then moved into place."""
    tmp = ir_path.with_name(ir_path.stem + "_saving.xml")
    ov.save_model(model, str(tmp))
    _replace_with_retry(tmp, ir_path)
    _replace_with_retry(tmp.with_suffix(".bin"), ir_path.with_suffix(".bin"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_rung(export_dir: Path, rung: str, extra: dict | None = None) -> Path:
    """Record rung provenance: rung name, per-file hashes, IR model size."""
    files: dict[str, str] = {}
    model_size = 0
    for path in sorted(p for p in export_dir.iterdir() if p.is_file()):
        if path.name == "rung.json":
            continue
        files[path.name] = _sha256_file(path)
        if path.suffix in (".xml", ".bin"):
            model_size += path.stat().st_size
    record: dict = {
        "rung": rung,
        "files": files,
        "model_size_bytes": model_size,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    if extra is not None:
        record.update(extra)
    rung_path = export_dir / "rung.json"
    rung_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return rung_path


def main() -> None:
    """CLI entry: run the ladder over one checkpoint, one subdirectory per rung."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path, help="trained ACT checkpoint (.ckpt)")
    parser.add_argument("--out", default=Path("artifacts/act_public"), type=Path)
    parser.add_argument(
        "--rung",
        choices=[RUNG_FP16, RUNG_FP32, RUNG_INT8_WEIGHTS, RUNG_INT8_PTQ, "all"],
        default="all",
    )
    parser.add_argument("--calibration-samples", type=int, default=CALIBRATION_SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.rung in (RUNG_FP16, "all"):
        export_openvino(args.ckpt, args.out / RUNG_FP16)
    if args.rung in (RUNG_FP32, "all"):
        export_fp32_reference(args.ckpt, args.out / RUNG_FP32)
    if args.rung in (RUNG_INT8_WEIGHTS, "all"):
        base = args.out / RUNG_INT8_WEIGHTS
        export_fp32_reference(args.ckpt, base)
        compress_int8_weights(base)
    if args.rung in (RUNG_INT8_PTQ, "all"):
        base = args.out / RUNG_INT8_PTQ
        export_fp32_reference(args.ckpt, base)
        ptq_int8(base, random_inputs(base, args.calibration_samples, seed=args.seed))
    print(f"ladder artifacts under {args.out}")


if __name__ == "__main__":
    main()
