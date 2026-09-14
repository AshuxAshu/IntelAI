"""NPU preflight: compile-audit of every model artifact for the NPU devices.

Each locally available model artifact (public stand-ins now, project models
after the parity gates) is compiled for NPU and for NPUW:CPU,NPU; the ops the
NPU does not support are listed per model via query_model. Results go to
bench/npu_preflight.json. The audit is purely informational: missing models,
unreadable artifacts, and failed compiles are recorded, never fatal - the
script always exits 0.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import openvino as ov

NPU_DEVICES = ("NPU", "NPUW:CPU,NPU")
ACT_EXPORT_ROOT = Path("artifacts/act_public")
OUT_DEFAULT = Path("bench") / "npu_preflight.json"
VLM_COMPONENTS = (
    "openvino_language_model.xml",
    "openvino_text_embeddings_model.xml",
    "openvino_vision_embeddings_model.xml",
    "openvino_vision_embeddings_merger_model.xml",
)

logger = logging.getLogger(__name__)


def _act_artifacts() -> list[dict]:
    """Every ACT export rung present under the export root."""
    entries: list[dict] = []
    if not ACT_EXPORT_ROOT.is_dir():
        return [{"name": "act", "skipped": f"no export rungs under {ACT_EXPORT_ROOT}"}]
    for rung_dir in sorted(ACT_EXPORT_ROOT.iterdir()):
        manifest_path = rung_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ir_path = rung_dir / manifest["model"]["artifacts"]["openvino"]
        except (OSError, ValueError, KeyError):
            logger.warning("unreadable manifest under %s; skipping", rung_dir)
            continue
        if ir_path.is_file():
            entries.append({"name": f"act/{rung_dir.name}", "path": str(ir_path)})
    if not entries:
        return [{"name": "act", "skipped": f"no export rungs under {ACT_EXPORT_ROOT}"}]
    return entries


def _yolo_artifact() -> list[dict]:
    """The stand-in YOLO ONNX (fetched on demand; the fetch is light)."""
    try:
        from dinner_table.runtime.engines import fetch_standin_yolo

        onnx_path, _names = fetch_standin_yolo()
    except Exception as exc:  # noqa: BLE001 - informational script, never fatal
        return [{"name": "yolo", "skipped": f"stand-in unavailable: {exc}"}]
    return [{"name": "yolo", "path": str(onnx_path)}]


def _vlm_artifacts(fetch: bool) -> list[dict]:
    """The stand-in VLM's compute components (local cache only unless fetched)."""
    try:
        from huggingface_hub import snapshot_download

        from dinner_table.runtime.engines import STANDIN_VLM_REPO

        model_dir = Path(snapshot_download(STANDIN_VLM_REPO, local_files_only=not fetch))
    except Exception as exc:  # noqa: BLE001 - informational script, never fatal
        return [{"name": "vlm", "skipped": f"stand-in not in the local HF cache: {exc}"}]
    entries = [
        {"name": f"vlm/{component}", "path": str(model_dir / component)}
        for component in VLM_COMPONENTS
        if (model_dir / component).is_file()
    ]
    if not entries:
        return [{"name": "vlm", "skipped": "no compute IRs in the cached stand-in"}]
    return entries


def _last_message(exc: Exception) -> str:
    """The specific final line of an OpenVINO exception chain."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines[-1][:300] if lines else str(exc)[:300]


def _unsupported_ops_on_npu(core: ov.Core, model: ov.Model) -> list[str] | None:
    """Ops the NPU device cannot run (absent from its query_model result);
    None when the device or the query is unavailable."""
    try:
        supported = set(core.query_model(model, "NPU").keys())
    except Exception:  # noqa: BLE001 - no NPU device means no op listing
        return None
    all_names = {op.get_friendly_name() for op in model.get_ops()}
    return sorted(all_names - supported)


def audit_model(entry: dict) -> dict:
    """Compile one artifact for both NPU devices and list unsupported ops."""
    if "skipped" in entry:
        return entry
    core = ov.Core()
    try:
        model = core.read_model(entry["path"])
    except Exception as exc:  # noqa: BLE001 - unreadable artifacts are recorded
        return {**entry, "skipped": f"cannot read model: {_last_message(exc)}"}
    results: dict[str, dict] = {}
    for device in NPU_DEVICES:
        try:
            core.compile_model(model, device)
            results[device] = {"compiles": True}
        except Exception as exc:  # noqa: BLE001 - a failed compile is a result
            results[device] = {"compiles": False, "error": _last_message(exc)}
    return {
        **entry,
        "results": results,
        "unsupported_ops_on_npu": _unsupported_ops_on_npu(core, model),
    }


def preflight(models: list[str], fetch_vlm: bool) -> dict:
    """The full audit record for the requested model groups."""
    core = ov.Core()
    devices = list(core.available_devices)
    entries: list[dict] = []
    if "act" in models:
        entries.extend(_act_artifacts())
    if "yolo" in models:
        entries.extend(_yolo_artifact())
    if "vlm" in models:
        entries.extend(_vlm_artifacts(fetch_vlm))
    return {
        "npu_available": "NPU" in devices,
        "npuw_registered": any(device.startswith("NPUW") for device in devices),
        "devices": devices,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "models": [audit_model(entry) for entry in entries],
    }


def _print_summary(record: dict) -> None:
    """One plain line per audited model (no colors, no decorative separators)."""
    print(f"npu available: {record['npu_available']} npuw registered: {record['npuw_registered']}")
    for model in record["models"]:
        if "skipped" in model:
            print(f"{model['name']}: skipped ({model['skipped']})")
            continue
        for device in NPU_DEVICES:
            result = model["results"][device]
            outcome = "compiles" if result["compiles"] else f"fails ({result.get('error', '')})"
            print(f"{model['name']} on {device}: {outcome}")
        unsupported = model["unsupported_ops_on_npu"]
        if unsupported is not None:
            print(f"{model['name']}: {len(unsupported)} unsupported ops on NPU")


def main() -> None:
    """CLI entry: run the audit and write the record; always exits 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=["act", "yolo", "vlm", "all"], default="all")
    parser.add_argument("--out", default=OUT_DEFAULT, type=Path)
    parser.add_argument(
        "--fetch-vlm",
        action="store_true",
        help="download the VLM stand-in when not cached (heavy; otherwise local-cache only)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    models = ["act", "yolo", "vlm"] if args.models == "all" else [args.models]
    record = preflight(models, args.fetch_vlm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _print_summary(record)
    print(f"preflight written: {args.out}")


if __name__ == "__main__":
    main()
