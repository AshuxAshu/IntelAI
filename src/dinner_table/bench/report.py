"""Benchmark report generation: results dict to CSV, markdown, and charts.

render() is the single entry point used by the bench CLI: it writes a
flattened CSV of every run cell, a markdown report with the host banner and
per-model tables, and one bar chart per model (matplotlib, Agg backend).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Agg must be selected before pyplot is imported
import matplotlib.pyplot as plt

CSV_FIELDS = (
    "model",
    "backend",
    "device",
    "precision",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "throughput_ips",
    "cold_compile_s",
    "weights",
    "activations",
    "rss_delta_mb",
    "prefill_tok_s",
    "decode_tok_s",
    "e2e_128tok_s",
    "num_iters",
    "error",
)
_RUN_KEYS = (
    "backend",
    "throughput_ips",
    "cold_compile_s",
    "weights",
    "activations",
    "rss_delta_mb",
    "prefill_tok_s",
    "decode_tok_s",
    "e2e_128tok_s",
    "num_iters",
)


def to_csv_rows(results: dict) -> list[dict]:
    """Flatten run records into one CSV row per cell."""
    rows = []
    for run in results.get("runs", []):
        row = {
            "model": run.get("model"),
            "device": run.get("device"),
            "precision": run.get("precision"),
            "error": run.get("error"),
        }
        for key in _RUN_KEYS:
            row[key] = run.get(key)
        latency = run.get("latency_ms")
        for percentile in ("p50", "p90", "p95", "p99"):
            row[f"{percentile}_ms"] = latency.get(percentile) if latency else None
        rows.append(row)
    return rows


def write_csv(results: dict, path: Path) -> None:
    """Write the flattened CSV of all run cells."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in to_csv_rows(results):
            writer.writerow(
                {key: "" if row.get(key) is None else row.get(key) for key in CSV_FIELDS}
            )


def _rows(results: dict, model: str) -> list[dict]:
    return [run for run in results.get("runs", []) if run.get("model") == model]


def _fmt(value: object) -> str:
    return "-" if value is None else str(value)


_LATENCY_HEADER = "| backend | device | precision | p50 ms | p90 ms | p95 ms | p99 ms | throughput ips | cold compile s | weights | activations | rss MB |"
_TABLE_RULE = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def _latency_table(runs: list[dict]) -> list[str]:
    """Markdown table lines for models measured by call latency."""
    lines = [_LATENCY_HEADER, _TABLE_RULE]
    for run in runs:
        latency = run.get("latency_ms", {})
        lines.append(
            "| {backend} | {device} | {precision} | {p50} | {p90} | {p95} | {p99} "
            "| {ips} | {cold} | {weights} | {acts} | {rss} |".format(
                backend=_fmt(run.get("backend", "openvino")),
                device=_fmt(run.get("device")),
                precision=_fmt(run.get("precision")),
                p50=_fmt(latency.get("p50")),
                p90=_fmt(latency.get("p90")),
                p95=_fmt(latency.get("p95")),
                p99=_fmt(latency.get("p99")),
                ips=_fmt(run.get("throughput_ips")),
                cold=_fmt(run.get("cold_compile_s")),
                weights=_fmt(run.get("weights")),
                acts=_fmt(run.get("activations")),
                rss=_fmt(run.get("rss_delta_mb")),
            )
        )
    return lines


def _comparison_table(act_runs: list[dict]) -> list[str]:
    """Headline speedup table: every ACT row against the PyTorch CPU baseline."""
    baseline = next(
        (run for run in act_runs if run.get("backend") == "pytorch" and "latency_ms" in run),
        None,
    )
    if baseline is None:
        return ["PyTorch comparison: no pytorch-cpu row present (pass --ckpt to add one)."]
    base_p50 = baseline["latency_ms"]["p50"]
    lines = [
        "| backend | device | precision | p50 ms | speedup vs pytorch-cpu |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in act_runs:
        latency = run.get("latency_ms")
        if not latency:
            continue
        speedup = round(base_p50 / latency["p50"], 2) if latency["p50"] else None
        lines.append(
            f"| {run.get('backend', 'openvino')} | {run.get('device')} | "
            f"{run.get('precision')} | {latency['p50']} | {_fmt(speedup)} |"
        )
    return lines


def _vlm_table(runs: list[dict]) -> list[str]:
    lines = [
        "| device | precision | prefill tok/s | decode tok/s | e2e 128 tok s |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        lines.append(
            f"| {_fmt(run.get('device'))} | {_fmt(run.get('precision'))} | "
            f"{_fmt(run.get('prefill_tok_s'))} | {_fmt(run.get('decode_tok_s'))} | "
            f"{_fmt(run.get('e2e_128tok_s'))} |"
        )
    return lines


def _host_section(host: dict) -> list[str]:
    stack_line = (
        f"- openvino {host.get('openvino')} | physicalai {host.get('physicalai')} "
        f"| physicalai-train {host.get('physicalai_train')} | nncf {host.get('nncf')}"
    )
    lines = [
        f"- schema host cpu: {host.get('cpu')}",
        f"- os: {host.get('os')}",
        f"- ram: {host.get('ram_total_gb')} GB",
        f"- devices: {', '.join(host.get('devices', []))}",
        f"- npu available: {host.get('npu_available')}",
        f"- power profile: {host.get('power_profile')}",
        stack_line,
    ]
    properties = host.get("device_properties", {})
    if properties:
        lines.append("")
        lines.append("| device | full name | type | architecture | execution units |")
        lines.append("| --- | --- | --- | --- | --- |")
        for device, entry in properties.items():
            lines.append(
                f"| {device} | {_fmt(entry.get('FULL_DEVICE_NAME'))} | "
                f"{_fmt(entry.get('DEVICE_TYPE'))} | {_fmt(entry.get('DEVICE_ARCHITECTURE'))} | "
                f"{_fmt(entry.get('GPU_EXECUTION_UNITS_COUNT'))} |"
            )
    return lines


def to_markdown(results: dict) -> str:
    """The full markdown report for one results dict."""
    host = results.get("host", {})
    runs = results.get("runs", [])
    lines = ["# Intel benchmark results", ""]
    lines.extend(_host_section(host))
    act_runs = _rows(results, "act")
    if act_runs:
        lines += ["", "## ACT policy", ""]
        lines.extend(_latency_table(act_runs))
        lines += ["", "### PyTorch comparison", ""]
        lines.extend(_comparison_table(act_runs))
    yolo_runs = _rows(results, "yolo")
    if yolo_runs:
        lines += ["", "## YOLO detector", ""]
        lines.extend(_latency_table(yolo_runs))
    vlm_runs = _rows(results, "vlm")
    if vlm_runs:
        lines += ["", "## VLM generation", ""]
        lines.extend(_vlm_table(vlm_runs))
    failures = [run for run in runs if run.get("error")]
    if failures:
        lines += [
            "",
            "## Failed cells",
            "",
            "| model | device | precision | error |",
            "| --- | --- | --- | --- |",
        ]
        for run in failures:
            lines.append(
                f"| {run.get('model')} | {run.get('device')} | {run.get('precision')} "
                f"| {run.get('error')} |"
            )
    system = results.get("system")
    if system:
        lines += ["", "## System mode", "", "```json", json.dumps(system, indent=2), "```"]
    return "\n".join(lines) + "\n"


def write_chart(results: dict, model: str, out_dir: Path, stem: str) -> Path | None:
    """One bar chart per model: p50 ms (latency models) or decode tok/s (VLM)."""
    runs = [run for run in _rows(results, model) if not run.get("error")]
    if not runs:
        return None
    if model == "vlm":
        values = [run.get("decode_tok_s") or 0.0 for run in runs]
        ylabel = "decode tok/s"
    else:
        values = [run["latency_ms"]["p50"] for run in runs]
        ylabel = "p50 latency (ms)"
    labels = [
        f"{run.get('backend', 'openvino')}\n{run.get('device')} {run.get('precision')}"
        for run in runs
    ]
    figure, axis = plt.subplots(figsize=(max(6.0, 0.9 * len(runs)), 4.0))
    axis.bar(range(len(runs)), values, color="#3372b9")
    axis.set_xticks(range(len(runs)))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(f"{model} benchmark ({results.get('schema', '')})")
    figure.tight_layout()
    path = out_dir / f"{stem}_{model}.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def render(results: dict, out_dir: Path, stem: str) -> list[Path]:
    """Write CSV, markdown, and chart files for one results dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    csv_path = out_dir / f"{stem}.csv"
    write_csv(results, csv_path)
    paths.append(csv_path)
    for model in ("act", "yolo", "vlm"):
        chart = write_chart(results, model, out_dir, stem)
        if chart is not None:
            paths.append(chart)
    markdown_path = out_dir / f"{stem}.md"
    markdown_path.write_text(to_markdown(results), encoding="utf-8")
    paths.append(markdown_path)
    return paths
