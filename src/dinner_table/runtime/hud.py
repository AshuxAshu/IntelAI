"""HUD overlay: per-frame composites from telemetry only (PIL, no matplotlib).

Pure function of (frame, records): no scene, no engines, no wall-clock reads —
the same inputs always produce the same pixels, so the overlay is reproducible
from a telemetry JSONL alone.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from dinner_table.runtime.telemetry import STAGE_KEYS, TickRecord

_PANEL_W = 300
_LINE_H = 14
_PAD = 6
_TEXT = (255, 255, 255)
_DIM = (170, 170, 170)
_ACCENT = (90, 200, 255)
_OK = (120, 220, 120)
_BAD = (255, 110, 110)
_PANEL_BG = (10, 10, 10)
_BOX = (90, 200, 255)


def render_hud(
    frame_rgb: np.ndarray,
    records: list[TickRecord],
    devices: dict[str, str] | None = None,
) -> np.ndarray:
    """Composite the HUD over one (H, W, 3) uint8 frame.

    `records` is oldest-first (HudTelemetryCallback.recent() order); the newest
    record drives the status panel and boxes, the window drives the checklist.
    `devices` maps stage -> device badge (e.g. {"policy": "GPU"}) and defaults
    to no badges.
    """
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (H, W, 3), got {frame.shape}")
    img = Image.fromarray(np.ascontiguousarray(frame).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    if not records:
        return np.asarray(img)
    latest = records[-1]
    _draw_boxes(draw, latest)
    _draw_panel(draw, latest, devices or {})
    _draw_checklist(draw, img.height, records)
    return np.asarray(img)


def _draw_boxes(draw: ImageDraw.ImageDraw, record: TickRecord) -> None:
    for det in record.detections:
        try:
            x0, y0, x1, y1 = (float(v) for v in det["xyxy"])
        except (KeyError, TypeError, ValueError):
            continue
        draw.rectangle([x0, y0, x1, y1], outline=_BOX, width=2)
        label = f"{det.get('label', '?')} {float(det.get('confidence', 0.0)):.2f}"
        draw.rectangle([x0, max(0, y0 - 12), x0 + 8 * len(label), y0], fill=_BOX)
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=(0, 0, 0))


def _panel_lines(record: TickRecord, devices: dict[str, str]) -> list[tuple[str, tuple]]:
    skill = record.active_skill or ("done" if record.postcondition == "pass" else "idle")
    arm = record.active_arm or "-"
    lines: list[tuple[str, tuple]] = [
        (f"skill: {skill}  arm: {arm}", _TEXT),
        (_vlm_line(record), _DIM if not record.vlm_pending else _ACCENT),
    ]
    for key in STAGE_KEYS:
        ms = record.stage_ms.get(key, 0.0)
        badge = devices.get(key, "")
        suffix = f" [{badge}]" if badge else ""
        lines.append((f"{key}: {ms:6.2f} ms{suffix}", _DIM))
    if record.postcondition is not None:
        ok = record.postcondition == "pass"
        lines.append((f"check: {record.postcondition}", _OK if ok else _BAD))
    if record.recovered is not None:
        lines.append((f"recovered: {record.recovered}", _ACCENT))
    lines.append((f"tick: {record.step}", _DIM))
    return lines


def _vlm_line(record: TickRecord) -> str:
    if record.vlm_call is not None:
        return f"vlm: {record.vlm_call}" + (" (pending)" if record.vlm_pending else "")
    return "vlm: pending" if record.vlm_pending else "vlm: -"


def _draw_panel(draw: ImageDraw.ImageDraw, record: TickRecord, devices: dict[str, str]) -> None:
    lines = _panel_lines(record, devices)
    height = _PAD * 2 + _LINE_H * len(lines)
    draw.rectangle([0, 0, _PANEL_W, height], fill=_PANEL_BG + (200,))
    y = _PAD
    for text, color in lines:
        draw.text((_PAD, y), text, fill=color)
        y += _LINE_H


def _draw_checklist(draw: ImageDraw.ImageDraw, frame_h: int, records: list[TickRecord]) -> None:
    """Ordered unique skills seen in the window; past skills ticked off."""
    seen: list[str] = []
    for record in records:
        skill = record.active_skill
        if skill is not None and skill not in seen:
            seen.append(skill)
    if not seen:
        return
    current = records[-1].active_skill
    rows = [f"[x] {s}" if s != current else f"[>] {s}" for s in seen]
    height = _PAD * 2 + _LINE_H * len(rows)
    y0 = max(0, frame_h - height)
    draw.rectangle([0, y0, _PANEL_W, frame_h], fill=_PANEL_BG + (200,))
    y = y0 + _PAD
    for row in rows:
        color = _ACCENT if row.startswith("[>]") else _OK
        draw.text((_PAD, y), row, fill=color)
        y += _LINE_H
