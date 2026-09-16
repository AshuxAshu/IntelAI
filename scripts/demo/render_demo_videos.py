"""Render labelled multi-camera demonstration videos, one per (run, seed).

Layout — a labelled camera row over a live telemetry strip:

    +----------------------+----------------------+
    | demo_cam (3rd-person)| overhead             |
    +----------------------+----------------------+
    | OpenVINO / stats / progress / graph bands   |
    +---------------------------------------------+

Only "Seed N" is burned into the scene area (top-left); each panel is labelled
with its camera angle and the bottom strip carries the measured OpenVINO and
task-progress telemetry. The run's purpose is recorded in the manifest, not
drawn over the video.

The gripper-bracket (`wrist_A` / `wrist_B`) views are deliberately NOT rendered:
each panel costs one offscreen render per frame, so excluding them roughly
halves the render time. Add them back by listing them in PANELS.

Usage:
  uv run python scripts/demo/render_demo_videos.py --seeds 0-3
  uv run python scripts/demo/render_demo_videos.py --runs table_setting --seeds 0-9 --ov
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runs as run_registry
from engine import execute_run, make_scene
from ovstats import LiveOpenvino

PANEL_W, PANEL_H = 640, 480
STRIP_H = 250                      # telemetry strip below the camera grid
FPS = 25
OV_EVERY = 5                       # infer one frame in N (detector is ~80 ms here)
TRAIL_TICKS = 30                   # hold the final state for ~1.2 s
FONT_DIR = Path("/usr/share/fonts/TTF")

# Panels to render, in reading order, with their labels. Each panel costs one
# offscreen render per frame, so this list is also the render-time knob: the
# two gripper-bracket (wrist) views are deliberately omitted.
PANELS = (
    ("demo_cam", "demo_cam (third-person)"),
    ("overhead", "overhead"),
)
PANEL_COLS = 2
GRID_ROWS = (len(PANELS) + PANEL_COLS - 1) // PANEL_COLS
GRID_W, GRID_H = PANEL_W * PANEL_COLS, PANEL_H * GRID_ROWS
OUT_W, OUT_H = GRID_W, GRID_H + STRIP_H

BG = (22, 24, 28)
FG = (235, 238, 242)
DIM = (150, 158, 168)
ACCENT = (96, 200, 255)
OKC = (120, 220, 140)
FAILC = (255, 120, 120)
BAND = (34, 38, 45)
PLOT_BG = (16, 18, 21)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a DejaVu font, falling back to PIL's bitmap default."""
    for candidate in (FONT_DIR / name, FONT_DIR / "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default()


F_SMALL = _font("DejaVuSansMono.ttf", 15)
F_LABEL = _font("DejaVuSansMono-Bold.ttf", 20)
F_MED = _font("DejaVuSansMono-Bold.ttf", 22)
F_BIG = _font("DejaVuSans-Bold.ttf", 34)
F_TITLE = _font("DejaVuSansMono-Bold.ttf", 24)


class Recorder:
    """Pipe raw RGB frames into an ffmpeg H.264 encode."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait()


# Pre-rendered text is pasted, not re-rasterised: `Font.render` dominated the
# frame budget (30 TrueType draws/frame ~= 74% of compose). A paste is a C-level
# memcpy. Bounded so the varying strings (elapsed time) cannot grow it without
# limit.
_TEXT_CACHE: dict[tuple, Image.Image] = {}
_TEXT_CACHE_MAX = 900
_SHADOW_OFFSETS = ((-2, 0), (2, 0), (0, -2), (0, 2))


def _text_image(s: str, font, fill, shadow: bool) -> Image.Image:
    """An RGBA image of `s`, with a baked dark outline, memoised."""
    key = (s, font.path if hasattr(font, "path") else id(font), font.size, fill, shadow)
    img = _TEXT_CACHE.get(key)
    if img is not None:
        return img
    pad = 3
    box = font.getbbox(s)
    w = max(1, box[2] - box[0]) + 2 * pad
    h = max(1, box[3] - box[1]) + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ox, oy = pad - box[0], pad - box[1]
    if shadow:
        for dx, dy in _SHADOW_OFFSETS:
            d.text((ox + dx, oy + dy), s, font=font, fill=(0, 0, 0, 255))
    d.text((ox, oy), s, font=font, fill=(*fill, 255))
    if len(_TEXT_CACHE) >= _TEXT_CACHE_MAX:
        _TEXT_CACHE.clear()
    _TEXT_CACHE[key] = img
    return img


def _text(img: Image.Image, xy, s, font, fill=FG, shadow=True) -> None:
    """Paste pre-rendered text at xy (top-left of the glyph box)."""
    if not s:
        return
    glyph = _text_image(s, font, fill, shadow)
    img.paste(glyph, (int(xy[0]) - 3, int(xy[1]) - 3), glyph)


def _banner(img: Image.Image, xy, s, font) -> None:
    """Draw text on a filled black rectangle (panel titles, seed badge)."""
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), s, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x, y = xy
    draw.rectangle([x - 6, y - 5, x + w + 8, y + h + 7], fill=(0, 0, 0))
    _text(img, (x, y), s, font, FG, shadow=False)


def _sparkline(img: Image.Image, box, series: np.ndarray, colour=ACCENT) -> None:
    """Latency-vs-inference sparkline with p50/p95 guide lines."""
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=PLOT_BG, outline=(70, 78, 90))
    if series.size == 0:
        _text(img, (x0 + 8, y0 + 8), "openvino: waiting for first inference",
              F_SMALL, DIM)
        return
    vals = series[:200]
    lo, hi = float(vals.min()), float(vals.max())
    spread = max(hi - lo, 1e-6)
    points: list[tuple[int, int]] = []
    for i, v in enumerate(vals):
        px = x0 + int(i * (x1 - x0 - 2) / max(len(vals) - 1, 1))
        py = y1 - 2 - int((v - lo) * (y1 - y0 - 4) / spread)
        points.append((px, py))
    if len(points) > 1:
        draw.line(points, fill=colour)
    p50, p95 = float(np.percentile(vals, 50)), float(np.percentile(vals, 95))
    for value, tag, col in ((p95, f"p95 {p95:.1f} ms", FAILC), (p50, f"p50 {p50:.1f} ms", OKC)):
        py = y1 - 2 - int((value - lo) * (y1 - y0 - 4) / spread)
        draw.line([(x0, py), (x1, py)], fill=col)
        _text(img, (x0 + 6, min(py + 2, y1 - 20)), tag, F_SMALL, col, shadow=False)


def _bars(img: Image.Image, box, values: list[tuple[str, float]], colour=ACCENT) -> None:
    """Horizontal bar chart for cross-precision comparisons."""
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=PLOT_BG, outline=(70, 78, 90))
    _text(img, (x0 + 8, y0 + 6), "ACT latency by precision - CPU (recorded bench)",
          F_SMALL, DIM, shadow=False)
    if not values:
        return
    top = max(v for _, v in values) or 1.0
    row_h = max(14, (y1 - y0 - 34) // max(len(values), 1))
    for i, (label, v) in enumerate(values):
        y = y0 + 26 + i * row_h
        w = int((v / top) * (x1 - x0 - 150))
        draw.rectangle([x0 + 92, y, x0 + 92 + w, y + row_h - 5], fill=colour)
        _text(img, (x0 + 8, y), label, F_SMALL, FG, shadow=False)
        _text(img, (x0 + 96 + w + 6, y), f"{v:.1f} ms", F_SMALL, FG, shadow=False)


def _progress(img: Image.Image, box, steps, current: int, phase: str,
              seed: int, run_name: str, elapsed: float, status: str,
              status_colour, failed_step: int = -1) -> None:
    """Run name, step chips, phase, sim time and outcome."""
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=BAND, outline=(70, 78, 90))
    _text(img, (x0 + 10, y0 + 8), f"RUN: {run_name}", F_MED, ACCENT, shadow=False)
    _text(img, (x0 + 10, y0 + 40), f"t = {elapsed:6.1f} s", F_SMALL, DIM, shadow=False)

    chips_x = x0 + 10
    for i, st in enumerate(steps):
        skill, _arm, obj, _tgt = st
        name = f"{obj}" if obj else skill
        if i == failed_step:
            col, mark = FAILC, "X"
        elif i < current:
            col, mark = OKC, "+"
        elif i == current:
            col, mark = ACCENT, ">"
        else:
            col, mark = DIM, "."
        chip = f"[{mark} {skill} {name}]"
        _text(img, (chips_x, y0 + 68), chip, F_SMALL, col, shadow=False)
        chips_x += draw.textlength(chip, font=F_SMALL) + 14
        if chips_x > x1 - 120:
            chips_x = x0 + 10
            break

    _text(img, (x0 + 10, y0 + 108), f"phase: {phase or '-':<10s}  seed: {seed}",
          F_SMALL, FG, shadow=False)
    _text(img, (x1 - 260, y0 + 8), status, F_TITLE, status_colour, shadow=False)


# Elements that never change within a video (seed badge, panel labels and the
# recorded-bench bar chart) are baked once into small RGBA tiles and pasted.
# Tiles are used rather than one full-frame layer because a full-frame RGBA
# composite costs ~3.9 ms/frame against ~0.7 ms for the two small regions.
_STATIC_TILES: dict[int, tuple[tuple[int, int, Image.Image], ...]] = {}

# (x0, y0, x1, y1) regions the static content occupies, in output pixels.
_BADGE_BOX = (0, 0, 340, 84)
_LABEL_BOX = (0, 0, PANEL_W, PANEL_H)          # first panel's label sits under the badge
_BARS_BOX = (640, GRID_H, OUT_W, GRID_H + 100)


def _static_tiles(seed: int) -> tuple[tuple[int, int, Image.Image], ...]:
    """Per-video constant tiles: (x, y, image) to paste each frame."""
    cached = _STATIC_TILES.get(seed)
    if cached is not None:
        return cached

    def tile(box) -> Image.Image:
        x0, y0, x1, y1 = box
        return Image.new("RGBA", (max(1, x1 - x0), max(1, y1 - y0)), (0, 0, 0, 0))

    tiles: list[tuple[int, int, Image.Image]] = []

    # Seed badge, top-LEFT of the scene area, as requested. The badge takes the
    # corner, so the first panel's own label is placed below it to avoid overlap.
    badge = tile(_BADGE_BOX)
    _banner(badge, (10 - _BADGE_BOX[0], 8 - _BADGE_BOX[1]), f"Seed-{seed}",
            _font("DejaVuSans-Bold.ttf", 40))
    tiles.append((_BADGE_BOX[0], _BADGE_BOX[1], badge))

    labels = tile(_LABEL_BOX)
    for i, (_cam, label) in enumerate(PANELS):
        row, col = divmod(i, PANEL_COLS)
        x, y = col * PANEL_W - _LABEL_BOX[0], row * PANEL_H - _LABEL_BOX[1]
        label_y = y + (58 + 8 if i == 0 else 8)
        _banner(labels, (x + 10, label_y), label, F_LABEL)
    tiles.append((_LABEL_BOX[0], _LABEL_BOX[1], labels))

    # The recorded-bench bars are constant for the whole run.
    from ovstats import ACT_REFERENCE
    bars = tile(_BARS_BOX)
    _bars(bars, (650 - _BARS_BOX[0], GRID_H + 8 - _BARS_BOX[1],
                 OUT_W - 10 - _BARS_BOX[0], GRID_H + 96 - _BARS_BOX[1]),
          [(f"{r['label']:>4s}/{r['device']}", r["p50_ms"]) for r in ACT_REFERENCE])
    tiles.append((_BARS_BOX[0], _BARS_BOX[1], bars))

    result = tuple(tiles)
    _STATIC_TILES[seed] = result
    return result


def compose(
    panels: dict[str, Image.Image],
    *,
    seed: int,
    run_name: str,
    steps,
    current: int,
    phase: str,
    elapsed: float,
    status: str,
    status_colour,
    ov: LiveOpenvino,
    step_outcomes: list[str],
    failure: str = "",
    failed_step: int = -1,
) -> Image.Image:
    """Assemble one output frame from the rendered panels plus the telemetry."""
    canvas = Image.new("RGB", (OUT_W, OUT_H), BG)

    for i, (cam, _label) in enumerate(PANELS):
        row, col = divmod(i, PANEL_COLS)
        canvas.paste(panels[cam], (col * PANEL_W, row * PANEL_H))
    for x, y, tile_img in _static_tiles(seed):
        canvas.paste(tile_img, (x, y), tile_img)

    # --- honest outcome marking in the scene area ---
    draw = ImageDraw.Draw(canvas)
    if failed_step >= 0:
        # A failed step is never rendered as a success: red frame + the failing
        # step and its attributed cause, so the video cannot overstate the run.
        for i, (cam, _label) in enumerate(PANELS):
            row, col = divmod(i, PANEL_COLS)
            x, y = col * PANEL_W, row * PANEL_H
            draw.rectangle([x + 1, y + 1, x + PANEL_W - 2, y + PANEL_H - 2],
                           outline=FAILC, width=5)
        step = steps[failed_step]
        what = f"{step[0]} {step[2] or ''}".strip()
        line1 = f"STEP FAILED: {what} (arm {step[1]})"
        _banner(canvas, (10, GRID_H - 96), line1, F_BIG)
        _banner(canvas, (10, GRID_H - 48), failure or "unknown cause", F_MED)
    elif status == "COMPLETED":
        _banner(canvas, (10, GRID_H - 60), "ALL STEPS COMPLETED", F_BIG)
    top = GRID_H
    draw.rectangle([0, top, OUT_W, OUT_H], fill=BG)
    draw.line([(0, top), (OUT_W, top)], fill=(70, 78, 90))

    left = (10, top + 8, 640, top + STRIP_H - 8)
    right = (650, top + 8, OUT_W - 10, top + STRIP_H - 8)

    _sparkline(canvas, (left[0], left[1], left[2], left[1] + 104),
               ov.series() if ov is not None else np.asarray([]))
    y = left[1] + 112
    if ov is None:
        lines = ["OpenVINO sampling disabled (--no-ov); run without it for live figures."]
        colour = DIM
    elif ov.available:
        p50 = ov.p50_ms or 0.0
        p95 = ov.p95_ms or 0.0
        lines = [
            f"OpenVINO runtime   device={ov.device}   model={ov.label} (stand-in)",
            (f"inferences={ov.n_inferences:<5d} last={ov.last_ms or 0:6.2f} ms   "
             f"p50={p50:6.2f} ms   p95={p95:6.2f} ms   {ov.ips or 0:6.1f} ips"),
            f"host CPU (proc)={ov.cpu_pct * 100:5.1f}%   RSS={ov.last_rss_mb:7.1f} MB",
            f"top detections: {ov.last_labels or '-'}",
        ]
        colour = FG
    else:
        lines = ["OpenVINO runtime   UNAVAILABLE",
                 f"reason: {ov.error or 'not initialised'}",
                 "install openvino and fetch the stand-in YOLO weights"]
        colour = FAILC
    for i, line in enumerate(lines):
        _text(canvas, (left[0] + 4, y + i * 22), line, F_SMALL, colour, shadow=False)

    # The bar chart is part of the static overlay; only live values are drawn here.
    _progress(canvas, (right[0], right[1] + 88, right[2], right[3]),
              steps, current, phase, seed, run_name, elapsed, status, status_colour,
              failed_step=failed_step)
    return canvas


def render_run(run_name: str, seed: int, out_path: Path, profile: str,
               use_ov: bool, ov_device: str = "GPU", force: bool = False) -> dict:
    """Render one run for one seed; returns the manifest record."""
    steps = run_registry.RUNS[run_name]
    t_start = time.perf_counter()
    scene = make_scene(seed, profile)
    model, data = scene.model, scene.data

    renderers = {cam: mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
                 for cam, _ in PANELS}
    rec = Recorder(out_path, OUT_W, OUT_H, FPS)
    ov = LiveOpenvino(device=ov_device) if use_ov else None

    state = {"current": 0, "phase": "", "status": "running", "colour": ACCENT,
             "outcomes": ["pending"] * len(steps), "failed_step": -1, "failure": ""}
    infer_tick = {"n": 0}

    def draw_frame() -> None:
        panels = {}
        want_ov = ov is not None and (infer_tick["n"] % OV_EVERY == 0)
        for cam, _label in PANELS:
            r = renderers[cam]
            r.update_scene(data, camera=cam)
            img = r.render()
            panels[cam] = Image.fromarray(img)
            if want_ov and cam == "overhead":
                ov.sample(img)
        infer_tick["n"] += 1
        frame = compose(panels, seed=seed, run_name=run_name, steps=steps,
                        current=state["current"], phase=state["phase"],
                        elapsed=float(data.time), status=state["status"],
                        status_colour=state["colour"], ov=ov,
                        step_outcomes=state["outcomes"],
                        failure=state["failure"],
                        failed_step=state["failed_step"])
        rec.write(np.asarray(frame))

    def on_step(idx: int, result) -> None:
        state["outcomes"][idx] = result.outcome
        if result.outcome != "success":
            state["failed_step"] = idx
            state["failure"] = f"{result.phase}/{result.cause}" if result.cause else result.outcome
            state["status"] = "FAILED"
            state["colour"] = FAILC
        state["current"] = idx + 1

    try:
        execute_run(run_name, steps, seed, profile, scene=scene,
                    on_phase=_phase_hook(state, draw_frame), on_step=on_step,
                    on_tick=draw_frame)
        if state["failed_step"] >= 0:
            state["status"] = "FAILED"
            state["colour"] = FAILC
        else:
            state["status"] = "COMPLETED"
            state["colour"] = OKC
        for _ in range(TRAIL_TICKS):
            draw_frame()
    except Exception as exc:
        state["status"] = f"ERROR {type(exc).__name__}"
        state["colour"] = FAILC
        state["failure"] = f"{type(exc).__name__}: {exc}"[:90]
        for _ in range(TRAIL_TICKS):
            draw_frame()
        rec.close()
        for r in renderers.values():
            r.close()
        raise

    rec.close()
    for r in renderers.values():
        r.close()

    failed = [i for i, o in enumerate(state["outcomes"]) if o != "success"]
    record = {
        "run": run_name, "seed": seed, "profile": profile,
        "title": run_registry.RUN_TITLE.get(run_name, ""),
        "primary_object": run_registry.PRIMARY_OBJECT.get(run_name),
        "steps": [{"skill": s[0], "arm": s[1], "object": s[2],
                   "target": str(s[3]) if s[3] is not None else None,
                   "outcome": state["outcomes"][i]}
                  for i, s in enumerate(steps)],
        "success": not failed,
        "status": state["status"],
        "failed_step": state["failed_step"],
        "failure": state["failure"],
        "expected_failure": run_name in run_registry.EXPECTED_FAILURE,
        "frames": rec.frames,
        "sim_time_s": round(float(data.time), 2),
        "wall_s": round(time.perf_counter() - t_start, 1),
        "openvino": {
            "available": bool(ov and ov.available),
            "device": ov.device if ov else None,
            "model": ov.label if ov else None,
            "inferences": ov.n_inferences if ov else 0,
            "p50_ms": round(ov.p50_ms, 3) if (ov and ov.p50_ms) else None,
            "p95_ms": round(ov.p95_ms, 3) if (ov and ov.p95_ms) else None,
            "throughput_ips": round(ov.ips, 1) if (ov and ov.ips) else None,
        },
        "file": str(out_path),
    }
    return record


def _phase_hook(state: dict, draw):
    def hook(_skill: str, phase: str, _scene) -> None:
        state["phase"] = phase
        draw()
    return hook


def parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=sorted(run_registry.RUNS))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--profile", default="dr_train")
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo/videos"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--no-ov", dest="ov", action="store_false",
                        help="skip the live OpenVINO sampling")
    parser.add_argument("--ov-device", default="GPU",
                        help="OpenVINO device for the live detector (CPU | GPU | AUTO)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    manifest_path = args.manifest or (args.out / "manifest.json")
    manifest = {"profile": args.profile, "runs": {}}
    if manifest_path.is_file() and not args.force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"warning: ignoring unreadable manifest {manifest_path}: {exc}")
    manifest.setdefault("runs", {})

    total = len(args.runs) * len(seeds)
    done = 0
    for run_name in args.runs:
        for seed in seeds:
            done += 1
            out_path = args.out / run_name / f"seed{seed:02d}.mp4"
            if out_path.is_file() and out_path.stat().st_size > 0 and not args.force:
                print(f"[{done}/{total}] skip (exists) {out_path}", flush=True)
                continue
            record = render_run(run_name, seed, out_path, args.profile,
                                args.ov, ov_device=args.ov_device, force=args.force)
            manifest["runs"].setdefault(run_name, {})[str(seed)] = record
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[{done}/{total}] {'OK  ' if record['success'] else 'FAIL'} "
                  f"{run_name:14s} seed={seed:2d} {record['frames']:5d} frames "
                  f"{record['wall_s']:6.1f}s -> {out_path}", flush=True)
    print(f"\ndone: {done} videos under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
