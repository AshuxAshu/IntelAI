"""CLI tool rendering a multi-camera contact sheet of the simulated scene."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

from dinner_table.scene.builder import Scene

FONT_5X7: dict[str, tuple[str, ...]] = {
    "A": ("..#..", ".#.#.", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "_": (".....", ".....", ".....", ".....", ".....", ".....", "#####"),
    "a": (".....", ".....", ".###.", "#...#", "#####", "#...#", "#...#"),
    "c": (".....", ".....", ".###.", "#....", "#....", "#....", ".###."),
    "d": ("...#.", "...#.", ".###.", "#...#", "#...#", "#...#", ".###."),
    "e": (".....", ".....", ".###.", "#....", "####.", "#....", ".###."),
    "h": ("#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#"),
    "i": ("..#..", ".....", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "m": (".....", ".....", "##.#.", "#.#.#", "#.#.#", "#...#", "#...#"),
    "o": (".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###."),
    "r": (".....", ".....", "#.##.", "##..#", "#....", "#....", "#...."),
    "s": (".....", ".....", ".####", "#....", ".###.", "....#", "####."),
    "t": ("..#..", "..#..", ".###.", "..#..", "..#..", "..#..", "..###"),
    "v": (".....", ".....", "#...#", "#...#", "#...#", "#...#", ".#.#."),
    "w": (".....", ".....", "#...#", "#...#", "#...#", "#.#.#", ".#.#."),
}


def _draw_label(img: np.ndarray, label: str) -> np.ndarray:
    """Overlay a black header bar with white label text atop the image."""
    scale = 2
    pad = 3
    bar_height = 7 * scale + 2 * pad
    labeled = img.copy()
    labeled[:bar_height, :] = (0, 0, 0)
    row = pad
    col = pad
    for char in label:
        glyph = FONT_5X7.get(char)
        if glyph is not None:
            for gy in range(7):
                for gx in range(5):
                    if glyph[gy][gx] == "#":
                        y0 = row + gy * scale
                        x0 = col + gx * scale
                        labeled[y0 : y0 + scale, x0 : x0 + scale] = (255, 255, 255)
        col += 6 * scale
    return labeled


def _write_png(buf: np.ndarray, file_path: Path) -> None:
    """Write uint8 RGB numpy array to PNG file without external dependencies."""
    try:
        from PIL import Image

        img = Image.fromarray(buf)
        img.save(file_path)
        return
    except ImportError:
        pass

    height, width, _ = buf.shape
    raw_bytes = b"".join(b"\x00" + buf[y].tobytes() for y in range(height))
    compressed = zlib.compress(raw_bytes)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr)
    idat_crc = zlib.crc32(b"IDAT" + compressed)
    iend_crc = zlib.crc32(b"IEND")

    with open(file_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", ihdr_crc))
        f.write(
            struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
        )
        f.write(struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc))


def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize uint8 3D image using nearest-neighbor interpolation."""
    src_h, src_w, _ = img.shape
    row_idx = (np.arange(target_h) * src_h // target_h).astype(int)
    col_idx = (np.arange(target_w) * src_w // target_w).astype(int)
    return img[row_idx[:, None], col_idx]


def build_contact_sheet(
    seed: int = 42, profile: str = "dr_train", out_path: Path | None = None
) -> Path:
    """Build the scene, render all four cameras, and save a labeled 2x2 grid PNG."""
    scene = Scene(seed=seed, dr_profile=profile)

    img_overhead = scene.render("overhead")
    img_demo = scene.render("demo_cam")
    img_wrist_a = scene.render("wrist_A")
    img_wrist_b = scene.render("wrist_B")

    h, w, _ = img_overhead.shape

    wrist_a_scaled = _resize_nearest(img_wrist_a, h, w)
    wrist_b_scaled = _resize_nearest(img_wrist_b, h, w)

    labeled_overhead = _draw_label(img_overhead, "overhead")
    labeled_demo = _draw_label(img_demo, "demo_cam")
    labeled_wrist_a = _draw_label(wrist_a_scaled, "wrist_A")
    labeled_wrist_b = _draw_label(wrist_b_scaled, "wrist_B")

    # Assemble 2x2 grid: top row [overhead, demo], bottom row [wrist_A, wrist_B]
    top_row = np.concatenate([labeled_overhead, labeled_demo], axis=1)
    bottom_row = np.concatenate([labeled_wrist_a, labeled_wrist_b], axis=1)
    grid = np.concatenate([top_row, bottom_row], axis=0)

    if out_path is None:
        target_dir = Path("artifacts")
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / f"scene_{seed}.png"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = out_path

    _write_png(grid, final_path)
    return final_path


def main() -> None:
    """CLI entrypoint for contact sheet generation."""
    parser = argparse.ArgumentParser(
        description="Render a 2x2 camera contact sheet for a simulation seed."
    )
    parser.add_argument("--seed", type=int, default=42, help="Randomization seed (default: 42)")
    parser.add_argument(
        "--profile", type=str, default="dr_train", help="DR profile name (default: dr_train)"
    )
    parser.add_argument("--out", type=str, default=None, help="Output path for the PNG file")
    args = parser.parse_args()

    if args.out is not None:
        target = Path(args.out)
    else:
        target = None

    saved_path = build_contact_sheet(seed=args.seed, profile=args.profile, out_path=target)
    print(f"Contact sheet saved to: {saved_path}")


if __name__ == "__main__":
    main()
