"""CLI: fail if any asset file lacks a license-manifest row. Exit code 1 on violation."""

from __future__ import annotations

import pathlib
import sys

ASSETS = pathlib.Path("assets")
MANIFEST = ASSETS / "ASSETS_LICENSES.md"
EXTENSIONS = {".stl", ".obj", ".png", ".jpg", ".jpeg", ".json"}


def main() -> int:
    """Check asset files against the license manifest."""
    if MANIFEST.exists():
        text = MANIFEST.read_text(encoding="utf-8")
    else:
        text = ""
    missing = []
    for path in sorted(ASSETS.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in EXTENSIONS:
            continue
        if path.name == "ASSETS_LICENSES.md":
            continue
        if path.name not in text and str(path.relative_to(ASSETS)) not in text:
            missing.append(str(path))
    if missing:
        for m in missing:
            print(f"unlicensed asset: {m}", file=sys.stderr)
        return 1
    print("license manifest ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
