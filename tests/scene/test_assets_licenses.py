"""Tests for asset existence, calibration schema, and license manifest integrity."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from dinner_table.contracts.geometry import SO101_JOINT_SUFFIXES


def test_license_manifest_check() -> None:
    """Run check_licenses.py and assert exit code is 0."""
    result = subprocess.run(
        [sys.executable, "scripts/check_licenses.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"check_licenses.py failed:\n{result.stderr}\n{result.stdout}"


def test_so101_calibration_json_exists_and_valid() -> None:
    """Assert calibration JSON exists and parses with all expected joint keys."""
    calib_path = pathlib.Path("assets/meshes/so101/so101_calibration.json")
    assert calib_path.is_file(), f"missing calibration file: {calib_path}"
    with open(calib_path, encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, dict), "calibration data must be a JSON object"
    for joint_suffix in SO101_JOINT_SUFFIXES:
        assert joint_suffix in data, f"missing joint key in calibration: {joint_suffix}"


def test_so101_stl_count() -> None:
    """Assert at least 8 STL files exist under assets/meshes/so101/."""
    so101_dir = pathlib.Path("assets/meshes/so101")
    stl_files = list(so101_dir.glob("*.stl"))
    assert len(stl_files) >= 8, f"expected at least 8 STL files in {so101_dir}, found {len(stl_files)}"


def test_texture_counts() -> None:
    """Assert minimum texture counts: table >= 5, floor >= 4, wall >= 3, placemat >= 3."""
    textures_dir = pathlib.Path("assets/textures")
    table_count = len(list((textures_dir / "table").glob("*.png")))
    floor_count = len(list((textures_dir / "floor").glob("*.png")))
    wall_count = len(list((textures_dir / "wall").glob("*.png")))
    placemat_count = len(list((textures_dir / "placemat").glob("*.png")))

    assert table_count >= 5, f"expected at least 5 table textures, found {table_count}"
    assert floor_count >= 4, f"expected at least 4 floor textures, found {floor_count}"
    assert wall_count >= 3, f"expected at least 3 wall textures, found {wall_count}"
    assert placemat_count >= 3, f"expected at least 3 placemat textures, found {placemat_count}"
