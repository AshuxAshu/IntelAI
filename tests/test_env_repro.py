"""Environment pinning checks (fast marker)."""

from __future__ import annotations

import importlib.metadata as md
import sys

import pytest

pytestmark = pytest.mark.fast


def test_mujoco_pinned():
    assert md.version("mujoco").startswith("3.2")


def test_pydantic_major():
    assert md.version("pydantic").startswith("2.")


@pytest.mark.skipif(sys.platform == "darwin", reason="Intel stack not installed on macOS")
def test_intel_stack_installed():
    for pkg in ("openvino", "physicalai", "physicalai-train"):
        md.version(pkg)  # raises PackageNotFoundError if missing
