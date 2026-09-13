"""Architectural boundary enforcement guarding against privileged-state leaks into deployed code."""

from __future__ import annotations

from pathlib import Path

TARGET_PACKAGES = ("runtime", "executor", "perception", "reasoning")
BANNED_STRINGS = (
    "object_pose(",
    "spawn_meta(",
    "from dinner_table.scene.builder import",
)


def test_privileged_state_isolation() -> None:
    """Validate that deployed modules never import or call privileged ground-truth state APIs."""
    repo_root = Path(__file__).resolve().parent.parent
    src_base = repo_root / "src" / "dinner_table"

    violations: list[str] = []

    for pkg_name in TARGET_PACKAGES:
        pkg_dir = src_base / pkg_name
        if pkg_dir.is_dir():
            for py_path in pkg_dir.rglob("*.py"):
                file_text = py_path.read_text(encoding="utf-8")
                for banned in BANNED_STRINGS:
                    if banned in file_text:
                        rel_path = py_path.relative_to(repo_root)
                        violations.append(f"{rel_path}: contains forbidden '{banned}'")

    assert len(violations) == 0, (
        f"Isolation boundary violations detected ({len(violations)}):\n"
        + "\n".join(violations)
    )
