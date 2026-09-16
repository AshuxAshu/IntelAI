"""Acceptance reporting must preserve failures and reject incomplete evidence."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "a6_reporting_subject", Path(__file__).with_name("test_a6_acceptance.py")
)
assert spec is not None and spec.loader is not None
a6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a6)
pytestmark = pytest.mark.fast


def test_failed_episode_keeps_contact_evidence(monkeypatch):
    scene = SimpleNamespace(hold_safe=lambda: None)
    audit = SimpleNamespace(errors={"cargo-world spoon_1/drawer_top"}, path_errors=set())
    monkeypatch.setattr(a6, "Scene", lambda **kwargs: scene)
    monkeypatch.setattr(a6, "TeacherContext", lambda scene: object())
    monkeypatch.setattr(a6, "Audit", lambda scene: audit)

    def fail(*args):
        raise AssertionError("deliberate failure after recorded contact")

    monkeypatch.setattr(a6, "_run", fail)
    row = a6._episode(4, "pick-spoon_1", "dr_train")
    assert row["ok"] is False
    assert "deliberate failure" in row["detail"]
    assert row["contacts"] == ["cargo-world spoon_1/drawer_top"]


def test_setup_failure_is_recorded_without_audit(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("scene setup failed")

    monkeypatch.setattr(a6, "Scene", fail)
    assert a6._episode(0, "pick-mug", "dr_train") == {
        "ok": False,
        "detail": "RuntimeError: scene setup failed",
        "contacts": [],
    }


def test_success_rate_cannot_hide_contact_failure():
    rows = [{"ok": True, "detail": "", "contacts": []} for _ in range(20)]
    rows[4]["contacts"] = ["cargo-world spoon_1/drawer_top"]
    with pytest.raises(AssertionError, match="audit failures"):
        a6._gate("pick-spoon_1", rows)


@pytest.mark.parametrize("condition,minimum", [("pick-mug", 20), ("place-mug", 19)])
def test_acceptance_rate_boundary(condition, minimum):
    rows = [
        {"ok": seed < minimum, "detail": "injected failure", "contacts": []}
        for seed in range(20)
    ]
    a6._gate(condition, rows)
    rows[minimum - 1]["ok"] = False
    with pytest.raises(AssertionError, match=f"need {minimum}/20"):
        a6._gate(condition, rows)


def test_aggregate_rejects_missing_records(monkeypatch, tmp_path):
    monkeypatch.setenv("A6_RESULTS", str(tmp_path))
    monkeypatch.delenv("A6_SEEDS", raising=False)
    with pytest.raises(AssertionError, match="missing"):
        a6.test_a6_aggregate(SimpleNamespace())
