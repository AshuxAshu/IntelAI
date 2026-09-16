"""Fast tests for teacher report gating; these do not prove physics success."""

from __future__ import annotations

import json

import pytest

from dinner_table.eval import teacher_eval
from dinner_table.teacher.teacher_policy import EpisodeLog, StepRecord


def _report(name="dinner_full", success=1.0, first=1.0, episodes=None):
    return {"graphs": {name: {
        "episodes": [{}] if episodes is None else episodes,
        "success_rate": success,
        "first_attempt_rate": first,
    }}}


@pytest.mark.parametrize("name", ["dinner_canonical", "dinner_full", "drawer_cycle"])
def test_every_selected_graph_is_gated(name):
    violations = teacher_eval.threshold_violations(_report(name, success=0.5, first=0.5))
    assert len(violations) == 2
    assert all(name in violation for violation in violations)


def test_threshold_boundaries():
    assert teacher_eval.threshold_violations(_report(success=0.98, first=0.95)) == []
    assert len(teacher_eval.threshold_violations(_report(first=0.94))) == 1
    assert len(teacher_eval.threshold_violations(_report(success=0.97))) == 1


def test_empty_report_fails():
    assert teacher_eval.threshold_violations({"graphs": {}})
    assert teacher_eval.threshold_violations(_report(episodes=[]))


@pytest.mark.parametrize("names,seeds", [([], [0]), (["dinner_full"], [])])
def test_empty_evaluation_rejected(names, seeds):
    with pytest.raises(ValueError, match="at least one graph and one seed"):
        teacher_eval.evaluate(names, seeds, "dr_train")


@pytest.mark.parametrize("success,expected_exit", [(False, 1), (True, 0)])
def test_cli_gates_full_graph(monkeypatch, tmp_path, success, expected_exit):
    class FakeScene:
        def __init__(self, **kwargs):
            pass

        def hold_safe(self):
            pass

    def fake_run(scene, graph, seed):
        return EpisodeLog(
            episode_id="test", seed=seed, dr_profile="dr_train",
            instruction=graph.instruction, task_id=graph.task_id,
            steps=[StepRecord(1, "pick", "A", "success" if success else "failed", 1)],
            success=success,
        )

    monkeypatch.setattr(teacher_eval, "Scene", FakeScene)
    monkeypatch.setattr(teacher_eval, "run_graph", fake_run)
    out = tmp_path / "report.json"
    code = teacher_eval.main(["--graphs", "dinner_full", "--seeds", "0", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == expected_exit
    assert report["ok"] is success
    assert report["thresholds"]["gated_graphs"] == ["dinner_full"]
    assert bool(report["violations"]) is not success


def test_retry_is_not_first_attempt():
    log = EpisodeLog(
        episode_id="test", seed=0, dr_profile="dr_train", instruction="test",
        task_id="test", success=True,
        steps=[StepRecord(1, "pick", "A", "success", 2)],
    )
    summary = teacher_eval.episode_summary(log)
    assert summary["success"]
    assert not summary["first_attempt"]
