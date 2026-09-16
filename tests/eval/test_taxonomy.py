"""A12: failure taxonomy over synthetic logs (headless, fast)."""

from dinner_table.eval.taxonomy import TaxonomyReport, attribute, chunk_error_curve, confusion


def _rec(step_id, skill, outcome="success", phase="", cause=""):
    return {
        "step_id": step_id,
        "skill": skill,
        "outcome": outcome,
        "phase_at_failure": phase,
        "failure_cause": cause,
    }


def _fr(skill="pick", step_id="s0", event=""):
    return {"tick": 0, "skill": skill, "step_id": step_id, "event": event}


def test_attribute_exact_counts():
    logs = [
        {
            "success": True,
            "profile": "calibrated",
            "steps": [_rec("s0", "pick"), _rec("s1", "place")],
            "frames": [_fr("pick", "s0"), _fr("place", "s1")],
        },
        {
            "success": False,
            "profile": "clean",
            "steps": [
                _rec("s0", "pick"),
                _rec("s1", "grasp", "failed", "close", "slip"),
                _rec("s2", "place"),
            ],
            "frames": [_fr("pick", "s0"), _fr("grasp", "s1", event="kick")],
        },
        {
            "success": False,
            "profile": "calibrated",
            "steps": [_rec("s0", "pick", "failed", "approach", "timeout")],
            "frames": [_fr("pick", "s0")],
        },
    ]
    rep = attribute(logs)
    assert rep.episodes == 3
    assert rep.successes == 1
    assert rep.skills["pick"] == {"attempts": 3, "successes": 2, "rate": 2 / 3}
    assert rep.skills["grasp"] == {"attempts": 1, "successes": 0, "rate": 0.0}
    assert rep.skills["place"] == {"attempts": 2, "successes": 2, "rate": 1.0}
    assert rep.failures == [
        {"skill": "grasp", "phase": "close", "cause": "slip", "count": 1},
        {"skill": "pick", "phase": "approach", "cause": "timeout", "count": 1},
    ]
    assert rep.episodes_with_event == {
        "with_event": {"episodes": 1, "failures": 1},
        "without_event": {"episodes": 2, "failures": 1},
    }
    assert rep.failures_by_event == {"kick": 1}
    assert rep.profiles == {
        "calibrated": {"episodes": 2, "failures": 1},
        "clean": {"episodes": 1, "failures": 1},
    }


def test_chunk_error_curve_exact():
    frames = []
    for i in range(8):
        err = float(i % 4)
        pred = [0.0] * 12
        pred[0] = err
        frames.append({"action": [0.0] * 12, "predicted_action": pred})
    curve = chunk_error_curve({"frames": frames}, chunk_size=4)
    assert curve == [e * e / 12 for e in (0.0, 1.0, 2.0, 3.0)]


def test_chunk_error_curve_empty():
    assert chunk_error_curve({"frames": []}, chunk_size=4) == [0.0] * 4


def test_confusion_exact():
    log = {
        "steps": [_rec("s0", "pick"), _rec("s1", "pour")],
        "frames": [
            _fr("pick", "s0"),
            _fr("pick", "s0"),
            _fr("pick", "s0"),
            _fr("home", "s0"),
            _fr("home", "s0"),
            _fr("pour+hold", "s1"),
            _fr("pour+hold", "s1"),
            _fr("stir", "s1"),
        ],
    }
    assert confusion([log]) == {"pick->home": 2, "pour->stir": 1}


def test_attribute_reads_dr_profile_key():
    logs = [{"success": True, "dr_profile": "dr_train", "steps": [], "frames": []}]
    rep = attribute(logs)
    assert rep.profiles == {"dr_train": {"episodes": 1, "failures": 0}}


def test_report_json_shape_frozen():
    d = TaxonomyReport(episodes=0, successes=0).to_dict()
    assert sorted(d) == [
        "episodes",
        "episodes_with_event",
        "failures",
        "failures_by_event",
        "profiles",
        "skills",
        "successes",
    ]
