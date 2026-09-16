"""Physical canonical-task regressions; multi-seed acceptance remains separate."""

import gc
import json

import pytest

from dinner_table.eval.teacher_eval import episode_summary
from dinner_table.scene.builder import Scene
from dinner_table.teacher.task_graphs import CANONICAL_GRAPHS
from dinner_table.teacher.teacher_policy import run_graph


@pytest.mark.slow
def test_canonical_twenty_seed_acceptance(tmp_path):
    episodes = []
    report_path = tmp_path / "canonical_acceptance.json"
    for seed in range(20):
        scene = Scene(seed=seed, dr_profile="dr_train")
        scene.hold_safe()
        log = run_graph(scene, CANONICAL_GRAPHS["dinner_canonical"], seed)
        episode = episode_summary(log)
        position, _ = scene.object_pose("plate")
        episode["final_state_ok"] = bool(
            scene.fill_fraction("mug") >= 0.6
            and abs(position[2] - 0.36) < 0.003
            and (position[0] + 0.06) ** 2 + (position[1] + 0.095) ** 2 <= 0.006 ** 2
            and scene.data.qpos[0] < 0.12 * 0.116
        )
        episodes.append(episode)
        report_path.write_text(json.dumps({"dr_profile": "dr_train", "episodes": episodes},
                                          indent=2))
        print(json.dumps(episode), flush=True)
        del log, scene
        gc.collect()
    first_attempt = sum(e["first_attempt"] and e["final_state_ok"] for e in episodes)
    with_retry = sum(e["success"] and e["final_state_ok"] for e in episodes)
    assert first_attempt >= 19 and with_retry == 20, (
        f"canonical first-attempt={first_attempt}/20; with-retry={with_retry}/20; "
        f"report={report_path}; episodes={episodes}"
    )


@pytest.mark.slow
def test_canonical_plate_prefix_clears_drawer():
    graph = CANONICAL_GRAPHS["dinner_canonical"]
    prefix = graph.model_copy(update={"steps": graph.steps[:4]})
    assert [step.skill for step in prefix.steps] == [
        "open_drawer", "close_drawer", "pick", "place",
    ]
    scene = Scene(seed=0, dr_profile="dr_train")
    scene.hold_safe()
    log = run_graph(scene, prefix, 0)
    assert log.success, log.to_dict()
    assert all(step.attempts == 1 for step in log.steps)
    position, _ = scene.object_pose("plate")
    assert abs(position[2] - 0.36) < 0.003
    assert (position[0] + 0.06) ** 2 + (position[1] + 0.095) ** 2 <= 0.006 ** 2
    assert scene.data.qpos[0] < 0.12 * 0.116


@pytest.mark.slow
def test_canonical_seed_zero_pours_on_first_attempt():
    scene = Scene(seed=0, dr_profile="dr_train")
    scene.hold_safe()
    log = run_graph(scene, CANONICAL_GRAPHS["dinner_canonical"], 0)
    assert log.success, log.to_dict()
    assert all(step.attempts == 1 for step in log.steps)
    assert scene.fill_fraction("mug") >= 0.6
    position, _ = scene.object_pose("plate")
    assert abs(position[2] - 0.36) < 0.003
    assert (position[0] + 0.06) ** 2 + (position[1] + 0.095) ** 2 <= 0.006 ** 2
    assert scene.data.qpos[0] < 0.12 * 0.116
