"""A11: free label geometry, VQA schema, and corpus balance.

The module fixture builds event-rich demos (perturbed for anomaly records,
clean for next-step records), labels them headlessly, and emits a 60/12 VQA
corpus. Pixel rendering is the only GL step and no test here needs it:
geometry, schema, and balance all verify without a display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dinner_table.data.demo_gen import generate_episode
from dinner_table.data.labels import LabeledBox, label_frame, render_yolo_set
from dinner_table.data.lerobot_export import replay_frames
from dinner_table.data.vqa_factory import (
    ANOMALY_TEMPLATES,
    GROUND_TEMPLATES,
    NEXT_TEMPLATES,
    PARSE_NEGATIVE,
    PARSE_POSITIVE,
    SPATIAL_TEMPLATES,
    STATE_TEMPLATES,
    VAL_TEMPLATES,
    generate_vqa,
)
from dinner_table.perception.interfaces import OBJECT_LABELS
from dinner_table.policies.conditioning import SKILLS
from dinner_table.reasoning.schema import TaskGraph, VlmDiagnosis
from dinner_table.scene.builder import Scene

pytestmark = pytest.mark.slow

PERTURBED = [
    (1, "drawer_cycle"),
    (2, "plate_setting"),
    (5, "drawer_cycle"),
    (7, "mug_setting"),
    (15, "mug_setting"),
    (23, "mug_setting"),
    (27, "mug_setting"),
    (30, "plate_setting"),
    (9, "drawer_cycle"),
    (19, "mug_setting"),
    (29, "drawer_cycle"),
]
CLEAN = [
    (0, "plate_setting"),
    (6, "mug_setting"),
    (17, "drawer_cycle"),
    (4, "park_cycle"),
    (39, "park_cycle"),
]
TRAIN_N = 60
VAL_N = 12


@pytest.fixture(scope="module")
def vqa_corpus(tmp_path_factory):
    """Event-rich demos plus headless labels and a 60/12 VQA corpus."""
    demos = tmp_path_factory.mktemp("demos_vqa")
    entries = []
    for seed, graph in PERTURBED:
        entries.append(generate_episode(seed, graph, "dr_train", demos, probability=1.0))
    for seed, graph in CLEAN:
        entries.append(generate_episode(seed, graph, "dr_train", demos, probability=0.0))
    entries.sort(key=lambda e: e["seed"])
    manifest = {
        "config": "dr_train",
        "focus": None,
        "seed_base": -1,
        "profile": "dr_train",
        "episodes": entries,
        "successes": sum(1 for e in entries if e["success"]),
    }
    (demos / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    yolo = tmp_path_factory.mktemp("yolo_vqa")
    render_yolo_set(demos, yolo, render=False)
    out = tmp_path_factory.mktemp("vqa_out")
    generate_vqa(demos, yolo, out, train_n=TRAIN_N, val_n=VAL_N, seed=0, copy_images=False)
    graphs = {seed: graph for seed, graph in PERTURBED + CLEAN}
    return {"demos": demos, "yolo": yolo, "out": out, "manifest": manifest, "graphs": graphs}


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def _read_txt(path: Path) -> list[LabeledBox]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cls, xc, yc, w, h = line.split()
            boxes.append(
                LabeledBox(OBJECT_LABELS[int(cls)], float(xc), float(yc), float(w), float(h))
            )
    return boxes


def test_bbox_iou(vqa_corpus) -> None:
    """50 sampled frames: label-file boxes match live projection at IoU >= 0.9."""
    demos, yolo, graphs = vqa_corpus["demos"], vqa_corpus["yolo"], vqa_corpus["graphs"]
    stems = sorted(p.stem for p in (yolo / "labels").glob("*.txt"))
    assert len(stems) >= 50
    sample = [stems[i * len(stems) // 50] for i in range(50)]
    by_episode: dict[str, list[int]] = {}
    for stem in sample:
        episode_id, tick = stem.rsplit("_f", 1)
        by_episode.setdefault(episode_id, []).append(int(tick))
    checked = 0
    for episode_id, ticks in by_episode.items():
        entry = next(e for e in vqa_corpus["manifest"]["episodes"] if e["episode_id"] == episode_id)
        log = json.loads(
            (demos / entry["split"] / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        wanted = set(ticks)
        for frame, _cond, scene in replay_frames(log, graphs[entry["seed"]]):
            if int(frame["tick"]) not in wanted:
                continue
            stem = f"{episode_id}_f{int(frame['tick']):05d}"
            live = {
                b.cls: b.xyxy_pixels(640, 480)
                for b in label_frame(scene.model, scene.data, "overhead")
            }
            for box in _read_txt(yolo / "labels" / f"{stem}.txt"):
                assert box.cls in live, f"{stem}: missing live box for {box.cls}"
                iou = _iou(box.xyxy_pixels(640, 480), live[box.cls])
                assert iou >= 0.9, f"{stem} {box.cls}: IoU {iou:.4f}"
                checked += 1
    assert checked >= 50 * 8, f"only {checked} boxes checked"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_vqa_schema(vqa_corpus) -> None:
    """Every line parses; roles alternate; templates are split-disjoint."""
    train = _read_jsonl(vqa_corpus["out"] / "train.jsonl")
    val = _read_jsonl(vqa_corpus["out"] / "val.jsonl")
    assert len(train) == TRAIN_N
    assert len(val) == VAL_N
    for record in train + val:
        assert set(record) == {"id", "split", "type", "template", "image", "messages"}
        assert record["split"] in ("train", "val")
        assert [m["role"] for m in record["messages"]] == ["user", "assistant"]
        assert all(m["content"] for m in record["messages"])
        image = record["image"]
        assert image is None or (image.startswith("frames/") and image.endswith(".jpg"))
        assistant = (
            json.loads(record["messages"][1]["content"])
            if record["type"] in ("parse", "next_step", "anomaly")
            else None
        )
        if record["type"] == "parse" and assistant.get("refuse") is not True:
            TaskGraph(**assistant)
        if record["type"] == "anomaly":
            VlmDiagnosis(**assistant)
        if record["type"] == "next_step":
            assert "next_skill" in assistant or assistant.get("done") is True
    train_templates = {r["template"] for r in train}
    val_templates = {r["template"] for r in val}
    assert train_templates, "no train templates recorded"
    assert val_templates, "no val templates recorded"
    assert not (train_templates & val_templates), train_templates & val_templates


def test_vqa_balance(vqa_corpus) -> None:
    """Types within +-20% of even; negatives >= 12% of parse records."""
    for split, total in (("train", TRAIN_N), ("val", VAL_N)):
        records = _read_jsonl(vqa_corpus["out"] / f"{split}.jsonl")
        counts: dict[str, int] = {}
        for record in records:
            counts[record["type"]] = counts.get(record["type"], 0) + 1
        assert len(counts) == 6, counts
        even = total / 6.0
        for rtype, count in counts.items():
            assert abs(count - even) <= 0.20 * even, (split, rtype, count)
    parse_train = [
        r for r in _read_jsonl(vqa_corpus["out"] / "train.jsonl") if r["type"] == "parse"
    ]
    refusals = sum(
        1 for r in parse_train if json.loads(r["messages"][1]["content"]).get("refuse") is True
    )
    assert refusals / len(parse_train) >= 0.12, f"{refusals}/{len(parse_train)}"


@pytest.mark.fast
def test_projection_direction_and_range() -> None:
    """Projected boxes follow the camera: +x right, +y up, all in 0..1."""
    scene = Scene(seed=0, dr_profile="dr_train")
    scene.hold_safe()
    boxes = label_frame(scene.model, scene.data, "overhead")
    assert len(boxes) == 8
    names = sorted(b.cls for b in boxes)
    assert "drawer_closed" in names
    assert "drawer_open" not in names
    for box in boxes:
        for value in (box.xc, box.yc, box.w, box.h):
            assert 0.0 <= value <= 1.0, (box.cls, value)
        assert box.w > 0.0 and box.h > 0.0
    pos = {b.cls: scene.object_pose(b.cls)[0] for b in boxes if not b.cls.startswith("drawer_")}
    xc = {b.cls: b.xc for b in boxes}
    yc = {b.cls: b.yc for b in boxes}
    assert sorted(pos, key=lambda n: pos[n][0]) == sorted(pos, key=lambda n: xc[n])
    assert sorted(pos, key=lambda n: pos[n][1]) == sorted(pos, key=lambda n: yc[n], reverse=True)


@pytest.mark.fast
def test_yolo_label_files(tmp_path: Path) -> None:
    """One park episode labels headlessly with a valid data.yaml."""
    demos = tmp_path / "demos"
    entry = generate_episode(0, "park_cycle", "dr_train", demos, probability=0.0)
    (demos / "manifest.json").write_text(
        json.dumps({"episodes": [entry], "successes": 1}, indent=2), encoding="utf-8"
    )
    import yaml

    result = render_yolo_set(demos, tmp_path / "yolo", render=False)
    assert result["frames"] == 39
    labels = list((tmp_path / "yolo" / "labels").glob("*.txt"))
    assert len(labels) == 39
    assert not (tmp_path / "yolo" / "images").exists()
    data_yaml = yaml.safe_load((tmp_path / "yolo" / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["nc"] == len(OBJECT_LABELS)
    assert data_yaml["names"] == list(OBJECT_LABELS)
    for line in labels[0].read_text(encoding="utf-8").splitlines():
        cls, *coords = line.split()
        assert 0 <= int(cls) < len(OBJECT_LABELS)
        assert len(coords) == 4
        assert all(0.0 <= float(v) <= 1.0 for v in coords)


@pytest.mark.fast
def test_template_registry_is_disjoint_and_complete() -> None:
    """Every template tuple spans both splits; val ids are all real and used."""
    groups = [
        PARSE_POSITIVE,
        PARSE_NEGATIVE,
        GROUND_TEMPLATES,
        STATE_TEMPLATES,
        NEXT_TEMPLATES,
        ANOMALY_TEMPLATES,
        SPATIAL_TEMPLATES,
    ]
    all_ids = {t[0] for group in groups for t in group}
    assert VAL_TEMPLATES <= all_ids, VAL_TEMPLATES - all_ids
    for group in groups:
        ids = {t[0] for t in group}
        assert ids & VAL_TEMPLATES, f"no val template in {[t[0] for t in group][:3]}"
        assert ids - VAL_TEMPLATES, "no train template"
    parse_ids = {t[0] for t in PARSE_POSITIVE} | {t[0] for t in PARSE_NEGATIVE}
    assert len(PARSE_POSITIVE) + len(PARSE_NEGATIVE) == 21
    assert len(parse_ids) == 21


@pytest.mark.fast
def test_parse_expected_graphs_validate() -> None:
    """Every positive parse template's expected graph is a valid TaskGraph."""
    from dinner_table.data.demo_gen import GRAPHS
    from dinner_table.data.vqa_factory import _template_graph

    for tid, _cores, graph_id in PARSE_POSITIVE:
        graph = _template_graph(graph_id)
        assert isinstance(graph, TaskGraph), tid
        graph.model_dump(mode="json")
    assert set(GRAPHS) >= {"dinner_canonical", "pour_only", "park_cycle"}
    for tid, cores, reason in PARSE_NEGATIVE:
        assert cores and reason, tid


@pytest.mark.fast
def test_full_scale_parse_quotas_feasible() -> None:
    """Template x wrapper x state candidates cover the 8k/800 parse quotas."""
    from dinner_table.data.vqa_factory import _template_graph, _world_states

    for split, quota in (("train", 1334), ("val", 134)):
        pos = 0
        for tid, cores, graph_id in PARSE_POSITIVE:
            if (tid in VAL_TEMPLATES) != (split == "val"):
                continue
            pos += len(cores) * 6 * len(_world_states(_template_graph(graph_id).steps))
        neg = 0
        for tid, cores, _reason in PARSE_NEGATIVE:
            if (tid in VAL_TEMPLATES) != (split == "val"):
                continue
            neg += len(cores) * 6 * 4
        assert pos >= quota - round(quota * 0.15), (split, pos)
        assert neg >= round(quota * 0.15), (split, neg)


@pytest.mark.fast
def test_vqa_vocabularies_cover_skills_and_events() -> None:
    """Reason/diagnosis maps cover every skill and every perturbation event."""
    from dinner_table.data.noise import Perturber
    from dinner_table.data.vqa_factory import EVENT_DIAGNOSIS, STEP_DONE_WORDS, STEP_REASONS

    assert set(STEP_DONE_WORDS) == set(SKILLS)
    assert set(STEP_REASONS) == set(SKILLS)
    assert set(EVENT_DIAGNOSIS) == set(Perturber.EVENTS)
