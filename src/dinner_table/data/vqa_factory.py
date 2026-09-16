"""VQA factory: six instruction-tuning record types from logs plus labels.

Records follow the Phase 3 example shape (id/split/image/messages) with two
added keys: ``type`` and ``template`` — the balance and template-disjointness
gates need them. Splits are respected two ways: frame-sourced records inherit
their episode's split, and val templates are an explicitly disjoint set (never
a hash that could starve a type). Parse variety comes from template cores
times speech-style wrappers times simulated world states.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.data.demo_gen import GRAPHS
from dinner_table.data.lerobot_export import attribute_frames
from dinner_table.reasoning.schema import RelativeTarget, Step, TaskGraph, VlmDiagnosis

logger = logging.getLogger(__name__)

TRAIN_N = 8000
VAL_N = 800
TYPES = ("parse", "ground", "state", "next_step", "anomaly", "spatial")
WRAPPERS = ("", "Please ", "Hey robot, ", "Could you ", "Um, ", "Robot, ")
VAL_TEMPLATES = {
    "p08",
    "p09",
    "n09",
    "g3",
    "s1",
    "s3",
    "x2",
    "a2",
    "r3",
}

PARSE_POSITIVE = (
    (
        "p00",
        ("set the table for dinner", "lay out the full dinner setting", "lay the table for dinner"),
        "dinner_canonical",
    ),
    ("p01", ("set a plate at placemat 1", "put a plate on placemat 1"), "plate_setting"),
    ("p02", ("set a mug at placemat 2", "put a mug on placemat 2"), "mug_setting"),
    ("p03", ("set the cutlery pair", "lay out a fork and spoon"), "cutlery_pair"),
    ("p04", ("open the drawer and close it again", "cycle the drawer"), "drawer_cycle"),
    ("p05", ("relay the bottle across the table", "hand the bottle over"), "bottle_relay"),
    ("p06", ("pour water into the mug", "fill the mug from the bottle"), "pour_only"),
    ("p07", ("park both arms", "stow the arms at home"), "park_cycle"),
    ("p08", ("set the table for two", "give each seat a fork"), "dinner_full"),
    ("p09", ("put a fork beside the plate", "set a fork left of the plate"), "fork_beside"),
    ("p10", ("pick up the plate on the left side", "grab the left-side plate"), "plate_setting"),
)
PARSE_NEGATIVE = (
    ("n00", ("bring me the wine glass", "fetch the wine glass"), "no wine glass exists"),
    ("n01", ("pick up the knife", "grab the knife"), "no knife exists"),
    ("n02", ("fetch the napkin", "bring a napkin"), "no napkin exists"),
    ("n03", ("move the bowl", "slide the bowl over"), "no bowl exists"),
    ("n04", ("pour the drawer into the mug",), "the drawer cannot be poured"),
    ("n05", ("lift the table", "raise the table"), "the table cannot be lifted"),
    ("n06", ("put the bottle inside the fork",), "the fork holds nothing"),
    ("n07", ("open the mug", "open the plate"), "vessels do not open"),
    ("n08", ("move it over there", "put this next to that"), "ambiguous referent"),
    ("n09", ("grab that thing", "hand me that one"), "ambiguous referent"),
)
GROUND_TEMPLATES = (
    ("g0", "where is the {object}?"),
    ("g1", "Locate the {object}."),
    ("g2", "Find the {object} in the image."),
    ("g3", "Which region contains the {object}?"),
)
STATE_TEMPLATES = (
    ("s0", "is the drawer open?"),
    ("s1", "Is the drawer open or closed?"),
    ("s2", "which arm holds the {object}?"),
    ("s3", "Is the {object} being held, and by which arm?"),
)
ANOMALY_TEMPLATES = (
    ("a0", "The teacher log flags event '{event}' at this tick. Diagnose the anomaly."),
    ("a1", "An injected event ('{event}') fired on this frame. What is the diagnosis?"),
    ("a2", "Given the flagged event '{event}', produce the anomaly diagnosis."),
)
NEXT_TEMPLATES = (
    ("x0", "What is the next step, and which arm executes it?"),
    ("x1", "Given the completed steps, what comes next?"),
    ("x2", "What should the robot do now?"),
)
SPATIAL_TEMPLATES = (
    ("r0", "is the {a} on the left or right half of the table?"),
    ("r1", "what is beside the {a}?"),
    ("r2", "Is the {a} above or below the {b}?"),
    ("r3", "Which object is closest to the {a}?"),
)
EVENT_DIAGNOSIS = {
    "object_kick": ("object_moved", "retry_skill"),
    "mid_skill_retarget": ("object_moved", "retry_skill"),
    "gripper_slip": ("grasp_lost", "retry_skill"),
    "waypoint_jitter": ("none", "retry_skill"),
    "drawer_friction_spike": ("drawer_jammed", "retry_skill"),
}
STEP_DONE_WORDS = {
    "pick": "picked up by arm {arm}",
    "place": "placed at {target}",
    "open_drawer": "drawer opened",
    "close_drawer": "drawer closed",
    "handoff": "handed to arm {arm}",
    "hold": "held by arm {arm}",
    "pour": "water poured into the mug",
    "home": "arm {arm} parked",
    "retract": "arm {arm} parked",
}
STEP_REASONS = {
    "pick": "the {object} must be secured before it can be moved.",
    "place": "the {object} is held and {target} is clear.",
    "open_drawer": "the utensils inside are needed next.",
    "close_drawer": "the drawer must shut before arms move over it.",
    "handoff": "the bottle must change hands for the placement.",
    "hold": "arm {arm} must keep holding the {object} while the other arm works.",
    "pour": "the mug is held and ready for water.",
    "home": "arm {arm} must park clear of the workspace.",
    "retract": "arm {arm} must park clear of the workspace.",
}
REGION_WORDS = (
    ("upper-left", "upper", "upper-right"),
    ("middle-left", "center", "middle-right"),
    ("lower-left", "lower", "lower-right"),
)


class VqaError(DinnerTableError):
    """Exception raised for VQA factory failures."""


def _quotas(total: int) -> list[int]:
    base, extra = divmod(total, len(TYPES))
    return [base + (1 if i < extra else 0) for i in range(len(TYPES))]


def _target_words(target) -> str:
    if target is None:
        return "its destination"
    if isinstance(target, str):
        return target
    return f"{target.relation.replace('_', ' ')} {target.anchor}"


def _describe_step(skill: str, arm: str, obj: str | None, target) -> str:
    words = STEP_DONE_WORDS[skill].format(arm=arm, target=_target_words(target))
    if obj is not None and skill in ("pick", "place"):
        return f"{obj} {words}"
    return words


def _load_demos(demos: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    manifest = json.loads((demos / "manifest.json").read_text(encoding="utf-8"))
    logs: dict[str, dict] = {}
    entries: dict[str, dict] = {}
    for entry in manifest["episodes"]:
        log = json.loads(
            (demos / entry["split"] / f"{entry['episode_id']}.json").read_text(encoding="utf-8")
        )
        logs[entry["episode_id"]] = log
        entries[entry["episode_id"]] = entry
    return manifest, logs, entries


def _read_boxes(labels_dir: Path) -> dict[str, list[tuple[str, float, float, float, float]]]:
    from dinner_table.perception.interfaces import OBJECT_LABELS

    frames: dict[str, list] = {}
    for path in sorted(labels_dir.glob("*.txt")):
        boxes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls, xc, yc, w, h = line.split()
                boxes.append((OBJECT_LABELS[int(cls)], float(xc), float(yc), float(w), float(h)))
        frames[path.stem] = boxes
    return frames


def _region(xc: float, yc: float) -> str:
    col = min(2, int(xc * 3.0))
    row = min(2, int(yc * 3.0))
    return REGION_WORDS[row][col]


def _size(w: float, h: float) -> str:
    area = w * h
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def _fork_beside_graph() -> TaskGraph:
    return TaskGraph(
        task_id="fork_beside",
        instruction="Put a fork beside the plate.",
        steps=[
            Step(id=1, skill="pick", arm="A", object="fork_1"),
            Step(
                id=2,
                skill="place",
                arm="A",
                object="fork_1",
                target=RelativeTarget(relation="left_of", anchor="plate"),
            ),
        ],
    )


def _template_graph(graph_id: str) -> TaskGraph:
    if graph_id == "fork_beside":
        return _fork_beside_graph()
    return GRAPHS[graph_id]


def _world_states(steps: list | None) -> list[dict]:
    # NOTE: states simulate carrying through completed prefixes of the template's
    # own graph, so parse contexts stay consistent with the expected TaskGraph.
    states = []
    prefixes: list[list] = [[]]
    if steps is not None:
        prefixes = [steps[:k] for k in range(len(steps) + 1)]
    for prefix in prefixes:
        held: dict[str, str] = {}
        for step in prefix:
            if step.skill == "pick" and step.object is not None:
                held[step.object] = step.arm
            elif step.skill == "place" and step.object is not None:
                held.pop(step.object, None)
            elif step.skill == "handoff" and step.object is not None:
                held[step.object] = step.arm
        for drawer_open in (False, True):
            objects = {}
            for name in ("plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2"):
                where = "in the drawer" if name.startswith(("fork", "spoon")) else "on the table"
                objects[name] = {"visible": True, "where": where, "held_by": held.get(name)}
            states.append(
                {
                    "objects": objects,
                    "drawer_open": drawer_open,
                    "completed_step_ids": [s.id for s in prefix],
                }
            )
    return states


def _parse_records(rng: np.random.Generator, split: str, quota: int) -> list[dict]:
    positive: list[tuple] = []
    for tid, cores, graph_id in PARSE_POSITIVE:
        if (tid in VAL_TEMPLATES) != (split == "val"):
            continue
        graph = _template_graph(graph_id)
        expected = graph.model_dump(mode="json")
        for core in cores:
            for wrap in WRAPPERS:
                for state in _world_states(graph.steps):
                    instruction = f"{wrap}{core}".strip()
                    state = {**state, "instruction": instruction}
                    positive.append((tid, instruction, state, expected))
    negative: list[tuple] = []
    for tid, cores, reason in PARSE_NEGATIVE:
        if (tid in VAL_TEMPLATES) != (split == "val"):
            continue
        for core in cores:
            for wrap in WRAPPERS:
                for state in _world_states(None)[:4]:
                    instruction = f"{wrap}{core}".strip()
                    state = {**state, "instruction": instruction}
                    refusal = {"refuse": True, "reason": reason}
                    negative.append((tid, instruction, state, refusal))
    neg_quota = round(quota * 0.15)
    if len(positive) < quota - neg_quota or len(negative) < neg_quota:
        raise VqaError(f"parse {split}: {len(positive)} pos / {len(negative)} neg")
    rng.shuffle(positive)
    rng.shuffle(negative)
    records = []
    for tid, instruction, state, expected in positive[: quota - neg_quota] + negative[:neg_quota]:
        user = (
            f"Instruction: '{instruction}'. Scene: {json.dumps(state, sort_keys=True)}. "
            "Respond with the TaskGraph JSON."
        )
        records.append(
            {
                "template": tid,
                "image": None,
                "user": user,
                "assistant": json.dumps(expected, sort_keys=True),
            }
        )
    return records


def _frame_sources(frames: dict, entries: dict, split: str) -> list[tuple[str, str, list]]:
    sources = []
    for stem, boxes in frames.items():
        episode_id = stem.rsplit("_f", 1)[0]
        if entries.get(episode_id, {}).get("split") != split:
            continue
        sources.append((stem, episode_id, boxes))
    return sources


def _split_templates(all_templates: tuple, split: str) -> list[tuple[str, str]]:
    return [t for t in all_templates if (t[0] in VAL_TEMPLATES) == (split == "val")]


def _ground_records(rng: np.random.Generator, sources, split: str, quota: int) -> list[dict]:
    templates = _split_templates(GROUND_TEMPLATES, split)
    records = []
    for stem, _episode, boxes in sources:
        objects = sorted({cls for cls, *_ in boxes if not cls.startswith("drawer_")})
        for obj in objects:
            box = next(b for b in boxes if b[0] == obj)
            records.append((stem, obj, box))
    rng.shuffle(records)
    out = []
    for stem, obj, (_cls, xc, yc, w, h) in records[:quota]:
        tid, form = templates[len(out) % len(templates)]
        answer = f"The {obj} is in the {_region(xc, yc)} region of the image ({_size(w, h)})."
        out.append(
            {
                "template": tid,
                "image": f"frames/{stem}.jpg",
                "user": f"<image>{form.format(object=obj)}",
                "assistant": answer,
            }
        )
    if len(out) < quota:
        raise VqaError(f"ground {split}: {len(out)} candidates for quota {quota}")
    return out


def _held_at_frame(log: dict, entries_graph: str, tick: int) -> dict[str, str]:
    steps = GRAPHS[entries_graph].steps
    by_id = {s.id: s for s in steps}
    frame = next(f for f in log["frames"] if f["tick"] == tick)
    order = [s["step_id"] for s in log["steps"]]
    current = order.index(frame["step_id"]) if frame["step_id"] in order else len(order)
    held: dict[str, str] = {}
    for record in log["steps"][:current]:
        step = by_id.get(record["step_id"])
        if step is None or record["outcome"] != "success":
            continue
        if step.skill == "pick" and step.object is not None:
            held[step.object] = step.arm
        elif step.skill == "place" and step.object is not None:
            held.pop(step.object, None)
        elif step.skill == "handoff" and step.object is not None:
            held[step.object] = step.arm
    if frame["skill"] == "pick":
        step = by_id.get(frame["step_id"])
        if (
            step is not None
            and step.object is not None
            and frame["phase"] in ("pregrasp", "approach", "close")
        ):
            held.pop(step.object, None)
    return held


def _state_records(
    rng: np.random.Generator, sources, logs, entries, split: str, quota: int
) -> list[dict]:
    records = []
    for stem, episode_id, boxes in sources:
        log = logs[episode_id]
        tick = int(stem.rsplit("_f", 1)[1])
        drawer = next((cls for cls, *_ in boxes if cls.startswith("drawer_")), None)
        if drawer is not None:
            if drawer == "drawer_open":
                answer = "The drawer is open."
            else:
                answer = "The drawer is closed."
            records.append((stem, "drawer", answer))
        held = _held_at_frame(log, entries[episode_id]["graph"], tick)
        for obj, arm in sorted(held.items()):
            records.append((stem, obj, f"Arm {arm} holds the {obj}."))
        if not held:
            records.append((stem, "plate", "No arm is holding the plate."))
    rng.shuffle(records)
    forms = dict(_split_templates(STATE_TEMPLATES, split))
    drawer_tid = "s1" if split == "val" else "s0"
    held_tid = "s3" if split == "val" else "s2"
    out = []
    for stem, key, answer in records[:quota]:
        if key == "drawer":
            tid, question = drawer_tid, forms[drawer_tid]
        else:
            tid, question = held_tid, forms[held_tid].format(object=key)
        out.append(
            {
                "template": tid,
                "image": f"frames/{stem}.jpg",
                "user": f"<image>{question}",
                "assistant": answer,
            }
        )
    if len(out) < quota:
        raise VqaError(f"state {split}: {len(out)} candidates for quota {quota}")
    return out


def _next_records(logs, entries, split: str, quota: int) -> list[dict]:
    candidates = []
    for episode_id, log in logs.items():
        if entries[episode_id]["split"] != split or not log["success"]:
            continue
        graph = GRAPHS[entries[episode_id]["graph"]]
        by_id = {s.id: s for s in graph.steps}
        done: list[str] = []
        for record in log["steps"]:
            step = by_id.get(record["step_id"])
            if step is None:
                continue
            history = "; ".join(done) if done else "nothing yet"
            reason = STEP_REASONS[step.skill].format(
                arm=step.arm, object=step.object or "", target=_target_words(step.target)
            )
            target = step.target
            if target is not None and not isinstance(target, str):
                target = {"relation": target.relation, "anchor": target.anchor}
            nxt = {
                "next_skill": step.skill,
                "arm": step.arm,
                "object": step.object,
                "target": target,
                "reason": reason,
            }
            candidates.append((log["instruction"], history, nxt))
            done.append(_describe_step(step.skill, step.arm, step.object, step.target))
        history = "; ".join(done)
        candidates.append(
            (
                log["instruction"],
                history,
                {"done": True, "reason": f"all {len(done)} steps are complete."},
            )
        )
    if len(candidates) < quota:
        raise VqaError(f"next_step {split}: {len(candidates)} candidates for quota {quota}")
    templates = _split_templates(NEXT_TEMPLATES, split)
    out = []
    for instruction, history, nxt in candidates[:quota]:
        tid, question = templates[len(out) % len(templates)]
        user = f"Instruction: '{instruction}'. Done so far: {history}. {question}"
        out.append(
            {
                "template": tid,
                "image": None,
                "user": user,
                "assistant": json.dumps(nxt, sort_keys=True),
            }
        )
    return out


def _anomaly_records(logs, entries, split: str, quota: int) -> list[dict]:
    templates = _split_templates(ANOMALY_TEMPLATES, split)
    candidates = []
    for episode_id, log in logs.items():
        if entries[episode_id]["split"] != split:
            continue
        conds = {
            f["tick"]: c
            for f, c in zip(
                log["frames"], attribute_frames(log["frames"], GRAPHS[entries[episode_id]["graph"]])
            )
        }
        for frame in log["frames"]:
            if not frame.get("event"):
                continue
            for event in frame["event"].split(","):
                anomaly, action = EVENT_DIAGNOSIS[event]
                obj = conds[frame["tick"]].object or "target object"
                explanations = {
                    "object_moved": f"an external impulse moved the {obj}; re-perceive it.",
                    "grasp_lost": f"the grip slipped on the {obj}; re-grasp it.",
                    "drawer_jammed": "the drawer rails stiffened mid-cycle; retry the pull.",
                    "none": "control-noise injection with no visible fault; re-execute.",
                }
                diagnosis = {
                    "anomaly": anomaly,
                    "explanation": explanations[anomaly],
                    "suggested_action": action,
                }
                VlmDiagnosis(**diagnosis)
                stem = f"{episode_id}_f{int(frame['tick']):05d}"
                candidates.append(
                    (stem, event, log["instruction"], json.dumps(diagnosis, sort_keys=True))
                )
    if len(candidates) < quota:
        raise VqaError(f"anomaly {split}: {len(candidates)} candidates for quota {quota}")
    out = []
    for stem, event, instruction, assistant in candidates[:quota]:
        tid, form = templates[len(out) % len(templates)]
        user = f"<image>Frame from '{instruction}'. {form.format(event=event)}"
        out.append(
            {"template": tid, "image": f"frames/{stem}.jpg", "user": user, "assistant": assistant}
        )
    return out


def _spatial_records(rng: np.random.Generator, sources, split: str, quota: int) -> list[dict]:
    records = []
    for stem, _episode, boxes in sources:
        objects = sorted({cls for cls, *_ in boxes if not cls.startswith("drawer_")})
        if len(objects) < 2:
            continue
        by_name = {cls: (xc, yc) for cls, xc, yc, _w, _h in boxes}
        a = objects[0]
        others = [
            (n, abs(by_name[n][0] - by_name[a][0]) + abs(by_name[n][1] - by_name[a][1]))
            for n in objects[1:]
        ]
        beside, _ = min(others, key=lambda t: t[1])
        b = objects[1]
        left = "left" if by_name[a][0] < 0.5 else "right"
        vertical = "above" if by_name[a][1] < by_name[b][1] else "below"
        records.append(
            (
                stem,
                "r0",
                f"is the {a} on the left or right half of the table?",
                f"The {a} is on the {left} half.",
            )
        )
        records.append((stem, "r1", f"what is beside the {a}?", f"The {beside} is beside it."))
        records.append(
            (stem, "r2", f"Is the {a} above or below the {b}?", f"The {a} is {vertical} the {b}.")
        )
        records.append((stem, "r3", f"Which object is closest to the {a}?", f"The {beside}."))
    rng.shuffle(records)
    allowed = {tid for tid, _ in _split_templates(SPATIAL_TEMPLATES, split)}
    records = [r for r in records if r[1] in allowed]
    if len(records) < quota:
        raise VqaError(f"spatial {split}: {len(records)} candidates for quota {quota}")
    return [
        {"template": tid, "image": f"frames/{stem}.jpg", "user": f"<image>{q}", "assistant": a}
        for stem, tid, q, a in records[:quota]
    ]


def _generate_split(
    rng: np.random.Generator, split: str, quota_total: int, logs, entries, frames
) -> list[dict]:
    quotas = _quotas(quota_total)
    sources = _frame_sources(frames, entries, split)
    per_type = [
        _parse_records(rng, split, quotas[0]),
        _ground_records(rng, sources, split, quotas[1]),
        _state_records(rng, sources, logs, entries, split, quotas[2]),
        _next_records(logs, entries, split, quotas[3]),
        _anomaly_records(logs, entries, split, quotas[4]),
        _spatial_records(rng, sources, split, quotas[5]),
    ]
    records = []
    for rtype, items in zip(TYPES, per_type):
        for item in items:
            records.append({"type": rtype, **item})
    return records


def generate_vqa(
    demos: str | Path,
    labels_dir: str | Path,
    out: str | Path,
    train_n: int = TRAIN_N,
    val_n: int = VAL_N,
    seed: int = 0,
    copy_images: bool = True,
) -> dict:
    """Emit train/val VQA JSONL plus the referenced frame images."""
    demos_path, labels_path, out_path = Path(demos), Path(labels_dir), Path(out)
    _manifest, logs, entries = _load_demos(demos_path)
    frames = _read_boxes(labels_path / "labels")
    if not frames:
        raise VqaError(f"no label files under {labels_path / 'labels'}")
    rng = np.random.default_rng(seed)
    result = {}
    counter = 0
    for split, total in (("train", train_n), ("val", val_n)):
        records = _generate_split(rng, split, total, logs, entries, frames)
        rng.shuffle(records)
        lines = []
        for record in records:
            counter += 1
            messages = [
                {"role": "user", "content": record["user"]},
                {"role": "assistant", "content": record["assistant"]},
            ]
            lines.append(
                json.dumps(
                    {
                        "id": f"vqa_{split}_{counter:05d}",
                        "split": split,
                        "type": record["type"],
                        "template": record["template"],
                        "image": record["image"],
                        "messages": messages,
                    },
                    sort_keys=True,
                )
            )
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result[split] = len(records)
    if copy_images:
        images_src = labels_path / "images"
        frames_dir = out_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        stems: set[str] = set()
        for split in ("train", "val"):
            for line in (out_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                image = json.loads(line).get("image")
                if image is not None:
                    stems.add(Path(image).stem)
        for stem in sorted(stems):
            src = images_src / f"{stem}.jpg"
            if not src.is_file():
                raise VqaError(f"VQA references missing frame image: {src}")
            (frames_dir / f"{stem}.jpg").write_bytes(src.read_bytes())
        result["images"] = len(stems)
    logger.info("wrote %s", {k: v for k, v in result.items()})
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Generate the VQA instruction corpus")
    parser.add_argument("--demos", default="demos")
    parser.add_argument("--labels", default="datasets/yolo")
    parser.add_argument("--out", default="datasets/vqa")
    parser.add_argument("--train-n", type=int, default=TRAIN_N)
    parser.add_argument("--val-n", type=int, default=VAL_N)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-images", dest="copy_images", action="store_true", default=True)
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = generate_vqa(
        args.demos, args.labels, args.out, args.train_n, args.val_n, args.seed, args.copy_images
    )
    print(f"wrote train={result.get('train', 0)} val={result.get('val', 0)} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
