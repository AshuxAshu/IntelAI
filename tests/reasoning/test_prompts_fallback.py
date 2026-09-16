"""A14: corpus round-trip, grammar edge cases, prompt prefix stability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dinner_table.reasoning.fallback_parser import (
    SET_TABLE_STEPS,
    FallbackCancel,
    FallbackError,
    FallbackRefusal,
    template_parse,
)
from dinner_table.reasoning.prompts import (
    SCHEMA_EXAMPLE_JSON,
    SYSTEM_PREFIX,
    SYSTEM_PROMPT,
    build_diagnosis_prompt,
    build_parse_prompt,
    build_precondition_prompt,
)
from dinner_table.reasoning.schema import ObjectStatus, SceneSummary, Step

CORPUS = Path(__file__).resolve().parents[2] / "configs" / "vlm" / "instruction_corpus.yaml"


def _load_corpus() -> list:
    return yaml.safe_load(CORPUS.read_text(encoding="utf-8"))["templates"]


def _summary() -> SceneSummary:
    return SceneSummary(
        instruction="place the plate",
        objects={"plate": ObjectStatus(visible=True, where="on table")},
        drawer_open=False,
        completed_step_ids=[],
    )


def test_corpus_round_trip():
    templates = _load_corpus()
    assert len(templates) >= 10
    matched, total, mismatches = 0, 0, []
    negatives, relatives, transcripts = 0, 0, 0
    for template in templates:
        assert template["split"] in ("train", "val")
        if template["id"] == "transcript":
            transcripts = len(template["entries"])
        for entry in template["entries"]:
            total += 1
            text = entry["text"]
            if "refuse" in entry:
                negatives += 1
                try:
                    result = template_parse(text)
                except FallbackRefusal as exc:
                    if str(exc) == entry["refuse"]:
                        matched += 1
                    else:
                        mismatches.append(f"{entry['id']}: wrong message {exc!s}")
                else:
                    mismatches.append(f"{entry['id']}: parsed instead of refusing: {result}")
                continue
            expected_steps = [Step(**s).model_dump() for s in entry["graph"]["steps"]]
            if any(isinstance(s.get("target"), dict) for s in entry["graph"]["steps"]):
                relatives += 1
            try:
                graph = template_parse(text)
            except FallbackError as exc:
                mismatches.append(f"{entry['id']}: refused instead of parsing: {exc!s}")
                continue
            assert graph is not None
            got = graph.model_dump()
            if (
                got["task_id"] == entry["graph"]["task_id"]
                and got["instruction"] == text
                and got["steps"] == expected_steps
            ):
                matched += 1
            else:
                mismatches.append(f"{entry['id']}: {json.dumps(got['steps'])[:200]}")
    assert negatives >= 12, f"only {negatives} negatives"
    assert relatives >= 8, f"only {relatives} spatial-relational entries"
    assert transcripts >= 6, f"only {transcripts} transcript entries"
    assert total >= 50, f"only {total} entries"
    rate = matched / total
    assert rate >= 0.90, f"round-trip {matched}/{total}:\n" + "\n".join(mismatches)


def test_grammar_edge_cases():
    graph = template_parse("pick up. The plate")
    assert [s.skill for s in graph.steps] == ["pick", "place"]
    with pytest.raises(FallbackRefusal, match="Negated"):
        template_parse("do not move the plate")
    chained = template_parse("pick up the mug then place it on the right spot")
    assert [s.object for s in chained.steps] == ["mug"] * 4
    with pytest.raises(FallbackRefusal, match="Separate"):
        template_parse("take the plate and put the mug")
    with pytest.raises(FallbackRefusal, match="another item"):
        template_parse("put the mug next to the mug")
    german = template_parse("put the tasse on the left spot")
    assert german.steps[0].object == "mug"
    long = " then ".join(["place the plate"] * 9)
    with pytest.raises(FallbackRefusal, match="eight"):
        template_parse(long)


def test_cancel_and_empty():
    with pytest.raises(FallbackCancel):
        template_parse("stop")
    with pytest.raises(FallbackCancel):
        template_parse("halt!")
    assert template_parse("") is None
    assert template_parse("   ") is None


def test_negatives_never_parse():
    for text in (
        "pick up the glass",
        "move the banana",
        "put the side plate on the left spot",
        "bring me the moon",
    ):
        with pytest.raises(FallbackRefusal):
            template_parse(text)


def _rendered(messages: list) -> str:
    return "\n".join(m["content"] for m in messages)


def test_prefix_stable():
    first = _rendered(build_parse_prompt("open the drawer", _summary()))
    second = _rendered(build_parse_prompt("pass the bottle to the left arm", _summary()))
    assert first[: len(SYSTEM_PREFIX)] == second[: len(SYSTEM_PREFIX)] == SYSTEM_PREFIX
    assert first.startswith(SYSTEM_PROMPT)
    assert len(SYSTEM_PREFIX) == len(SYSTEM_PROMPT + "\n\nSchema example:\n" + SCHEMA_EXAMPLE_JSON)


def test_precondition_and_diagnosis_share_prefix():
    graph = template_parse("place the plate")
    step = graph.steps[0]
    pre = _rendered(build_precondition_prompt(graph, step, _summary()))
    diag = _rendered(build_diagnosis_prompt(graph, step, _summary()))
    assert pre.startswith(SYSTEM_PREFIX) and diag.startswith(SYSTEM_PREFIX)
    assert "PreconditionReport" in pre and "VlmDiagnosis" in diag


def test_schema_example_matches_canonical():
    from dinner_table.teacher.task_graphs import CANONICAL_GRAPHS

    canonical = CANONICAL_GRAPHS["dinner_canonical"].model_dump_json()
    assert json.loads(SCHEMA_EXAMPLE_JSON) == json.loads(canonical)


def test_set_table_matches_full_graph():
    from dinner_table.teacher.task_graphs import CANONICAL_GRAPHS

    full = CANONICAL_GRAPHS["dinner_full"].steps
    assert [s.model_dump() for s in SET_TABLE_STEPS] == [s.model_dump() for s in full]
