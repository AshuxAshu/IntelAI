"""A15: transformers VLM engine protocol, async, fallback, latency."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from dinner_table.reasoning.fallback_parser import template_parse
from dinner_table.reasoning.interfaces import VlmEngine
from dinner_table.reasoning.schema import ObjectStatus, SceneSummary
from dinner_table.reasoning.vlm_runtime import TransformersVlmEngine

ADAPTER = Path("artifacts/vlm_lora")
CORPUS = Path(__file__).resolve().parents[2] / "configs" / "vlm" / "instruction_corpus.yaml"


def _summary(instruction: str = "place the plate") -> SceneSummary:
    return SceneSummary(
        instruction=instruction,
        objects={"plate": ObjectStatus(visible=True, where="on table")},
        drawer_open=False,
        completed_step_ids=[],
    )


def _engine(generate_fn, **kwargs) -> TransformersVlmEngine:
    kwargs.setdefault("image_provider", lambda: None)
    return TransformersVlmEngine(
        _load_fn=lambda model_id, adapter: (object(), object()),
        _generate_fn=generate_fn,
        **kwargs,
    )


@pytest.mark.nightly
def test_vlm_parse():
    if not ADAPTER.is_dir():
        pytest.skip("needs the GPU-trained LoRA adapter")
    engine = TransformersVlmEngine(adapter_path=ADAPTER, image_provider=lambda: None)
    try:
        templates = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))["templates"]
        val = [e for t in templates if t["split"] == "val" for e in t["entries"]]
        from dinner_table.reasoning.schema import Step

        matched, refused, neg_total, json_ok = 0, 0, 0, 0
        for entry in val:
            graph = engine.parse_task(entry["text"], _summary(entry["text"]))
            json_ok += engine.last_metadata["model_json_valid"]
            if "refuse" in entry:
                neg_total += 1
                refused += graph is None
                continue
            expected = [Step(**s).model_dump() for s in entry["graph"]["steps"]]
            if graph is not None and graph.model_dump()["steps"] == expected:
                matched += 1
        positives = len(val) - neg_total
        assert matched / positives >= 0.90
        assert json_ok / len(val) >= 1.0
        assert refused / neg_total >= 0.90
    finally:
        engine.close()


@pytest.mark.nightly
def test_vlm_qa():
    from dinner_table.reasoning.sft_train import _real_generate_fn, evaluate_sft

    val_jsonl = Path("datasets/vqa/val.jsonl")
    if not ADAPTER.is_dir() or not val_jsonl.is_file():
        pytest.skip("needs the adapter and datasets/vqa/val.jsonl")
    engine = TransformersVlmEngine(adapter_path=ADAPTER, image_provider=lambda: None)
    try:
        metrics = evaluate_sft(
            val_jsonl,
            "datasets/vqa",
            _real_generate_fn(engine._model, engine._processor),
            max_samples=120,
        )
    finally:
        engine.close()
    assert metrics["state_qa"] >= 0.90
    assert metrics["next_em"] >= 0.90
    assert metrics["anomaly_acc"] >= 0.85


def test_vlm_engine_protocol():
    engine = _engine(lambda conv, image, schema, tokens: "{}")
    try:
        assert isinstance(engine, VlmEngine)
    finally:
        engine.close()


@pytest.mark.slow
def test_vlm_latency():
    if ADAPTER.is_dir():
        engine = TransformersVlmEngine(adapter_path=ADAPTER, image_provider=lambda: None)
    else:
        report = '{"ok": true, "reason": "clear"}'
        engine = _engine(lambda conv, image, schema, tokens: report)
    try:
        graph = template_parse("place the plate")
        summary = _summary()
        durations = []
        for _ in range(5):
            start = time.monotonic()
            result = engine.check_preconditions(graph, 1, summary)
            durations.append(time.monotonic() - start)
            assert result.ok is True or result.reason
        durations.sort()
        assert durations[2] < 3.0
    finally:
        engine.close()


def test_async_semantics():
    calls = []

    def slow_generate(conv, image, schema, tokens):
        time.sleep(0.3)
        calls.append(1)
        user = conv[-1]["content"]
        text = user[0]["text"] if isinstance(user, list) else user
        instruction = text.split("Task instruction: ")[1].split("\n")[0]
        return template_parse(instruction).model_dump_json()

    engine = _engine(slow_generate)
    try:
        inputs = [
            "place the plate",
            "open the drawer",
            "pass the bottle to the left arm",
            "put a fork beside the plate",
            "pick up the mug",
        ]
        summary = _summary()
        expected = [engine.parse_task(text, summary) for text in inputs]
        start = time.monotonic()
        engine.submit_parse(inputs[0], summary)
        assert time.monotonic() - start < 0.3
        deadline = time.monotonic() + 10.0
        while not engine.ready and time.monotonic() < deadline:
            time.sleep(0.05)
        assert engine.ready
        assert engine.result() is not None
        assert engine.result().model_dump() == expected[0].model_dump()
        assert engine.last_error() is None
        assert engine.last_metadata["attempts"] == 1
        assert not engine.last_metadata["fallback_used"]
    finally:
        engine.close()


def test_vlm_fallback_ladder():
    engine = _engine(lambda conv, image, schema, tokens: "not json {{{")
    try:
        graph = engine.parse_task("place the plate", _summary())
        assert graph is not None
        assert [s.skill for s in graph.steps] == ["pick", "place"]
        assert engine.last_metadata["fallback_used"]
        assert engine.last_metadata["attempts"] == 2
        assert engine.last_error() is None
        assert engine.parse_task("dance with the plate", _summary()) is None
        assert engine.last_error().startswith("refused:")
    finally:
        engine.close()


def test_vlm_boundary_caps():
    valid = '{"ok": true, "reason": "clear"}'

    def hanging(conv, image, schema, tokens):
        time.sleep(5.0)
        return valid

    engine = _engine(hanging, budget_s=0.2)
    try:
        graph = template_parse("place the plate")
        summary = _summary()
        timed = engine.check_preconditions(graph, 1, summary)
        assert timed.ok is False and "timed out" in timed.reason
        unknown = engine.check_preconditions(graph, 99, summary)
        assert unknown.ok is False and "unknown step" in unknown.reason
        diag = engine.diagnose(graph, 99, summary)
        assert diag.anomaly == "unknown"
    finally:
        engine.close()


def test_vlm_refusal_json_goes_to_fallback():
    seen = []

    def refuse_then_parse(conv, image, schema, tokens):
        seen.append(1)
        if len(seen) == 1:
            return '{"refuse": true, "reason": "no such object"}'
        return "still not a graph {{{"

    engine = _engine(refuse_then_parse)
    try:
        assert engine.parse_task("pick up the glass", _summary()) is None
        assert engine.last_error().startswith("refused:")
        assert engine.last_metadata["fallback_used"]
    finally:
        engine.close()
