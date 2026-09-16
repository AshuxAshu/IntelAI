"""A15: SFT dataset, configs, eval scoring, and train wiring (headless)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image

from dinner_table.reasoning import sft_train
from dinner_table.reasoning.sft_train import (
    SftDataCollator,
    SftError,
    _EpochEvalState,
    build_sft_dataset,
    compare_4b_run,
    evaluate_sft,
    format_example,
    parse_union_schema,
    prepare_tokenizer,
    sft_lora_config,
    sft_training_args,
    to_conversation,
    train_sft,
)


def _write_vqa(tmp_path: Path) -> tuple[Path, Path]:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (16, 16), (10, 0, 0)).save(frames / "s0.jpg")
    records = [
        {
            "template": "a0",
            "image": "frames/s0.jpg",
            "user": "<image>Diagnose.",
            "assistant": '{"anomaly": "none"}',
        },
        {
            "template": "p00",
            "image": None,
            "user": "Instruction: 'set the table'. Respond with the TaskGraph JSON.",
            "assistant": '{"task_id": "x", "instruction": "y", "steps": []}',
        },
    ]
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return jsonl, tmp_path


def test_build_sft_dataset_columns_and_null_image(tmp_path):
    jsonl, images = _write_vqa(tmp_path)
    ds = build_sft_dataset(jsonl, images)
    assert set(ds.column_names) == {"image", "messages"}
    assert len(ds) == 2
    assert ds[0]["image"] is not None
    assert ds[1]["image"] is None


def test_build_sft_dataset_missing_image_raises(tmp_path):
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(
        json.dumps({"image": "frames/gone.jpg", "user": "u", "assistant": "a"}),
        encoding="utf-8",
    )
    with pytest.raises(SftError, match="missing"):
        build_sft_dataset(jsonl, tmp_path)


def test_to_conversation_blocks():
    conv = to_conversation(
        [
            {"role": "user", "content": "<image>What is wrong?"},
            {"role": "assistant", "content": '{"anomaly": "none"}'},
        ]
    )
    assert conv[0]["content"][0] == {"type": "image"}
    assert conv[1]["content"] == [{"type": "text", "text": '{"anomaly": "none"}'}]


class _FakeChatTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        return " ".join(b.get("text", "<img>") for turn in conversation for b in turn["content"])


class _FakeProcessor:
    tokenizer = _FakeChatTokenizer()
    seen_images: ClassVar[list] = []

    def __call__(self, text, images, **kwargs):
        type(self).seen_images.append(images)
        ids = [ord(c) % 97 + 3 for c in text[:24]]
        return {"input_ids": ids, "pixel_values": [[[0.0]]]}


def test_format_example_masks_prompt_and_blanks_missing_image():
    _FakeProcessor.seen_images = []
    full = format_example(
        {
            "messages": [
                {"role": "user", "content": "<image>Hi?"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "image": None,
        },
        _FakeProcessor(),
    )
    assert full["labels"][0] == -100
    assert sum(1 for label in full["labels"] if label == -100) == 9
    assert all(len(v) == len(full["input_ids"]) for v in (full["attention_mask"], full["labels"]))
    assert _FakeProcessor.seen_images and _FakeProcessor.seen_images[0] is not None


def test_sft_lora_config_rank_alpha():
    cfg = sft_lora_config()
    assert (cfg.r, cfg.lora_alpha) == (16, 32)
    assert set(cfg.target_modules) >= {"q_proj", "v_proj"}
    assert sft_lora_config(rank=8, alpha=16).r == 8
    with pytest.raises(SftError, match="rank"):
        sft_lora_config(rank=0)
    with pytest.raises(SftError, match="frozen"):
        sft_lora_config(target_modules=("q_proj", "visual_proj"))


class _FakeTokenizer:
    padding_side = "right"
    pad_token_id = None
    eos_token_id = 7


def test_prepare_tokenizer_left():
    tok = prepare_tokenizer(_FakeTokenizer())
    assert tok.padding_side == "left"
    assert tok.pad_token_id == 7


def test_collator_pads_left_and_stacks():
    batch = SftDataCollator(pad_token_id=0)(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
                "pixel_values": [[[0.5]]],
            },
            {
                "input_ids": [4, 5, 6, 7, 8],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [4, 5, 6, 7, 8],
                "pixel_values": [[[1.5]]],
            },
        ]
    )
    assert batch["input_ids"].tolist() == [[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 2, 3], [4, 5, 6, 7, 8]]
    assert batch["pixel_values"].shape == (2, 1, 1, 1)


def test_training_args_plumbing(tmp_path):
    args = sft_training_args(tmp_path, lr=2e-4, batch=2, epochs=1, seed=9)
    assert args.learning_rate == 2e-4
    assert args.per_device_train_batch_size == 2
    assert args.num_train_epochs == 1
    assert args.seed == 9
    assert args.remove_unused_columns is False


def test_parse_union_schema_shape():
    schema = parse_union_schema()
    assert len(schema["anyOf"]) == 2
    assert schema["anyOf"][1]["required"] == ["refuse", "reason"]


def _write_eval(tmp_path: Path) -> tuple[Path, Path]:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (8, 8)).save(frames / "f.jpg")
    graph = {"task_id": "t", "instruction": "i", "steps": []}
    records = [
        ("p00", None, "parse this", json.dumps(graph)),
        ("p01", None, "parse that", json.dumps(graph)),
        ("n00", None, "parse nope", json.dumps({"refuse": True, "reason": "nope"})),
        ("s0", "frames/f.jpg", "<image>drawer?", "The drawer is open."),
        ("s2", "frames/f.jpg", "<image>held?", "Arm A holds the mug."),
        ("x0", None, "next?", json.dumps({"skill": "pick"})),
        ("x1", None, "next2?", json.dumps({"skill": "place"})),
        ("a0", "frames/f.jpg", "<image>diag?", json.dumps({"anomaly": "none"})),
        ("g0", "frames/f.jpg", "<image>where?", "top left region."),
    ]
    jsonl = tmp_path / "val.jsonl"
    lines = [
        json.dumps({"template": t, "image": img, "user": u, "assistant": a})
        for t, img, u, a in records
    ]
    jsonl.write_text("\n".join(lines), encoding="utf-8")
    return jsonl, tmp_path


def test_evaluate_sft_exact_metrics(tmp_path):
    jsonl, images = _write_eval(tmp_path)
    canned = {
        "parse this": '{"steps": [], "instruction": "i", "task_id": "t"}',
        "parse that": '{"task_id": "wrong"}',
        "parse nope": '{"reason": "nope", "refuse": true}',
        "<image>drawer?": "The drawer is open.",
        "<image>held?": "Arm B holds the mug.",
        "next?": '{"skill": "pick"}',
        "next2?": "not json",
        "<image>diag?": '{"anomaly": "none"}',
        "<image>where?": "top left region.",
    }
    seen_kinds = []

    def generate(user_text, image, kind, tokens):
        seen_kinds.append(kind)
        return canned[user_text]

    metrics = evaluate_sft(jsonl, images, generate, max_samples=None)
    assert metrics["parse_em"] == pytest.approx(2 / 3)
    assert metrics["state_qa"] == pytest.approx(1 / 2)
    assert metrics["next_em"] == pytest.approx(1 / 2)
    assert metrics["anomaly_acc"] == pytest.approx(1.0)
    assert metrics["json_valid"] == pytest.approx(5 / 6)
    assert metrics["n"] == 9
    assert seen_kinds == ["p", "p", "n", "s", "s", "x", "x", "a", "g"]


class _FakeModel:
    def save_pretrained(self, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, "adapter_saved.txt").write_text("ok", encoding="utf-8")


class _FakeProcSaver:
    tokenizer = _FakeChatTokenizer()

    def save_pretrained(self, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, "processor_saved.txt").write_text("ok", encoding="utf-8")


def test_train_sft_wiring(tmp_path, monkeypatch):
    train_jsonl, images = _write_vqa(tmp_path)
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text(train_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    seen = {}

    def fake_load(model_id, rank, alpha, dtype):
        seen.update(model_id=model_id, rank=rank, alpha=alpha, dtype=dtype)
        return _FakeProcSaver(), _FakeModel()

    class FakeMapProcessor(_FakeProcessor):
        tokenizer = _FakeChatTokenizer()

        def save_pretrained(self, out):
            Path(out).mkdir(parents=True, exist_ok=True)

    def fake_load_map(model_id, rank, alpha, dtype):
        fake_load(model_id, rank, alpha, dtype)
        proc = FakeMapProcessor()
        proc.tokenizer = _FakeChatTokenizer()
        return proc, _FakeModel()

    def fake_run(train_ds, val_ds, collator, args, epoch_eval):
        assert len(train_ds) == 2 and len(val_ds) == 2
        assert args.learning_rate == 2e-4
        epoch_eval({"sentinel": "model"})
        return {"train_loss": 0.25}

    monkeypatch.setattr(sft_train, "_load_backbone", fake_load_map)
    monkeypatch.setattr(sft_train, "_run_train", fake_run)
    monkeypatch.setattr(sft_train, "compare_4b_run", lambda *a, **k: {"parse_em": 0.1})
    monkeypatch.setattr(
        sft_train, "_real_generate_fn", lambda model, proc: lambda u, i, k, t: '{"a": 1}'
    )
    out = tmp_path / "adapter"
    report = train_sft(
        train_jsonl, val_jsonl, images, out, rank=8, alpha=16, lr=2e-4, epochs=1, dtype="float32"
    )
    assert seen == {
        "model_id": sft_train.MODEL_ID_2B,
        "rank": 8,
        "alpha": 16,
        "dtype": "float32",
    }
    assert report["train"] == {"train_loss": 0.25}
    assert report["qwen4b_zero_shot"] == {"parse_em": 0.1}
    assert "epoch_1" in report["per_epoch"]
    assert (out / "adapter_saved.txt").is_file()
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["train"] == {
        "train_loss": 0.25
    }


def test_epoch_eval_state_binds_val(tmp_path, monkeypatch):
    jsonl, images = _write_eval(tmp_path)
    store: dict = {}
    monkeypatch.setattr(
        sft_train,
        "_real_generate_fn",
        lambda model, proc: lambda u, i, k, t: '{"a": 1}',
    )
    state = _EpochEvalState(object(), object(), jsonl, images, 4, store)
    metrics = state(object())
    assert metrics["n"] == 4
    assert store["epoch_1"]["n"] == 4
    assert "backend" in metrics


def test_compare_4b_run_uses_loader(tmp_path, monkeypatch):
    jsonl, images = _write_eval(tmp_path)
    monkeypatch.setattr(
        sft_train,
        "_real_generate_fn",
        lambda model, proc: lambda u, i, k, t: '{"a": 1}',
    )
    seen = {}

    def fake_4b(model_id):
        seen["id"] = model_id
        return object(), object()

    metrics = compare_4b_run(jsonl, images, eval_samples=2, _load_fn=fake_4b)
    assert metrics["n"] == 2
    assert seen["id"] == sft_train.MODEL_ID_4B
