"""Qwen3-VL LoRA SFT on the A11 VQA corpus (A15).

Fine-tunes ``Qwen/Qwen3-VL-2B-Instruct`` (vision tower frozen, LoRA on the
language tower) and evaluates parse/state/next/anomaly accuracy plus
JSON-validity on val after each epoch. A zero-shot Qwen3-VL-4B run on the
same val set doubles as the G3 promotion evidence. Heavy dependencies are
imported lazily; training itself runs on the GPU box.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dinner_table.config import ARTIFACT_VLM_LORA, DinnerTableError

logger = logging.getLogger(__name__)

MODEL_ID_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_ID_4B = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_LR = 1e-4
DEFAULT_BATCH = 4
DEFAULT_EPOCHS = 3
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 1024
DEFAULT_EVAL_SAMPLES = 60
DEFAULT_EVAL_TOKENS = 256
IMAGE_TOKEN = "<image>"
JSON_TYPES = ("p", "n", "x", "a")  # template-id prefixes with JSON answers


def parse_union_schema() -> dict:
    """JSON schema for parse answers: a TaskGraph or a refusal object."""
    from dinner_table.reasoning.schema import TaskGraph

    return {
        "anyOf": [
            TaskGraph.model_json_schema(),
            {
                "type": "object",
                "properties": {"refuse": {"const": True}, "reason": {"type": "string"}},
                "required": ["refuse", "reason"],
            },
        ]
    }


class SftError(DinnerTableError):
    """Raised when the SFT dataset or training setup is invalid."""


def build_sft_dataset(jsonl: str | Path, images_dir: str | Path, out: str | Path | None = None):
    """Build an HF dataset with ``image`` + ``messages`` columns from VQA JSONL.

    Text-only records (parse/state questions without a frame) carry a null
    image; `format_example` substitutes a blank frame so batches stay
    rectangular. Every referenced image file must exist.
    """
    import datasets

    jsonl_path, images_path = Path(jsonl), Path(images_dir)
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SftError(f"no VQA records in {jsonl_path}")
    missing = sorted(
        {r["image"] for r in records if r["image"] and not (images_path / r["image"]).is_file()}
    )
    if missing:
        shown = ", ".join(missing[:5])
        raise SftError(f"{len(missing)} images missing under {images_path}: {shown}")
    dataset = datasets.Dataset.from_dict(
        {
            "image": [str(images_path / r["image"]) if r["image"] else None for r in records],
            "messages": [
                [
                    {"role": "user", "content": r["user"]},
                    {"role": "assistant", "content": r["assistant"]},
                ]
                for r in records
            ],
        }
    ).cast_column("image", datasets.Image())
    if out is not None:
        dataset.save_to_disk(str(out))
    logger.info("built SFT dataset: %d examples from %s", len(dataset), jsonl_path)
    return dataset


def to_conversation(messages: list) -> list:
    """Convert role/content messages to content-block chat turns."""
    conversation = []
    for turn in messages:
        text = turn["content"]
        if turn["role"] == "user" and text.startswith(IMAGE_TOKEN):
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text[len(IMAGE_TOKEN) :]},
                    ],
                }
            )
        else:
            conversation.append({"role": turn["role"], "content": [{"type": "text", "text": text}]})
    return conversation


def format_example(example: dict, processor, max_length: int = DEFAULT_MAX_LENGTH) -> dict:
    """Tokenize one example; mask the user prompt with -100 in labels."""
    from PIL import Image

    conversation = to_conversation(example["messages"])
    image = example["image"] if example["image"] is not None else Image.new("RGB", (16, 16))
    full_text = processor.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    prompt_text = processor.tokenizer.apply_chat_template(
        conversation[:-1], tokenize=False, add_generation_prompt=True
    )
    full = processor(text=full_text, images=image, truncation=True, max_length=max_length)
    prompt = processor(text=prompt_text, images=image)
    input_ids = list(full["input_ids"])
    cut = min(len(prompt["input_ids"]), len(input_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * cut + input_ids[cut:],
        "pixel_values": full["pixel_values"],
    }


def sft_lora_config(
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
):
    """PEFT LoRA config for the Qwen language tower (vision tower frozen)."""
    from peft import LoraConfig, TaskType

    if rank <= 0:
        raise SftError(f"rank must be positive, got {rank}")
    if any("visual" in m or "vision" in m for m in target_modules):
        raise SftError("vision tower stays frozen; target language modules only")
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
    )


def prepare_tokenizer(tokenizer):
    """Left-pad for batched generation-style fine-tuning; fills a missing pad token."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


class SftDataCollator:
    """Pad text left, pad labels with -100, stack pixel values."""

    def __init__(self, pad_token_id: int, padding_side: str = "left", label_pad: int = -100):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.label_pad = label_pad

    def __call__(self, features: list) -> dict:
        import torch

        width = max(len(f["input_ids"]) for f in features)

        def pad(values, fill):
            short = width - len(values)
            if self.padding_side == "left":
                return [fill] * short + list(values)
            return list(values) + [fill] * short

        return {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features]),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features]),
            "labels": torch.tensor([pad(f["labels"], self.label_pad) for f in features]),
            "pixel_values": torch.stack(
                [torch.as_tensor(f["pixel_values"], dtype=torch.float32) for f in features]
            ),
        }


def sft_training_args(
    out: str | Path,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
    dtype: str = "bfloat16",
):
    """HF training arguments; fp32 + CPU whenever CUDA is absent."""
    import torch
    from transformers import TrainingArguments

    cuda = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=str(out),
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=cuda and dtype == "bfloat16",
        fp16=cuda and dtype == "float16",
        use_cpu=not cuda,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        logging_steps=10,
        save_total_limit=1,
    )


def _load_backbone(model_id: str, rank: int, alpha: int, dtype: str):
    """Load processor + LoRA-wrapped model. Separated for test injection."""
    import torch
    from peft import get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.float32)
    processor = AutoProcessor.from_pretrained(model_id)
    prepare_tokenizer(processor.tokenizer)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.config.use_cache = False
    return processor, get_peft_model(model, sft_lora_config(rank, alpha))


def _score_json(pred: str, gold: str) -> tuple[bool, bool]:
    """Return (valid_json, exact_match) for one JSON-expected answer."""
    try:
        parsed = json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        return False, False
    try:
        return True, parsed == json.loads(gold)
    except (json.JSONDecodeError, TypeError):
        return True, False


def evaluate_sft(
    val_jsonl: str | Path,
    images_dir: str | Path,
    generate_fn,
    max_samples: int | None = None,
    max_new_tokens: int = DEFAULT_EVAL_TOKENS,
) -> dict:
    """Score the five A15 metrics over val records in file order.

    `generate_fn(user_text, image_or_None, kind, max_new_tokens) -> str`
    produces one completion per record (`kind` = template-id prefix).
    Exact-match slices: parse (p/n, JSON), state-QA (s, text), next-step
    (x, JSON), anomaly (a, JSON); JSON-validity covers the JSON-expected
    records only. Measures the raw model (no fallback).
    """
    from PIL import Image

    val_path, images_path = Path(val_jsonl), Path(images_dir)
    lines = [line for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = {"parse": [0, 0], "state": [0, 0], "next": [0, 0], "anomaly": [0, 0]}
    valid, json_total, scored = 0, 0, 0
    for line in lines[:max_samples]:
        record = json.loads(line)
        kind = record["template"][0]
        image = None
        if record["image"]:
            image = Image.open(images_path / record["image"]).convert("RGB")
        pred = generate_fn(record["user"], image, kind, max_new_tokens)
        scored += 1
        if kind in ("p", "n"):
            ok_json, same = _score_json(pred, record["assistant"])
            valid += ok_json
            json_total += 1
            hits["parse"][0] += same
            hits["parse"][1] += 1
        elif kind == "s":
            hits["state"][0] += pred.strip() == record["assistant"].strip()
            hits["state"][1] += 1
        elif kind == "x":
            ok_json, same = _score_json(pred, record["assistant"])
            valid += ok_json
            json_total += 1
            hits["next"][0] += same
            hits["next"][1] += 1
        elif kind == "a":
            ok_json, same = _score_json(pred, record["assistant"])
            valid += ok_json
            json_total += 1
            hits["anomaly"][0] += same
            hits["anomaly"][1] += 1
    metrics = {
        "parse_em": hits["parse"][0] / hits["parse"][1] if hits["parse"][1] else 0.0,
        "state_qa": hits["state"][0] / hits["state"][1] if hits["state"][1] else 0.0,
        "next_em": hits["next"][0] / hits["next"][1] if hits["next"][1] else 0.0,
        "anomaly_acc": hits["anomaly"][0] / hits["anomaly"][1] if hits["anomaly"][1] else 0.0,
        "json_valid": valid / json_total if json_total else 0.0,
        "n": scored,
    }
    logger.info("sft eval: %s", json.dumps(metrics, sort_keys=True))
    return metrics


def _real_generate_fn(model, processor):
    from dinner_table.compat.constrained import generate_json, generate_text

    def generate(user_text: str, image, kind: str, max_new_tokens: int) -> str:
        conversation = to_conversation([{"role": "user", "content": user_text}])
        if image is not None:
            conversation[0]["content"] = [
                {"type": "image"},
                *[
                    b
                    for b in conversation[0]["content"]
                    if not (isinstance(b, dict) and b.get("type") == "image")
                ],
            ]
        if kind in ("p", "n"):
            schema = parse_union_schema()
            return generate_json(model, processor, conversation, image, schema, max_new_tokens)
        if kind in ("x", "a"):
            return generate_json(model, processor, conversation, image, None, max_new_tokens)
        return generate_text(model, processor, conversation, image, max_new_tokens)

    return generate


def _run_train(train_dataset, val_dataset, collator, args, epoch_eval) -> dict:
    """Run the HF Trainer loop with per-epoch SFT eval. Separated for tests."""
    from transformers import Trainer, TrainerCallback

    class _EpochEval(TrainerCallback):
        def on_epoch_end(self, trainer_args, state, control, model=None, **kwargs):
            epoch_eval(model)
            return control

    trainer = Trainer(
        model=epoch_eval.model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[_EpochEval()],
    )
    result = trainer.train()
    return {"train_loss": result.training_loss, **trainer.evaluate()}


class _EpochEvalState:
    """Carries the model reference plus the val-eval binding for the callback."""

    def __init__(self, model, processor, val_jsonl, images_dir, eval_samples, store: dict):
        self.model = model
        self._processor = processor
        self._val_jsonl = val_jsonl
        self._images_dir = images_dir
        self._eval_samples = eval_samples
        self._store = store

    def __call__(self, model) -> dict:
        from dinner_table.compat.constrained import backend_name

        metrics = evaluate_sft(
            self._val_jsonl,
            self._images_dir,
            _real_generate_fn(model, self._processor),
            max_samples=self._eval_samples,
        )
        metrics["backend"] = backend_name()
        self._store[f"epoch_{len(self._store) + 1}"] = metrics
        return metrics


def train_sft(
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    images_dir: str | Path,
    out: str | Path = "artifacts/vlm_lora",
    model_id: str = MODEL_ID_2B,
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
    dtype: str = "bfloat16",
    max_length: int = DEFAULT_MAX_LENGTH,
    eval_samples: int = DEFAULT_EVAL_SAMPLES,
    compare_4b: bool = True,
    push: bool = False,
) -> dict:
    """Fine-tune the adapter, evaluate per epoch, and record the comparison."""
    from transformers import set_seed

    set_seed(seed)
    processor, model = _load_backbone(model_id, rank, alpha, dtype)
    train_dataset = build_sft_dataset(train_jsonl, images_dir).map(
        lambda ex: format_example(ex, processor, max_length),
        remove_columns=["image", "messages"],
    )
    val_dataset = build_sft_dataset(val_jsonl, images_dir).map(
        lambda ex: format_example(ex, processor, max_length),
        remove_columns=["image", "messages"],
    )
    collator = SftDataCollator(processor.tokenizer.pad_token_id)
    per_epoch: dict = {}
    epoch_eval = _EpochEvalState(model, processor, val_jsonl, images_dir, eval_samples, per_epoch)
    metrics = _run_train(
        train_dataset,
        val_dataset,
        collator,
        sft_training_args(out, lr, batch, epochs, seed, dtype),
        epoch_eval,
    )
    out_path = Path(out)
    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))
    report = {"train": metrics, "per_epoch": per_epoch}
    if compare_4b:
        report["qwen4b_zero_shot"] = compare_4b_run(
            val_jsonl, images_dir, eval_samples=eval_samples
        )
    (out_path / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if push:
        from huggingface_hub import HfApi

        HfApi().upload_folder(repo_id=ARTIFACT_VLM_LORA, folder_path=str(out_path))
    logger.info("sft done: %s", json.dumps(report, sort_keys=True)[:500])
    return report


def compare_4b_run(
    val_jsonl: str | Path,
    images_dir: str | Path,
    model_id: str = MODEL_ID_4B,
    eval_samples: int = DEFAULT_EVAL_SAMPLES,
    _load_fn=None,
) -> dict:
    """Zero-shot Qwen3-VL-4B on the same val set (G3 comparison evidence)."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if _load_fn is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        model.eval()
    else:
        model, processor = _load_fn(model_id)
    return evaluate_sft(
        val_jsonl, images_dir, _real_generate_fn(model, processor), max_samples=eval_samples
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; runs on the GPU box. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="LoRA SFT Qwen3-VL-2B on the VQA corpus")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--images", required=True, help="VQA output directory")
    parser.add_argument("--out", default="artifacts/vlm_lora")
    parser.add_argument("--model-id", default=MODEL_ID_2B)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--alpha", type=int, default=DEFAULT_ALPHA)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--push", action="store_true", help=f"upload to {ARTIFACT_VLM_LORA}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = train_sft(
        args.train_jsonl,
        args.val_jsonl,
        args.images,
        out=args.out,
        model_id=args.model_id,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        batch=args.batch,
        epochs=args.epochs,
        seed=args.seed,
        eval_samples=args.eval_samples,
        compare_4b=not args.no_compare,
        push=args.push,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
