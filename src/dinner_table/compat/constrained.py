"""Constrained JSON generation backend for the transformers VLM path (A15).

Prefers `outlines` (grammar-constrained decoding) when installed; otherwise
falls back to greedy generation plus a JSON repair retry. The engine's own
retry-then-fallback ladder absorbs any backend failure, so an outlines
regression degrades to the fallback parser instead of crashing the tick.

NOTE: the outlines branch needs verification on the first GPU-box run
(outlines is not installed on this host); the repair loop is fully covered
headless through the injected-generator seams.
"""

from __future__ import annotations

import importlib.util
import json
import logging

logger = logging.getLogger(__name__)


def outlines_available() -> bool:
    """True when the outlines constrained-decoding stack is importable."""
    return importlib.util.find_spec("outlines") is not None


def backend_name() -> str:
    """The backend `generate_json` will use on this host."""
    return "outlines" if outlines_available() else "repair-loop"


def generate_text(model, processor, conversation: list, image, max_new_tokens: int) -> str:
    """One greedy generation from a content-block conversation plus image."""
    import torch

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
    if image is not None:
        kwargs["images"] = [image]
    inputs = processor(**kwargs)
    device = getattr(model, "device", torch.device("cpu"))
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1] :]
    return processor.decode(gen, skip_special_tokens=True).strip()


def generate_json(model, processor, conversation: list, image, schema, max_new_tokens: int) -> str:
    """Generate JSON, constrained to `schema` (a pydantic model or schema dict).

    With outlines installed the constraint is structural; otherwise the loop
    generates once, and on invalid JSON retries once with a repair note. The
    caller always validates the returned text against its own model.
    """
    if outlines_available() and schema is not None:
        return _outlines_json(model, processor, conversation, image, schema, max_new_tokens)
    text = generate_text(model, processor, conversation, image, max_new_tokens)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError as exc:
        error = str(exc)
        logger.info("json repair retry: %s", error)
    repaired = [
        *conversation,
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"That was not valid JSON ({error}). Respond with JSON only.",
                }
            ],
        },
    ]
    return generate_text(model, processor, repaired, image, max_new_tokens)


def _outlines_json(model, processor, conversation, image, schema, max_new_tokens: int) -> str:
    import outlines

    schema_dict = schema if isinstance(schema, dict) else schema.model_json_schema()
    ow_model = outlines.from_transformers(model, processor.tokenizer)
    generator = outlines.Generator(ow_model, outlines.json_schema(schema_dict))
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return generator(prompt, media=image, max_tokens=max_new_tokens)
