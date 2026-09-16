"""Transformers-backend VLM engine (A15).

Implements the §10.3 `VlmEngine` protocol on HF transformers + the LoRA
adapter, mirroring `OvGenaiVlmEngine`'s structure (single-worker async seam,
repair retry, budgeted boundary calls) while using the A14 prompt builders.
When the model refuses or produces invalid JSON twice, `parse_task` falls
back to `fallback_parser.template_parse`; the fallback use is recorded in
`last_metadata` for the HUD.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import TYPE_CHECKING

from dinner_table.reasoning.fallback_parser import FallbackError, template_parse
from dinner_table.reasoning.prompts import (
    build_diagnosis_prompt,
    build_parse_prompt,
    build_precondition_prompt,
)
from dinner_table.reasoning.schema import (
    PreconditionReport,
    SceneSummary,
    TaskGraph,
    VlmDiagnosis,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
BUDGET_S = 3.0  # mirror of runtime/engines.py VLM_BUDGET_S (which needs openvino to import)
PARSE_MAX_NEW_TOKENS = 350
BOUNDARY_MAX_NEW_TOKENS = 48


def _with_image(messages: list, image) -> list:
    """Attach the frame to the user turn as an image content block."""
    if image is None:
        return messages
    converted = []
    for turn in messages:
        if turn["role"] == "user" and isinstance(turn["content"], str):
            converted.append(
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": turn["content"]}],
                }
            )
        else:
            converted.append(turn)
    return converted


def _default_load(model_id: str, adapter_path: str | Path | None):
    """Load processor + LoRA-wrapped Qwen3-VL. Separated for test injection."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, processor


class TransformersVlmEngine:
    """HF transformers VLM engine (§10.3 protocol, frozen signatures)."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        adapter_path: str | Path | None = None,
        *,
        image_provider: Callable[[], object] | None = None,
        budget_s: float = BUDGET_S,
        parse_max_new_tokens: int = PARSE_MAX_NEW_TOKENS,
        boundary_max_new_tokens: int = BOUNDARY_MAX_NEW_TOKENS,
        _load_fn=None,
        _generate_fn=None,
    ) -> None:
        load = _load_fn or _default_load
        self._model, self._processor = load(model_id, adapter_path)
        if _generate_fn is not None:
            self._generate = _generate_fn
        else:
            from dinner_table.compat.constrained import generate_json

            model, processor = self._model, self._processor
            self._generate = lambda conv, image, schema, tokens: generate_json(
                model, processor, conv, image, schema, tokens
            )
        self._image_provider = image_provider
        self._budget_s = budget_s
        self._parse_tokens = parse_max_new_tokens
        self._boundary_tokens = boundary_max_new_tokens
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._future = None
        self._result: TaskGraph | None = None
        self._error: str | None = None
        self._metadata: dict = {"fallback_used": False, "attempts": 0, "model_json_valid": False}

    def _frame(self):
        if self._image_provider is None:
            return None
        return self._image_provider()

    def submit_parse(self, instruction: str, summary: SceneSummary) -> None:
        """Begin an async parse of instruction into a TaskGraph."""
        self._error = None
        self._result = None
        self._metadata = {"fallback_used": False, "attempts": 0, "model_json_valid": False}
        self._future = self._executor.submit(self.parse_task, instruction, summary)

    @property
    def ready(self) -> bool:
        """True when the most recent submit_* call has finished."""
        return self._future is None or self._future.done()

    def result(self) -> TaskGraph | None:
        """The finished TaskGraph, or None if not ready or parsing failed."""
        if self._future is None or not self._future.done():
            return None
        if self._result is None and self._error is None:
            try:
                self._result = self._future.result()
            except Exception as exc:  # noqa: BLE001 - surfaced through last_error
                self._error = f"parse crashed: {exc}"
        return self._result

    def last_error(self) -> str | None:
        """Human-readable error of the last failed call, else None."""
        return self._error

    @property
    def last_metadata(self) -> dict:
        """Result metadata for the HUD (fallback use, attempts, JSON validity)."""
        return dict(self._metadata)

    def parse_task(self, instruction: str, summary: SceneSummary) -> TaskGraph | None:
        """Blocking parse with one repair retry, then the grammar fallback."""
        conversation = _with_image(build_parse_prompt(instruction, summary), self._frame())
        text = self._generate(conversation, self._frame(), TaskGraph, self._parse_tokens)
        self._metadata["attempts"] = 1
        graph = self._validate(text)
        if graph is not None:
            self._metadata["model_json_valid"] = True
            return graph
        repair = [
            *conversation,
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "That reply was not a valid TaskGraph. Every skill except "
                        "home and retract needs its object field filled. Reply with the "
                        "TaskGraph JSON only.",
                    }
                ],
            },
        ]
        text = self._generate(repair, self._frame(), TaskGraph, self._parse_tokens)
        self._metadata["attempts"] = 2
        graph = self._validate(text)
        if graph is not None:
            self._metadata["model_json_valid"] = True
            return graph
        self._metadata["fallback_used"] = True
        logger.info("parse fell back to the bounded grammar")
        try:
            graph = template_parse(instruction)
        except FallbackError as exc:
            self._error = f"refused: {exc}"
            return None
        if graph is None:
            self._error = "refused: empty instruction"
            return None
        return graph

    @staticmethod
    def _validate(text: str) -> TaskGraph | None:
        try:
            return TaskGraph.model_validate_json(text)
        except ValueError:
            return None

    def _bounded(self, func, *args):
        """Run one generation with the wall-clock cap; None on timeout."""
        worker = ThreadPoolExecutor(max_workers=1)
        try:
            future = worker.submit(func, *args)
            return future.result(timeout=self._budget_s)
        except FutureTimeout:
            return None
        finally:
            worker.shutdown(wait=False, cancel_futures=True)

    def check_preconditions(
        self, graph: TaskGraph, current_step_id: int, summary: SceneSummary
    ) -> PreconditionReport:
        """Blocking precondition check; generation capped at the budget."""
        step = next((s for s in graph.steps if s.id == current_step_id), None)
        if step is None:
            return PreconditionReport(ok=False, reason=f"unknown step id {current_step_id}")
        conversation = _with_image(build_precondition_prompt(graph, step, summary), self._frame())
        text = self._bounded(
            self._generate, conversation, self._frame(), PreconditionReport, self._boundary_tokens
        )
        if text is None:
            return PreconditionReport(ok=False, reason="vlm check timed out")
        try:
            return PreconditionReport.model_validate_json(text)
        except ValueError as exc:
            return PreconditionReport(ok=False, reason=f"vlm check failed: {exc}")

    def diagnose(
        self, graph: TaskGraph, failed_step_id: int, summary: SceneSummary
    ) -> VlmDiagnosis:
        """Blocking failure diagnosis; generation capped at the budget."""
        step = next((s for s in graph.steps if s.id == failed_step_id), None)
        if step is None:
            return VlmDiagnosis(
                anomaly="unknown",
                explanation=f"unknown step id {failed_step_id}",
                suggested_action="retry_skill",
            )
        conversation = _with_image(build_diagnosis_prompt(graph, step, summary), self._frame())
        text = self._bounded(
            self._generate, conversation, self._frame(), VlmDiagnosis, self._boundary_tokens
        )
        if text is None:
            return VlmDiagnosis(
                anomaly="unknown",
                explanation="vlm diagnosis timed out",
                suggested_action="retry_skill",
            )
        try:
            return VlmDiagnosis.model_validate_json(text)
        except ValueError as exc:
            return VlmDiagnosis(
                anomaly="unknown",
                explanation=f"vlm diagnosis failed: {exc}",
                suggested_action="retry_skill",
            )

    def close(self) -> None:
        """Shut down the async worker."""
        self._executor.shutdown(wait=True, cancel_futures=True)
