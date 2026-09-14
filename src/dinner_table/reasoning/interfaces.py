"""VLM engine protocol. Implemented twice, both real components:
- TransformersVlmEngine  (reasoning/vlm_runtime.py, Dev A) - HF transformers
  backend used for SFT evaluation, the Qwen-VL comparison, and hosts
  without openvino-genai.
- OvGenaiVlmEngine       (runtime/engines.py, Dev B) - openvino-genai INT8
  backend used in the deployed Intel pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import PreconditionReport, SceneSummary, TaskGraph, VlmDiagnosis


@runtime_checkable
class VlmEngine(Protocol):
    """All methods must be safe to call from the control thread. The submit/
    ready/result triple exists so the 1-3 s reasoning call never blocks a
    25 Hz tick (SkillExecutorSource polls `ready` each tick)."""

    def submit_parse(self, instruction: str, summary: SceneSummary) -> None:
        """Begin an async parse of instruction into a TaskGraph."""

    @property
    def ready(self) -> bool:
        """True when the most recent submit_* call has finished."""

    def result(self) -> TaskGraph | None:
        """The finished TaskGraph, or None if not ready or parsing failed."""

    def last_error(self) -> str | None:
        """Human-readable error of the last failed call, else None."""

    def check_preconditions(
        self, graph: TaskGraph, current_step_id: int, summary: SceneSummary
    ) -> PreconditionReport:
        """Blocking call at a skill boundary; budget < 3 s."""

    def diagnose(
        self, graph: TaskGraph, failed_step_id: int, summary: SceneSummary
    ) -> VlmDiagnosis:
        """Blocking call after a postcondition failure; budget < 3 s."""
