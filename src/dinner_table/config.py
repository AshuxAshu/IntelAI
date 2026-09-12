"""Global configuration constants and the shared exception base."""

from __future__ import annotations

ARTIFACT_DATASET = "dinner-table/train"
ARTIFACT_YOLO = "dinner-table/yolo"
ARTIFACT_VLM_LORA = "dinner-table/vlm-lora"
ARTIFACT_ACT_EXPORT = "dinner-table/act"
ARTIFACT_BENCH = "dinner-table/bench"


class DinnerTableError(Exception):
    """Base class for all project errors."""
