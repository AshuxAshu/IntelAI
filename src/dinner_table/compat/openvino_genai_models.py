"""GenAI model-directory fixups for upstream export-format gaps (rule R3).

The pinned openvino-genai VLMPipeline requires tokenizer/detokenizer IRs in
the model directory; some public OpenVINO exports on the Hub ship the weights
but omit them. ensure_genai_tokenizers generates the missing IRs in place
from the directory's own HF tokenizer (idempotent)."""

from __future__ import annotations

import logging
from pathlib import Path

import openvino as ov
import openvino_tokenizers as ovt

logger = logging.getLogger(__name__)


def ensure_genai_tokenizers(model_dir: Path) -> Path:
    """Make a converted model directory VLMPipeline-loadable; returns the dir.

    Writes openvino_tokenizer.xml / openvino_detokenizer.xml when missing.
    """
    model_dir = Path(model_dir)
    tokenizer_xml = model_dir / "openvino_tokenizer.xml"
    if tokenizer_xml.is_file() and (model_dir / "openvino_detokenizer.xml").is_file():
        return model_dir
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer_model, detokenizer_model = ovt.convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    ov.save_model(tokenizer_model, str(tokenizer_xml))
    ov.save_model(detokenizer_model, str(model_dir / "openvino_detokenizer.xml"))
    logger.info("generated tokenizer IRs under %s", model_dir)
    return model_dir
