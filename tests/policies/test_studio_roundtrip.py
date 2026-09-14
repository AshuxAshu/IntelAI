"""Studio roundtrip tests: public-dataset train, export rungs, InferenceModel parity."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from openvino import Core as OvCore
from physicalai.inference import InferenceModel
from physicalai.inference.component_factory import component_registry
from physicalai.policies import ACT

from dinner_table.policies.export_quantize import (
    RUNG_FP16,
    RUNG_FP32,
    RUNG_INT8_PTQ,
    RUNG_INT8_WEIGHTS,
    compress_int8_weights,
    export_fp32_reference,
    export_openvino,
    ptq_int8,
    random_inputs,
)
from dinner_table.policies.studio_train import run as train_run

PUSHT_ACT_CONFIG = """\
model:
  class_path: physicalai.policies.ACT
  init_args:
    chunk_size: 25
    n_action_steps: 25
    image_size: [224, 224]
    use_vae: true
    optimizer_lr: 2.5e-4
data:
  class_path: physicalai.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: lerobot/pusht
    data_format: physicalai
    train_batch_size: 8
    episodes: [0, 1, 2]
    video_backend: pyav
trainer:
  accelerator: cpu
  devices: 1
  max_steps: 50
"""

PARITY_INPUTS = 100
CALIBRATION_INPUTS = 32
FP32_REL_ERROR_LIMIT = 1e-3
FP16_REL_ERROR_LIMIT = 5e-3
INT8_NORMALIZED_MSE_LIMIT = 5e-3


def _train_stack_available() -> bool:
    try:
        importlib.import_module("physicalai.train")
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(not _train_stack_available(), reason="physicalai-train not importable"),
]


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    """50-step public-dataset training run with a checkpoint (shared by the rungs)."""
    root = tmp_path_factory.mktemp("roundtrip")
    config = root / "pusht_act.yaml"
    config.write_text(PUSHT_ACT_CONFIG, encoding="utf-8")
    meta_path = train_run(
        config,
        overrides=["--trainer.default_root_dir", str(root / "experiments")],
        smoke=True,
        runs_dir=root / "runs",
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checkpoints = sorted(Path(meta["experiment_dir"]).glob("checkpoints/*.ckpt"))
    assert checkpoints, "the training run produced no checkpoint"
    return checkpoints[0], root


@pytest.fixture(scope="module")
def exported(trained_checkpoint):
    """The FP32 reference and FP16 rungs of the shared checkpoint."""
    ckpt, root = trained_checkpoint
    fp32_dir = root / RUNG_FP32
    fp16_dir = root / RUNG_FP16
    export_fp32_reference(ckpt, fp32_dir)
    export_openvino(ckpt, fp16_dir)
    return {"ckpt": ckpt, "root": root, RUNG_FP32: fp32_dir, RUNG_FP16: fp16_dir}


def _torch_action_chunks(ckpt: Path, inputs: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
    """Checkpoint-model actions for the fixed inputs; (chunk, action_dim) each."""
    policy = ACT.load_from_checkpoint(str(ckpt))
    policy.eval()
    chunks = []
    with torch.no_grad():
        for sample in inputs:
            batch = {name: torch.from_numpy(array) for name, array in sample.items()}
            out = policy.model(batch)
            actions = out[0] if isinstance(out, tuple) else out
            chunks.append(actions.numpy()[0])
    return chunks


def _max_relative_error(model: InferenceModel, inputs, references) -> float:
    worst = 0.0
    for sample, reference in zip(inputs, references):
        chunk = model.predict_action_chunk(sample)
        error = np.linalg.norm(chunk - reference) / (np.linalg.norm(reference) + 1e-12)
        worst = max(worst, float(error))
    return worst


def _normalized_mse(model: InferenceModel, inputs, references) -> float:
    """Mean squared action error normalized by the reference magnitude."""
    squared = 0.0
    reference_scale = 0.0
    count = 0
    for sample, reference in zip(inputs, references):
        chunk = model.predict_action_chunk(sample)
        squared += float(((chunk - reference) ** 2).sum())
        reference_scale += float((reference**2).sum())
        count += reference.size
    return squared / (reference_scale + 1e-12)


@pytest.mark.timeout(1500)
class TestRungParity:
    def test_exports_match_the_torch_checkpoint(self, exported):
        inputs = random_inputs(exported[RUNG_FP32], PARITY_INPUTS, seed=0)
        references = _torch_action_chunks(exported["ckpt"], inputs)
        limits = {RUNG_FP32: FP32_REL_ERROR_LIMIT, RUNG_FP16: FP16_REL_ERROR_LIMIT}
        for rung in (RUNG_FP32, RUNG_FP16):
            rung_record = json.loads((exported[rung] / "rung.json").read_text(encoding="utf-8"))
            assert rung_record["rung"] == rung
            assert rung_record["files"] and rung_record["model_size_bytes"] > 0
            model = InferenceModel(exported[rung], device="CPU")
            assert _max_relative_error(model, inputs, references) < limits[rung]


@pytest.mark.timeout(600)
class TestManifestContract:
    def test_manifest_matches_compiled_inputs_and_registry(self, exported):
        export_dir = exported[RUNG_FP32]
        manifest_json = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
        features = manifest_json["model"]["input_features"]
        preprocessors = manifest_json["model"]["preprocessors"]
        assert features, "manifest declares no input features"
        assert preprocessors, "manifest declares no preprocessors"

        compiled = OvCore().read_model(
            str(export_dir / manifest_json["model"]["artifacts"]["openvino"])
        )
        ir_inputs = {
            i.get_any_name(): tuple(d.get_length() for d in i.partial_shape)
            for i in compiled.inputs
        }
        assert {f["init_args"]["name"] for f in features} == set(ir_inputs)

        resize_resolution = None
        for spec in preprocessors:
            resolved = component_registry.resolve(spec["type"])
            assert resolved.startswith("physicalai."), f"{spec['type']} does not resolve"
            component_registry.get_class(spec["type"])
            if spec["type"] == "resize":
                resize_resolution = tuple(spec["image_resolution"])
        assert resize_resolution is not None, "no resize preprocessor declared"

        for feature in features:
            name = feature["init_args"]["name"]
            shape = tuple(feature["init_args"]["shape"])
            if feature["init_args"]["ftype"] == "STATE":
                assert ir_inputs[name] == (1, *shape)
            else:
                channels = shape[0]
                assert ir_inputs[name] == (1, channels, *resize_resolution)

        model = InferenceModel(export_dir, device="CPU")
        loaded_types = {type(p) for p in model.preprocessors}
        for spec in preprocessors:
            assert component_registry.get_class(spec["type"]) in loaded_types


@pytest.mark.timeout(900)
class TestQuantizationRungs:
    def _rung_export(self, exported, name):
        base = exported["root"] / f"test_{name}"
        shutil.rmtree(base, ignore_errors=True)
        shutil.copytree(exported[RUNG_FP32], base)
        return base

    def test_int8_weight_compression_parity(self, exported):
        base = self._rung_export(exported, RUNG_INT8_WEIGHTS)
        rung_path = compress_int8_weights(base)
        record = json.loads(rung_path.read_text(encoding="utf-8"))
        assert record["rung"] == RUNG_INT8_WEIGHTS
        assert record["mode"] in ("int8_sym", "int8_asym")
        assert record["model_size_bytes"] > 0
        reference = InferenceModel(exported[RUNG_FP32], device="CPU")
        compressed = InferenceModel(base, device="CPU")
        inputs = random_inputs(exported[RUNG_FP32], PARITY_INPUTS, seed=2)
        references = [reference.predict_action_chunk(sample) for sample in inputs]
        assert _normalized_mse(compressed, inputs, references) < INT8_NORMALIZED_MSE_LIMIT

    def test_int8_ptq_parity(self, exported):
        base = self._rung_export(exported, RUNG_INT8_PTQ)
        calibration = random_inputs(exported[RUNG_FP32], CALIBRATION_INPUTS, seed=1)
        rung_path = ptq_int8(base, calibration)
        record = json.loads(rung_path.read_text(encoding="utf-8"))
        assert record["rung"] == RUNG_INT8_PTQ
        assert record["calibration_samples"] == CALIBRATION_INPUTS
        reference = InferenceModel(exported[RUNG_FP32], device="CPU")
        quantized = InferenceModel(base, device="CPU")
        inputs = random_inputs(exported[RUNG_FP32], PARITY_INPUTS, seed=2)
        references = [reference.predict_action_chunk(sample) for sample in inputs]
        assert _normalized_mse(quantized, inputs, references) < INT8_NORMALIZED_MSE_LIMIT
