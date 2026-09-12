"""Studio training-harness tests: public-dataset validation and preflight gates."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from dinner_table.config import ARTIFACT_DATASET
from dinner_table.policies import studio_train
from dinner_table.policies.studio_train import (
    StudioTrainError,
    preflight,
    resolve_config,
    run,
)

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
  enable_checkpointing: false
"""


def _train_stack_available() -> bool:
    try:
        importlib.import_module("physicalai.train")
    except ImportError:
        return False
    return True


def _loss_step_points(metrics_csv: Path) -> list[tuple[int, float]]:
    points = []
    with metrics_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["train/loss_step"]:
                points.append((int(row["step"]), float(row["train/loss_step"])))
    points.sort()
    return points


@pytest.mark.nightly
@pytest.mark.skipif(not _train_stack_available(), reason="physicalai-train not importable")
def test_public_dataset_harness_validation(tmp_path):
    config = tmp_path / "pusht_act.yaml"
    config.write_text(PUSHT_ACT_CONFIG, encoding="utf-8")
    meta_path = run(
        config,
        overrides=["--trainer.default_root_dir", str(tmp_path / "experiments")],
        smoke=True,
        runs_dir=tmp_path / "runs",
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["smoke"] is True
    assert meta["run_id"]
    assert meta["config"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert meta["dataset"]["repo_id"] == "lerobot/pusht"
    assert meta["dataset"]["sha256"]
    assert meta["git_rev"]
    assert meta_path.parent == tmp_path / "runs" / meta["run_id"]
    experiment_dir = Path(meta["experiment_dir"])
    metrics_csv = experiment_dir / "metrics.csv"
    assert metrics_csv.is_file()
    points = _loss_step_points(metrics_csv)
    assert len(points) >= 10
    last = points[-10:]
    slope = float(np.polyfit([step for step, _ in last], [loss for _, loss in last], 1)[0])
    assert slope < 0


class TestResolveConfig:
    @pytest.mark.fast
    def test_resolves_name_with_and_without_suffix(self):
        expected = Path("configs/physicalai") / "act_dinner.yaml"
        assert resolve_config("act_dinner") == expected
        assert resolve_config("act_dinner.yaml") == expected

    @pytest.mark.fast
    def test_unknown_name_clear_error(self):
        with pytest.raises(StudioTrainError, match="no training config"):
            resolve_config("does_not_exist")


class TestPreflight:
    @pytest.fixture
    def config_factory(self, tmp_path):
        def _write(root, repo_id):
            config = tmp_path / "config.yaml"
            config.write_text(
                "data:\n"
                "  class_path: physicalai.data.lerobot.LeRobotDataModule\n"
                "  init_args:\n"
                f"    repo_id: {repo_id}\n"
                f"    root: {root}\n",
                encoding="utf-8",
            )
            return config

        return _write

    @pytest.fixture
    def broken_train_import(self, monkeypatch):
        def _raise():
            raise ImportError("No module named 'physicalai.train'")

        monkeypatch.setattr(studio_train, "_import_train_module", _raise)

    @pytest.fixture
    def stubbed_train_import(self, monkeypatch):
        monkeypatch.setattr(studio_train, "_import_train_module", lambda: None)

    @pytest.mark.fast
    def test_missing_root_fetch_failure_clear_error(self, tmp_path, config_factory, monkeypatch):
        config = config_factory(str(tmp_path / "missing"), ARTIFACT_DATASET)

        def _fail(**kwargs):
            raise RuntimeError("hub unreachable")

        monkeypatch.setattr(studio_train, "snapshot_download", _fail)
        with pytest.raises(StudioTrainError, match="could not be fetched"):
            preflight(config)

    @pytest.mark.fast
    def test_missing_root_non_artifact_clear_error(self, tmp_path, config_factory):
        config = config_factory(str(tmp_path / "missing"), "other/dataset")
        with pytest.raises(StudioTrainError, match="meta/info.json"):
            preflight(config)

    @pytest.mark.fast
    def test_missing_package_clear_error(self, config_factory, broken_train_import):
        config = config_factory("null", "lerobot/pusht")
        with pytest.raises(StudioTrainError, match="not importable"):
            preflight(config)

    @pytest.mark.fast
    def test_wandb_env_required_only_for_full_runs(
        self, config_factory, stubbed_train_import, monkeypatch
    ):
        config = config_factory("null", "lerobot/pusht")
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        with pytest.raises(StudioTrainError, match="W&B"):
            preflight(config)
        preflight(config, smoke=True)

    @pytest.mark.fast
    def test_local_dataset_root_passes(
        self, tmp_path, config_factory, stubbed_train_import, monkeypatch
    ):
        root = tmp_path / "dataset"
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
        config = config_factory(str(root), "dinner-table/train")
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        assert preflight(config, smoke=True)["data"]["init_args"]["root"] == str(root)
