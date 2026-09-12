"""Studio training harness: a thin, exact wrapper around Intel's `physicalai fit`.

Never re-implements training. It resolves a config under configs/physicalai/,
runs the preflight gates (dataset present, training stack importable, W&B
environment set), invokes the identical CLI command via subprocess (so the
laptop, CI, and the cloud GPU all execute the same string), captures the
Lightning run id, and records run provenance (config hash, dataset hash, git
revision) in runs/<run_id>/meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from dinner_table.config import ARTIFACT_DATASET, DinnerTableError

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("configs/physicalai")
RUNS_DIR = Path("runs")
DEFAULT_EXPERIMENTS_DIR = Path("experiments")
DEFAULT_EXPERIMENT_NAME = "lightning_logs"
SMOKE_MAX_STEPS = 50
SMOKE_LOG_EVERY_N_STEPS = 1


class StudioTrainError(DinnerTableError):
    """Training-harness misuse or preflight failure."""


def resolve_config(name: str) -> Path:
    """Path of a training config under configs/physicalai/ ('.yaml' optional)."""
    if name.endswith(".yaml"):
        candidate = CONFIG_DIR / name
    else:
        candidate = CONFIG_DIR / f"{name}.yaml"
    if candidate.is_file():
        return candidate
    available: list[str] = []
    if CONFIG_DIR.is_dir():
        available = sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))
    raise StudioTrainError(
        f"no training config {candidate} (available under {CONFIG_DIR}: "
        f"{', '.join(available) if available else 'none'})"
    )


def _import_train_module() -> None:
    """Import the training package so a broken install fails in preflight, not mid-run."""
    importlib.import_module("physicalai.train")


def _load_config(config_path: Path) -> dict:
    """Parsed YAML config; raises StudioTrainError on unreadable or invalid files."""
    try:
        with Path(config_path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        raise StudioTrainError(f"cannot read training config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise StudioTrainError(f"training config {config_path} is not valid YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise StudioTrainError(f"training config {config_path} must be a YAML mapping")
    return config


def _is_lerobot_dataset(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file()


def _fetch_artifact_dataset(root: Path) -> None:
    try:
        snapshot_download(repo_id=ARTIFACT_DATASET, repo_type="dataset", local_dir=str(root))
    except Exception as exc:
        raise StudioTrainError(
            f"dataset root {root} is missing and {ARTIFACT_DATASET} could not be "
            f"fetched from the hub: {exc}"
        ) from exc


def _require_wandb_env() -> None:
    mode = os.environ.get("WANDB_MODE")
    if os.environ.get("WANDB_API_KEY"):
        return
    if mode in ("offline", "disabled"):
        return
    raise StudioTrainError(
        "W&B environment not set for a full training run: export WANDB_API_KEY "
        "(or WANDB_MODE=offline / disabled), or run with --smoke"
    )


def preflight(config_path: Path, smoke: bool = False) -> dict:
    """Gate a training run; returns the parsed config, raises StudioTrainError.

    Checks, in order: config readable, dataset root a local LeRobot dataset
    (fetched from ARTIFACT_DATASET when it is the configured repo), the
    physicalai-train package importable, and the W&B environment set (skipped
    for smoke runs).
    """
    config = _load_config(config_path)
    init_args = (config.get("data") or {}).get("init_args") or {}
    root = init_args.get("root")
    repo_id = init_args.get("repo_id")
    if root is not None and not _is_lerobot_dataset(Path(root)):
        if repo_id == ARTIFACT_DATASET:
            _fetch_artifact_dataset(Path(root))
        else:
            raise StudioTrainError(
                f"dataset root {root} is not a LeRobot dataset (meta/info.json missing); "
                "generate or download it, or clear data.init_args.root to stream from the hub"
            )
    try:
        _import_train_module()
    except ImportError as exc:
        raise StudioTrainError(f"physicalai-train is not importable: {exc}") from exc
    if not smoke:
        _require_wandb_env()
    return config


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_rev() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip()


def _lerobot_home() -> Path:
    if os.environ.get("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "lerobot"
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def _dataset_hash(repo_id: str | None, root: str | None) -> str:
    """sha256 of the dataset's meta/info.json when a local copy is resolvable,
    else an identity hash of the repo id (hub dataset not cached yet)."""
    bases: list[Path] = []
    if root is not None:
        bases.append(Path(root))
    elif repo_id is not None:
        snapshots = _lerobot_home() / "hub" / f"datasets--{repo_id.replace('/', '--')}"
        snapshots = snapshots / "snapshots"
        if snapshots.is_dir():
            bases.extend(sorted(snapshots.iterdir()))
    for base in bases:
        info = base / "meta" / "info.json"
        if info.is_file():
            return _sha256_bytes(info.read_bytes())
    return _sha256_bytes(f"{repo_id or 'unknown'}@default".encode())


def _override_value(overrides: Sequence[str], key: str, default: str) -> str:
    """Value of a `key value` or `key=value` CLI override; `default` when absent."""
    for index, token in enumerate(overrides):
        if token == key and index + 1 < len(overrides):
            return overrides[index + 1]
        if token.startswith(f"{key}="):
            return token.split("=", 1)[1]
    return default


def _version_numbers(experiment_dir: Path) -> set[int]:
    if not experiment_dir.is_dir():
        return set()
    numbers = set()
    for path in experiment_dir.iterdir():
        suffix = path.name.removeprefix("version_")
        if path.name.startswith("version_") and suffix.isdigit():
            numbers.add(int(suffix))
    return numbers


def run(
    config: Path,
    overrides: Sequence[str] = (),
    smoke: bool = False,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Train via `physicalai fit` and record provenance; returns the meta.json path.

    The subprocess runs `[sys.executable, -m, physicalai.cli.main, fit,
    --config <config>, *overrides]` (+ smoke caps), so the recorded command is
    the identical string for every machine. # NOTE: `runs_dir` only exists so
    tests can isolate provenance writes; callers use the cwd-relative default.
    """
    parsed = preflight(config, smoke=smoke)
    command = [
        sys.executable,
        "-m",
        "physicalai.cli.main",
        "fit",
        "--config",
        str(config),
        *overrides,
    ]
    if smoke:
        command += [
            "--trainer.max_steps",
            str(SMOKE_MAX_STEPS),
            "--trainer.log_every_n_steps",
            str(SMOKE_LOG_EVERY_N_STEPS),
        ]
    trainer_cfg = parsed.get("trainer") or {}
    experiments_root = Path(
        _override_value(overrides, "--trainer.default_root_dir", str(DEFAULT_EXPERIMENTS_DIR))
    )
    experiment_name = _override_value(
        overrides,
        "--trainer.experiment_name",
        str(trainer_cfg.get("experiment_name") or DEFAULT_EXPERIMENT_NAME),
    )
    experiment_dir_root = experiments_root / experiment_name
    before = _version_numbers(experiment_dir_root)
    env = dict(os.environ)
    if smoke:
        env["WANDB_MODE"] = "disabled"
    logger.info("training command: %s", " ".join(command))
    proc = subprocess.run(command, env=env, check=False)
    if proc.returncode != 0:
        raise StudioTrainError(f"physicalai fit failed with exit code {proc.returncode}")
    after = _version_numbers(experiment_dir_root)
    created = after - before
    if created:
        version = max(created)
        run_id = f"{experiment_name}_version_{version}"
        experiment_dir: Path | None = experiment_dir_root / f"version_{version}"
    else:
        # NOTE: a successful run that wrote no Lightning version dir (e.g. a
        # logger-disabled override) still gets provenance, keyed by wall clock.
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{experiment_name}_{stamp}"
        experiment_dir = None
    data_init = (parsed.get("data") or {}).get("init_args") or {}
    meta = {
        "run_id": run_id,
        "smoke": smoke,
        "command": command,
        "config": {"path": str(config), "sha256": _sha256_bytes(Path(config).read_bytes())},
        "dataset": {
            "repo_id": data_init.get("repo_id"),
            "root": data_init.get("root"),
            "sha256": _dataset_hash(data_init.get("repo_id"), data_init.get("root")),
        },
        "git_rev": _git_rev(),
        "experiment_dir": str(experiment_dir) if experiment_dir is not None else None,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


def main() -> None:
    """CLI entry: wrap `physicalai fit` with preflight gates and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="act_dinner", help="name under configs/physicalai/")
    parser.add_argument(
        "--smoke", action="store_true", help=f"cap at {SMOKE_MAX_STEPS} steps, no W&B"
    )
    args, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config_path = resolve_config(args.config)
        meta_path = run(config_path, overrides=overrides, smoke=args.smoke)
    except StudioTrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"run recorded: {meta_path}")


if __name__ == "__main__":
    main()
