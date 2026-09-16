"""YOLO training on the A11 label pool (A13).

`render_yolo_set` writes one flat images/labels pool; this module splits it
deterministically into train/val, trains `yolo11n`, copies `best.pt` to the
artifact path, and reports per-class mAP@50.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

import yaml

from dinner_table.config import DinnerTableError

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = "yolo11n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.1


class YoloTrainError(DinnerTableError):
    """Raised when the YOLO dataset or training setup is invalid."""


def prepare_yolo_split(
    data: str | Path, split_dir: str | Path, val_fraction: float = DEFAULT_VAL_FRACTION
) -> Path:
    """Split the flat A11 pool into train/val dirs; return the split data.yaml.

    Assignment is deterministic: a stem lands in val when the first sha1 byte
    falls below the val fraction. Single-sided outcomes are repaired so both
    splits are non-empty.
    """
    data_path, split_path = Path(data), Path(split_dir)
    spec = yaml.safe_load((data_path / "data.yaml").read_text(encoding="utf-8"))
    stems = sorted(p.stem for p in (data_path / "images").glob("*.jpg"))
    if not stems:
        raise YoloTrainError(f"no training images under {data_path / 'images'}")
    train, val = [], []
    for stem in stems:
        digest = hashlib.sha1(stem.encode("utf-8")).digest()[0] / 256.0
        (val if digest < val_fraction else train).append(stem)
    if not val:
        val.append(train.pop())
    if not train:
        train.append(val.pop(0))
    for split, names in (("train", train), ("val", val)):
        for stem in names:
            for sub in ("images", "labels"):
                src = data_path / sub / f"{stem}.{'jpg' if sub == 'images' else 'txt'}"
                if not src.is_file():
                    raise YoloTrainError(f"missing {sub} file for frame: {stem}")
                dst = split_path / split / sub / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    yaml_path = split_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(split_path.resolve()),
                "train": "train/images",
                "val": "val/images",
                "nc": spec["nc"],
                "names": spec["names"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("yolo split: %d train / %d val under %s", len(train), len(val), split_path)
    return yaml_path


def train_yolo(
    data: str | Path = "datasets/yolo",
    out: str | Path = "artifacts/yolo_best.pt",
    weights: str = DEFAULT_WEIGHTS,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    device: str | None = None,
    project: str | Path = "artifacts/yolo_runs",
    name: str = "train",
    exist_ok: bool = False,
    verbose: bool = False,
    workers: int = 0,
) -> dict:
    """Train YOLO, copy best.pt to `out`, and return mAP@50 metrics."""
    from ultralytics import YOLO

    data_path, out_path = Path(data), Path(out)
    split_yaml = prepare_yolo_split(data_path, out_path.parent / "yolo_split", val_fraction)
    model = YOLO(weights)
    model.train(
        data=str(split_yaml),
        epochs=epochs,
        imgsz=imgsz,
        seed=seed,
        deterministic=True,
        device=device,
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        verbose=verbose,
        workers=workers,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_path)
    metrics = model.val(data=str(split_yaml), split="val", verbose=False)
    names = model.names
    per_class = {str(names[i]): float(ap) for i, ap in enumerate(metrics.box.ap50) if i in names}
    report = {"map50": float(metrics.box.map50), "per_class": per_class, "best": str(out_path)}
    for label, ap in sorted(per_class.items()):
        print(f"{label}: mAP@50={ap:.4f}")
    print(f"mean: mAP@50={report['map50']:.4f}")
    logger.info("yolo training done: %s", json.dumps(report, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; runs on the GPU box. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Train yolo11n on the A11 label pool")
    parser.add_argument("--data", default="datasets/yolo")
    parser.add_argument("--out", default="artifacts/yolo_best.pt")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_yolo(
        data=args.data,
        out=args.out,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
