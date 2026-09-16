"""A13: detector protocol, grounding, tracker, multiview gate, YOLO training."""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from dinner_table.perception.depth_fusion import multiview_check
from dinner_table.perception.detector import DetectorError, UltralyticsDetector
from dinner_table.perception.interfaces import (
    Detection,
    Detector,
    ObjectPose3D,
    PerceptionSnapshot,
    Tracker,
)
from dinner_table.perception.tracker import ObjectTracker
from dinner_table.perception.yolo_train import prepare_yolo_split, train_yolo

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
R = np.eye(3)
T = np.array([0.0, 0.0, 1.5])


def _detector() -> UltralyticsDetector:
    return UltralyticsDetector(camera_intrinsics=K, camera_position=T, camera_rotation=R)


def _snapshot(depth: np.ndarray, detections: list[Detection]) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        overhead_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        overhead_depth=depth.astype(np.float32),
        detections=tuple(detections),
        timestamp=0.0,
    )


def _box(cx: int, cy: int, half: int = 10) -> tuple[float, float, float, float]:
    return (float(cx - half), float(cy - half), float(cx + half), float(cy + half))


def test_detector_protocol():
    assert isinstance(_detector(), Detector)
    assert isinstance(ObjectTracker(), Tracker)
    assert list(inspect.signature(UltralyticsDetector.ground).parameters) == [
        "self",
        "snapshot",
        "target",
    ]
    with pytest.raises(DetectorError, match="model"):
        _detector().detect(np.zeros((480, 640, 3), dtype=np.uint8))


def test_ground_center_exact():
    snap = _snapshot(
        np.full((480, 640), 1.0),
        [Detection(label="plate", xyxy=_box(320, 240), confidence=0.9)],
    )
    pose = _detector().ground(snap, "plate")
    assert pose is not None
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 2.5], atol=1e-9)
    assert pose.name == "plate"
    assert pose.held_by is None


def test_ground_offset_exact():
    snap = _snapshot(
        np.full((480, 640), 2.0),
        [Detection(label="mug", xyxy=_box(420, 240), confidence=0.9)],
    )
    pose = _detector().ground(snap, "mug")
    assert pose is not None
    np.testing.assert_allclose(pose.position, [0.4, 0.0, 3.5], atol=1e-9)


def test_ground_accuracy_randomized():
    rng = np.random.default_rng(0)
    detector = _detector()
    for _ in range(100):
        cx, cy = int(rng.integers(10, 630)), int(rng.integers(10, 470))
        z = float(rng.uniform(0.3, 3.5))
        snap = _snapshot(
            np.full((480, 640), z),
            [Detection(label="bottle", xyxy=_box(cx, cy), confidence=0.9)],
        )
        pose = detector.ground(snap, "bottle")
        assert pose is not None
        expected = np.array([(cx - 320) / 500 * z, (cy - 240) / 500 * z, 1.5 + z])
        assert abs(pose.position[0] - expected[0]) <= 0.02
        assert abs(pose.position[1] - expected[1]) <= 0.02
        assert abs(pose.position[2] - expected[2]) <= 0.03


def test_ground_skips_invalid_depth():
    depth = np.zeros((480, 640))
    depth[238:243, 418:423] = 2.0
    snap = _snapshot(
        depth,
        [
            Detection(label="mug", xyxy=_box(100, 100), confidence=0.95),
            Detection(label="mug", xyxy=_box(420, 240), confidence=0.8),
        ],
    )
    pose = _detector().ground(snap, "mug")
    assert pose is not None
    np.testing.assert_allclose(pose.position, [0.4, 0.0, 3.5], atol=1e-9)


def test_ground_none_cases():
    detector = _detector()
    snap = _snapshot(
        np.zeros((480, 640)), [Detection(label="mug", xyxy=_box(320, 240), confidence=0.9)]
    )
    assert detector.ground(snap, "mug") is None
    assert detector.ground(snap, "plate") is None
    assert detector.ground(_snapshot(np.full((480, 640), 1.0), []), "mug") is None


def test_tracker_recovery(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    tracker = ObjectTracker()
    pose = ObjectPose3D(name="plate", position=np.array([1.0, 0.0, 0.5]), held_by=None)
    for i in range(3):
        clock[0] = i * 0.1
        out = tracker.update({"plate": pose})
        np.testing.assert_allclose(out["plate"].position, [1.0, 0.0, 0.5])
    clock[0] = 0.5
    held = tracker.update({})
    np.testing.assert_allclose(held["plate"].position, [1.0, 0.0, 0.5])
    clock[0] = 0.6
    moved = ObjectPose3D(name="plate", position=np.array([1.2, 0.0, 0.5]), held_by=None)
    out = tracker.update({"plate": moved})
    np.testing.assert_allclose(out["plate"].position, [1.0, 0.0, 0.5])
    clock[0] = 5.0
    assert tracker.update({}) == {}


def test_tracker_median_and_fuse():
    tracker = ObjectTracker(ee_provider=lambda: {"A": np.array([0.0, 0.0, 0.0])})
    for i in range(5):
        out = tracker.update(
            {"mug": ObjectPose3D(name="mug", position=np.array([float(i), 0.0, 0.0]), held_by=None)}
        )
    np.testing.assert_allclose(out["mug"].position, [2.0, 0.0, 0.0])
    near = {"mug": ObjectPose3D(name="mug", position=np.array([0.03, 0.0, 0.0]), held_by=None)}
    assert tracker.fuse_held(near, {"A": 0.5})["mug"].held_by == "A"
    assert tracker.fuse_held(near, {"A": 0.7})["mug"].held_by is None
    far = {"mug": ObjectPose3D(name="mug", position=np.array([1.0, 0.0, 0.0]), held_by=None)}
    assert tracker.fuse_held(far, {"A": 0.1})["mug"].held_by is None
    assert tracker.fuse_held(near, {"B": 0.1})["mug"].held_by is None


def test_multiview_check():
    det = [Detection(label="plate", xyxy=_box(320, 240), confidence=0.9)]
    pose = ObjectPose3D(name="plate", position=np.zeros(3), held_by=None)
    assert multiview_check(_snapshot(np.full((480, 640), 1.0), det), pose)
    assert not multiview_check(_snapshot(np.zeros((480, 640)), det), pose)
    assert not multiview_check(_snapshot(np.full((480, 640), 9.0), det), pose)
    assert not multiview_check(_snapshot(np.full((480, 640), 1.0), []), pose)
    single = np.zeros((480, 640))
    single[240, 320] = 1.0
    assert not multiview_check(_snapshot(single, det), pose)


def _write_pool(root: Path, n: int, names: list[str]) -> Path:
    for split in ("images", "labels"):
        (root / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (64, 64), (32, 32, 32))
        draw = ImageDraw.Draw(img)
        cls = i % len(names)
        color = (220, 30, 30) if cls == 0 else (30, 30, 220)
        draw.rectangle([8, 8, 40, 40], fill=color)
        img.save(root / "images" / f"f{i:04d}.jpg")
        (root / "labels" / f"f{i:04d}.txt").write_text(
            "0 0.375 0.375 0.5 0.5\n" if cls == 0 else "1 0.375 0.375 0.5 0.5\n"
        )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images",
                "val": "images",
                "nc": len(names),
                "names": names,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_prepare_yolo_split_deterministic(tmp_path):
    pool = _write_pool(tmp_path / "pool", 10, ["plate", "mug"])
    first = prepare_yolo_split(pool, tmp_path / "s1")
    prepare_yolo_split(pool, tmp_path / "s2")
    spec = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert spec["nc"] == 2 and spec["names"] == ["plate", "mug"]
    for split in ("train", "val"):
        a = sorted(p.name for p in (tmp_path / "s1" / split / "images").glob("*.jpg"))
        b = sorted(p.name for p in (tmp_path / "s2" / split / "images").glob("*.jpg"))
        assert a == b and a
    tiny = _write_pool(tmp_path / "tiny", 2, ["plate", "mug"])
    prepare_yolo_split(tiny, tmp_path / "s3")
    assert list((tmp_path / "s3" / "train" / "images").glob("*.jpg"))
    assert list((tmp_path / "s3" / "val" / "images").glob("*.jpg"))


class _SyncPool:
    """Synchronous ThreadPool stand-in: some sandboxes deny the semaphores
    multiprocessing pools need, while label scanning is semantically serial."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap(self, func, iterable):
        return map(func, iterable)


@pytest.mark.slow
@pytest.mark.smoke
def test_yolo_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr("ultralytics.data.dataset.ThreadPool", _SyncPool)
    pool = _write_pool(tmp_path / "pool", 200, ["plate", "mug"])
    report = train_yolo(
        data=pool,
        out=tmp_path / "yolo_best.pt",
        weights="yolo11n.yaml",
        epochs=2,
        imgsz=128,
        seed=42,
        device="cpu",
        project=tmp_path / "runs",
    )
    assert (tmp_path / "yolo_best.pt").is_file()
    assert sorted(report["per_class"]) == ["mug", "plate"]
    assert 0.0 <= report["map50"] <= 1.0


@pytest.mark.nightly
def test_yolo_map(tmp_path):
    ckpt = Path("artifacts/yolo_best.pt")
    data = Path("datasets/yolo")
    if not ckpt.is_file() or not (data / "data.yaml").is_file():
        pytest.skip("needs the GPU-trained checkpoint and datasets/yolo")
    from ultralytics import YOLO

    split_yaml = prepare_yolo_split(data, tmp_path / "split")
    metrics = YOLO(str(ckpt)).val(data=str(split_yaml), split="val", verbose=False)
    assert float(metrics.box.map50) >= 0.95
    holdout = Path("datasets/yolo_holdout")
    if not (holdout / "data.yaml").is_file():
        pytest.skip("texture-holdout set not rendered")
    holdout_yaml = prepare_yolo_split(holdout, tmp_path / "holdout_split", val_fraction=1.0)
    holdout_metrics = YOLO(str(ckpt)).val(data=str(holdout_yaml), split="val", verbose=False)
    assert float(holdout_metrics.box.map50) >= 0.85
