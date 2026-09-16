"""Free YOLO labels: GT geom AABBs projected to overhead pixels, no humans.

`label_frame` follows the Phase 3 sketch: each labeled body's own geoms yield
world-frame corners, projected through the analytic camera intrinsics, boxed
in pixels, and normalized to `cls xc yc w h`. `render_yolo_set` replays demo
episodes and writes `images/*.jpg` + `labels/*.txt` + `data.yaml`, sampling
every Nth frame plus every perturbation-event frame (anomaly-adjacent states
are the detector's hardest cases).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import DRAWER_TRAVEL
from dinner_table.data.lerobot_export import replay_frames
from dinner_table.perception.interfaces import OBJECT_LABELS
from dinner_table.scene.cameras import OVERHEAD_RESOLUTION, POLICY_IMAGE_SIZE, CameraRig

logger = logging.getLogger(__name__)

EVERY_NTH = 4
MAX_YOLO_FRAMES = 6000
LABEL_PRECISION = 6
DRAWER_BODY = "drawer_top"


class LabelsError(DinnerTableError):
    """Exception raised for label projection or YOLO set failures."""


@dataclass(frozen=True)
class LabeledBox:
    """One YOLO-normalized box: class name plus center-format coords in 0..1."""

    cls: str
    xc: float
    yc: float
    w: float
    h: float

    def to_line(self) -> str:
        """Ultralytics label line for this box."""
        index = OBJECT_LABELS.index(self.cls)
        values = [self.xc, self.yc, self.w, self.h]
        numbers = " ".join(f"{v:.{LABEL_PRECISION}f}" for v in values)
        return f"{index} {numbers}"

    def xyxy_pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Denormalize back to pixel corners (for the IoU audit)."""
        w, h = self.w * width, self.h * height
        x, y = self.xc * width, self.yc * height
        return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)


def _local_corners(geom_type: int, size: np.ndarray) -> np.ndarray:
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        ext = np.abs(size)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        ext = np.array([size[0], size[0], size[0]])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        ext = np.array([size[0], size[0], size[1]])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        ext = np.array([size[0], size[0], size[1] + size[0]])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        ext = np.abs(size)
    else:
        raise LabelsError(f"unsupported label geom type: {geom_type}")
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    return signs * ext


def _body_corners(model, data, body_id: int) -> np.ndarray:
    corners = []
    first = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    for gid in range(first, first + count):
        local = _local_corners(int(model.geom_type[gid]), np.asarray(model.geom_size[gid]))
        rot = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
        corners.append(np.asarray(data.geom_xpos[gid]) + local @ rot.T)
    if not corners:
        raise LabelsError(f"body id {body_id} has no geoms to label")
    return np.concatenate(corners, axis=0)


def _project(
    K: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray, points: np.ndarray
) -> np.ndarray | None:
    rel = points - cam_pos
    cam = rel @ cam_mat
    depth = -cam[:, 2]
    if bool(np.any(depth <= 1e-6)):
        return None
    u = K[0, 0] * cam[:, 0] / depth + K[0, 2]
    v = -K[1, 1] * cam[:, 1] / depth + K[1, 2]
    return np.stack([u, v], axis=1)


def _drawer_class(model, data) -> str:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    qadr = int(model.jnt_qposadr[joint])
    if float(data.qpos[qadr]) > DRAWER_TRAVEL * 0.5:
        return "drawer_open"
    return "drawer_closed"


def _resolution(camera: str) -> tuple[int, int]:
    if camera in ("wrist_A", "wrist_B"):
        return POLICY_IMAGE_SIZE
    return OVERHEAD_RESOLUTION


def label_frame(model, data, camera: str = "overhead") -> list[LabeledBox]:
    """Project every labeled body's GT geoms to YOLO-normalized boxes."""
    rig = CameraRig(model)
    K = rig.intrinsics(camera)
    height, width = _resolution(camera)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id == -1:
        raise LabelsError(f"camera not found in model: {camera}")
    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    boxes = []
    for name in OBJECT_LABELS:
        body = DRAWER_BODY if name.startswith("drawer_") else name
        if name.startswith("drawer_") and name != _drawer_class(model, data):
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid == -1:
            raise LabelsError(f"labeled body missing from model: {body}")
        uv = _project(K, cam_pos, cam_mat, _body_corners(model, data, bid))
        if uv is None:
            continue
        umin, vmin = float(np.min(uv[:, 0])), float(np.min(uv[:, 1]))
        umax, vmax = float(np.max(uv[:, 0])), float(np.max(uv[:, 1]))
        umin = float(np.clip(umin, 0.0, width))
        umax = float(np.clip(umax, 0.0, width))
        vmin = float(np.clip(vmin, 0.0, height))
        vmax = float(np.clip(vmax, 0.0, height))
        boxes.append(
            LabeledBox(
                cls=name,
                xc=(umin + umax) / 2.0 / width,
                yc=(vmin + vmax) / 2.0 / height,
                w=(umax - umin) / width,
                h=(vmax - vmin) / height,
            )
        )
    return boxes


def render_yolo_set(
    demos: str | Path = "demos",
    out: str | Path = "datasets/yolo",
    every: int = EVERY_NTH,
    max_frames: int = MAX_YOLO_FRAMES,
    render: bool = True,
) -> dict:
    """Replay demos and write images/labels/data.yaml; returns the summary.

    Every Nth frame plus every event frame is labeled, over successful and
    failed episodes alike (a dropped plate is still a plate for detection).
    Rendering needs GL; ``render=False`` writes labels and data.yaml only.
    """
    demos_path = Path(demos)
    manifest = json.loads((demos_path / "manifest.json").read_text(encoding="utf-8"))
    out_path = Path(out)
    images = out_path / "images"
    labels = out_path / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    if render:
        images.mkdir(parents=True, exist_ok=True)
        from PIL import Image
    count = 0
    episodes = 0
    for entry in sorted(manifest["episodes"], key=lambda e: e["seed"]):
        log = json.loads(
            (demos_path / entry["split"] / f"{entry['episode_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        episodes += 1
        for index, (frame, _cond, scene) in enumerate(replay_frames(log, entry["graph"])):
            if index % every != 0 and not frame.get("event"):
                continue
            boxes = label_frame(scene.model, scene.data, "overhead")
            stem = f"{log['episode_id']}_f{int(frame['tick']):05d}"
            (labels / f"{stem}.txt").write_text(
                "\n".join(box.to_line() for box in boxes) + "\n", encoding="utf-8"
            )
            if render:
                Image.fromarray(scene.render("overhead")).save(images / f"{stem}.jpg")
            count += 1
            if count >= max_frames:
                break
        if count >= max_frames:
            break
    # NOTE: one flat pool; A13's trainer splits it (train/val point at images).
    data_yaml = {
        "path": str(out_path),
        "train": "images",
        "val": "images",
        "nc": len(OBJECT_LABELS),
        "names": list(OBJECT_LABELS),
    }
    (out_path / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=True), encoding="utf-8")
    logger.info("labeled %d frames from %d episodes under %s", count, episodes, out)
    return {"frames": count, "episodes": episodes, "out": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Render the free YOLO label set")
    parser.add_argument("out", nargs="?", default="datasets/yolo", help="output directory")
    parser.add_argument("--demos", default="demos", help="demo_gen output directory")
    parser.add_argument("--every", type=int, default=EVERY_NTH, help="label every Nth frame")
    parser.add_argument("--max-frames", type=int, default=MAX_YOLO_FRAMES)
    parser.add_argument("--render", dest="render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = render_yolo_set(args.demos, args.out, args.every, args.max_frames, args.render)
    print(f"labeled {result['frames']} frames from {result['episodes']} episodes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
