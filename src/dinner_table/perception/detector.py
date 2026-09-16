"""YOLO detector on the torch backend (§10.4 Detector protocol, frozen signatures)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.perception.interfaces import Detection, ObjectPose3D, PerceptionSnapshot

MAX_DEPTH_M = 4.0  # mirror of runtime/engines.py (which needs OpenVINO to import)


class DetectorError(DinnerTableError):
    """Raised when detection or grounding cannot run."""


class UltralyticsDetector:
    """torch-backend YOLO detector; `ground` mirrors OpenvinoDetector exactly."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        camera_intrinsics: np.ndarray | None = None,
        camera_position: np.ndarray | None = None,
        camera_rotation: np.ndarray | None = None,
    ) -> None:
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._intrinsics = camera_intrinsics
        self._cam_position = camera_position
        self._cam_rotation = camera_rotation
        self._model = None
        if model_path is not None:
            from ultralytics import YOLO

            self._model = YOLO(str(model_path))

    def detect(self, overhead_rgb: np.ndarray) -> list[Detection]:
        """Run detection on one (480, 640, 3) uint8 overhead frame."""
        if self._model is None:
            raise DetectorError("detect requires a model path at construction")
        result = self._model.predict(overhead_rgb, conf=self._conf, iou=self._iou, verbose=False)[0]
        names = self._model.names
        detections = []
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            detections.append(
                Detection(
                    label=str(names[int(cls)]),
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(conf),
                )
            )
        return detections

    def ground(self, snapshot: PerceptionSnapshot, target: str) -> ObjectPose3D | None:
        """3D grounding of `target` from the snapshot's detections + depth.

        Back-projects the best detection's box-center median depth (5x5
        kernel; invalid depth skips to the next-best detection) through the
        overhead camera geometry; the detector never sets held_by.
        """
        if self._intrinsics is None or self._cam_position is None or self._cam_rotation is None:
            raise DetectorError("grounding requires camera geometry at construction")
        depth = snapshot.overhead_depth
        candidates = sorted(
            (d for d in snapshot.detections if d.label == target),
            key=lambda d: -d.confidence,
        )
        for detection in candidates:
            cx = int((detection.xyxy[0] + detection.xyxy[2]) / 2)
            cy = int((detection.xyxy[1] + detection.xyxy[3]) / 2)
            y0, x0 = max(0, cy - 2), max(0, cx - 2)
            patch = depth[y0 : y0 + 5, x0 : x0 + 5]
            valid = patch[(patch > 0) & (patch <= MAX_DEPTH_M)]
            # NOTE: the spec's "two consistent detections" is read as two
            # consistent depth samples inside the kernel.
            if valid.size < 2:
                continue
            z = float(np.median(valid))
            ray = np.linalg.inv(self._intrinsics) @ np.array([cx, cy, 1.0])
            camera_point = ray * z
            world = self._cam_rotation @ camera_point + self._cam_position
            return ObjectPose3D(name=target, position=world, held_by=None)
        return None
