"""Deployment-engine tests on public stand-ins: protocol conformance, detector
latency (recorded to bench/engines_smoke.json), detector equivalence against
the torch path, and VLM parse plus boundary-call latency."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
# The Intel stack has no macOS wheels (see pyproject's sys_platform markers), so
# the module-level imports below abort COLLECTION of the whole suite on a dev Mac
# rather than being skipped by the pytestmark further down. Skip at import time
# instead; on the Intel target host the stack is present and nothing is skipped.
pytest.importorskip("openvino", reason="Intel stack (openvino) not installed on this host")

from openvino import Core as OvCore
from ultralytics import YOLO
from ultralytics.utils import ASSETS

from dinner_table.perception.interfaces import Detector
from dinner_table.reasoning.interfaces import VlmEngine
from dinner_table.reasoning.schema import SceneSummary, TaskGraph
from dinner_table.runtime.engines import (
    OpenvinoDetector,
    OvGenaiVlmEngine,
    fetch_standin_vlm,
    fetch_standin_yolo,
)

LATENCY_ITERATIONS = 100
GPU_P50_LIMIT_MS = 8.0
EQUIVALENCE_IMAGES = 50
EQUIVALENCE_MIN_IOU = 0.98
VLM_BOUNDARY_LIMIT_S = 3.0
PARSE_INSTRUCTIONS = (
    "Pick up the plate with arm A.",
    "Open the top drawer.",
    "Pick up the fork with arm B.",
)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


pytestmark = [
    pytest.mark.openvino,
    pytest.mark.skipif(not _has("openvino"), reason="openvino not installed"),
]


@pytest.fixture(scope="module")
def standin_yolo():
    return fetch_standin_yolo()


@pytest.fixture(scope="module")
def detector_cpu(standin_yolo):
    onnx_path, names = standin_yolo
    return OpenvinoDetector(onnx_path, names, "CPU")


@pytest.fixture(scope="module")
def vlm_engine():
    return OvGenaiVlmEngine(fetch_standin_vlm(), "CPU")


def _render_variant_images() -> list[np.ndarray]:
    """EQUIVALENCE_IMAGES deterministic crops/flips/tones of the bundled
    public sample photos (each keeps detectable content)."""
    bases = [cv2.imread(str(path)) for path in sorted(Path(ASSETS).glob("*.jpg"))]
    images = []
    for index in range(EQUIVALENCE_IMAGES):
        img = bases[index % len(bases)]
        height, width = img.shape[:2]
        crop_h = int(height * (0.6 + 0.4 * ((index * 37) % 10) / 10))
        crop_w = int(width * (0.6 + 0.4 * ((index * 53) % 10) / 10))
        y0, x0 = (index * 97) % (height - crop_h), (index * 131) % (width - crop_w)
        out = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
        if index % 3 == 0:
            out = out[:, ::-1]
        if index % 5 == 0:
            out = cv2.resize(out, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        if index % 2 == 0:
            out = cv2.convertScaleAbs(out, alpha=0.8 + 0.4 * ((index % 7) / 7))
        images.append(out)
    return images


def _box_iou(a: tuple[float, float, float, float], b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-12)


class TestProtocols:
    def test_engines_satisfy_the_frozen_protocols(self, detector_cpu, vlm_engine):
        assert isinstance(detector_cpu, Detector)
        assert isinstance(vlm_engine, VlmEngine)


class TestDetectorLatency:
    def test_latency_recorded_to_bench(self, standin_yolo, detector_cpu):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        detectors = {"CPU": detector_cpu}
        # multi-GPU hosts expose GPU.0, GPU.1, ...; single-GPU hosts expose GPU
        gpus = [d for d in OvCore().available_devices if d == "GPU" or d.startswith("GPU.")]
        if gpus:
            onnx_path, names = standin_yolo
            detectors[gpus[0]] = OpenvinoDetector(onnx_path, names, gpus[0])
        results = {}
        for device, detector in detectors.items():
            detector.detect(frame)
            times = []
            for _ in range(LATENCY_ITERATIONS):
                start = time.perf_counter()
                detector.detect(frame)
                times.append((time.perf_counter() - start) * 1000.0)
            times.sort()
            results[device] = {
                "p50_ms": round(times[len(times) // 2], 3),
                "p95_ms": round(times[int(len(times) * 0.95)], 3),
            }
        assert results[gpus[0]]["p50_ms"] < GPU_P50_LIMIT_MS
        bench_dir = Path("bench")
        bench_dir.mkdir(exist_ok=True)
        (bench_dir / "engines_smoke.json").write_text(
            json.dumps({"yolo": results}, indent=2) + "\n", encoding="utf-8"
        )


class TestDetectorEquivalence:
    def test_matches_the_torch_path(self, standin_yolo, detector_cpu):
        onnx_path, names = standin_yolo
        reference = YOLO(str(onnx_path.parent / "yolo11n.pt"))
        worst_iou = 1.0
        total_detections = 0
        for image in _render_variant_images():
            detections = detector_cpu.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result = reference.predict(image, conf=0.4, iou=0.5, imgsz=640, verbose=False)[0]
            ref_boxes = result.boxes.xyxy.numpy()
            ref_classes = result.boxes.cls.numpy().astype(int)
            assert sorted(d.label for d in detections) == sorted(names[c] for c in ref_classes)
            total_detections += len(detections)
            for detection in detections:
                same_class = [
                    box for box, cls in zip(ref_boxes, ref_classes) if names[cls] == detection.label
                ]
                best = max(_box_iou(detection.xyxy, box) for box in same_class)
                worst_iou = min(worst_iou, best)
        assert total_detections > 0
        assert worst_iou >= EQUIVALENCE_MIN_IOU


class TestVlmEngine:
    def test_parse_and_boundary_latency(self, vlm_engine):
        summary = SceneSummary(
            instruction="set the table",
            objects={},
            drawer_open=False,
            completed_step_ids=[],
        )
        for instruction in PARSE_INSTRUCTIONS:
            vlm_engine.submit_parse(instruction, summary)
            deadline = time.monotonic() + 120.0
            while not vlm_engine.ready:
                assert time.monotonic() < deadline, f"parse of {instruction!r} timed out"
                time.sleep(0.25)
            graph = vlm_engine.result()
            assert graph is not None, vlm_engine.last_error()
            assert isinstance(graph, TaskGraph)
            assert graph.instruction
        graph = TaskGraph.model_validate(
            {
                "task_id": "t",
                "instruction": "pick the plate",
                "steps": [{"id": 1, "skill": "pick", "arm": "A", "object": "plate"}],
            }
        )
        times = []
        for _ in range(5):
            start = time.perf_counter()
            report = vlm_engine.check_preconditions(graph, 1, summary)
            times.append(time.perf_counter() - start)
            assert isinstance(report.ok, bool)
        times.sort()
        assert times[2] < VLM_BOUNDARY_LIMIT_S
