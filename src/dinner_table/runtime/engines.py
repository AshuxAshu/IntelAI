"""Deployment engines: the OpenVINO-backed perception and reasoning services.

OpenvinoDetector runs YOLO on the OpenVINO runtime (ONNX input, LATENCY hint,
model cache) with numpy pre/post mirroring the ultralytics predict math
exactly. OvGenaiVlmEngine runs a converted VLM through openvino-genai with
grammar-constrained JSON decoding and the protocol's async submit/ready/result
seam. load_bundle constructs both per a device map; the ACT policy is NOT
loaded here - it is the physicalai runtime's InferenceModel, wired in the
skill source.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import openvino.properties as ov_props
import openvino.properties.hint as ov_hint
import openvino_genai as genai

from dinner_table.compat.openvino_genai_models import ensure_genai_tokenizers
from dinner_table.config import DinnerTableError
from dinner_table.perception.interfaces import (
    Detection,
    ObjectPose3D,
    PerceptionSnapshot,
)
from dinner_table.reasoning.schema import (
    PreconditionReport,
    SceneSummary,
    TaskGraph,
    VlmDiagnosis,
)

logger = logging.getLogger(__name__)

STANDIN_VLM_REPO = "madnesslab/Qwen2-VL-2B-Instruct-OpenVINO-INT8-SYM"
STANDIN_YOLO_TAG = "yolo11n"
LETTERBOX_SIZE = 640
STRIDE = 32
PAD_VALUE = 114
CLASS_OFFSET = 7680.0
VLM_MAX_IMAGE = 512
VLM_BUDGET_S = 3.0
MAX_DEPTH_M = 4.0


class EngineError(DinnerTableError):
    """Engine construction or inference failure."""


def _letterbox(
    image: np.ndarray, size: int = LETTERBOX_SIZE, auto: bool = True
) -> tuple[np.ndarray, float, float, float]:
    """Aspect-preserving resize with 114 padding, mirroring ultralytics' math.

    `auto` pads to the stride multiple (the torch predict path); otherwise the
    image is padded to the full square (static ONNX inputs). Returns the
    padded image, the scale gain, and the x/y padding."""
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    new_unpad = (round(width * gain), round(height * gain))
    pad_x, pad_y = size - new_unpad[0], size - new_unpad[1]
    if auto:
        pad_x, pad_y = pad_x % STRIDE, pad_y % STRIDE
    pad_x /= 2
    pad_y /= 2
    if (width, height) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return image, gain, pad_x, pad_y


def _greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy score-ordered NMS (the torchvision ops.nms mirror)."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        index = order[0]
        keep.append(int(index))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[index, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[index, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[index, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = (boxes[index, 2] - boxes[index, 0]) * (boxes[index, 3] - boxes[index, 1])
        area_b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        ious = inter / (area_a + area_b - inter + 1e-12)
        order = rest[ious <= iou_threshold]
    return np.asarray(keep, dtype=int)


class OpenvinoDetector:
    """YOLO on the OpenVINO runtime (§10.4 Detector protocol, frozen signatures).

    detect mirrors the ultralytics torch path exactly: rect letterbox when the
    model input is dynamic, square letterbox for static inputs (the same rule
    ultralytics' predictor applies), /255 normalization, best-class confidence
    filter, and class-offset greedy NMS."""

    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        device: str = "GPU",
        *,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        cache_dir: Path | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_position: np.ndarray | None = None,
        camera_rotation: np.ndarray | None = None,
    ) -> None:
        config: dict = {ov_hint.performance_mode: ov_hint.PerformanceMode.LATENCY}
        if cache_dir is not None:
            # ov_props.cache_dir(path) returns a (key, value) pair, not a dict key
            key, value = ov_props.cache_dir(str(cache_dir))
            config[key] = value
        core = ov.Core()
        self._compiled = core.compile_model(core.read_model(str(model_path)), device, config)
        self._input_name = self._compiled.input(0).get_any_name()
        self._output_name = self._compiled.output(0).get_any_name()
        self._class_names = class_names
        self._conf = conf_threshold
        self._iou = iou_threshold
        input_shape = self._compiled.input(0).partial_shape
        self._auto_letterbox = any(dim.is_dynamic for dim in input_shape)
        self._intrinsics = camera_intrinsics
        self._cam_position = camera_position
        self._cam_rotation = camera_rotation

    def detect(self, overhead_rgb: np.ndarray) -> list[Detection]:
        """Run detection on one (H, W, 3) uint8 RGB frame."""
        padded, gain, pad_x, pad_y = _letterbox(overhead_rgb, auto=self._auto_letterbox)
        tensor = np.ascontiguousarray(padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        result = self._compiled({self._input_name: tensor})
        pred = result[self._output_name][0].T  # (N, 4 + n_classes)
        class_ids = pred[:, 4:].argmax(1)
        confidences = pred[np.arange(len(pred)), class_ids + 4]
        mask = confidences > self._conf
        boxes_cxcywh = pred[:, :4][mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        if boxes_cxcywh.shape[0] == 0:
            return []
        cx, cy, bw, bh = boxes_cxcywh.T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        keep = _greedy_nms(boxes + class_ids[:, None] * CLASS_OFFSET, confidences, self._iou)
        boxes, confidences, class_ids = boxes[keep], confidences[keep], class_ids[keep]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, overhead_rgb.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, overhead_rgb.shape[0])
        return [
            Detection(
                label=self._class_names[int(cls)],
                xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(conf),
            )
            for box, cls, conf in zip(boxes, class_ids, confidences)
        ]

    def ground(self, snapshot: PerceptionSnapshot, target: str) -> ObjectPose3D | None:
        """3D grounding of `target` from the snapshot's detections + depth.

        Back-projects the best detection's box-center median depth (5x5
        kernel; invalid depth skips to the next-best detection) through the
        overhead camera geometry; the detector never sets held_by."""
        if self._intrinsics is None or self._cam_position is None or self._cam_rotation is None:
            raise EngineError("grounding requires camera geometry at construction")
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


def _grammar_schema(model: type) -> str:
    """Self-contained JSON schema for grammar-constrained decoding.

    Resolves the pydantic schema's $defs and requires every property, so the
    grammar forces the model to emit each field explicitly (null when not
    applicable) instead of silently dropping optional ones."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].split("/")[-1]])
        return {key: walk(value) for key, value in node.items() if key != "$defs"}

    schema = walk(schema)

    def require_all(node) -> None:
        if isinstance(node, list):
            for item in node:
                require_all(item)
        elif isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = sorted(node["properties"])
            for value in node.values():
                require_all(value)

    require_all(schema)
    return json.dumps(schema)


_PARSE_SYSTEM = (
    "You convert robot instructions into task-graph JSON. Every step must fill "
    "id, skill, arm, object, target, amount, parallel_group (null when not "
    "applicable). open_drawer, close_drawer, pick, place, handoff, hold, pour "
    "need object (plate, mug, bottle, spoon_1, spoon_2, fork_1, fork_2, "
    "drawer_top). place, handoff, pour need target (placemat_1, placemat_2, "
    "drawer_tray, hand_of_A, hand_of_B, mug). pour needs amount in (0, 1]. "
    "Example: 'Pick up the mug with arm B and put it on placemat 2' becomes "
    '{"task_id": "t1", "instruction": "Pick up the mug with arm B and put it '
    'on placemat 2", "steps": [{"id": 1, "skill": "pick", "arm": "B", '
    '"object": "mug", "target": null, "amount": null, "parallel_group": null}, '
    '{"id": 2, "skill": "place", "arm": "B", "object": "mug", "target": '
    '"placemat_2", "amount": null, "parallel_group": null}]}.'
)

_BOUNDARY_SYSTEM = (
    "You check robot task preconditions. Given the task, the current step, "
    "and the scene state, report whether the step's precondition holds. "
    "Reply with the report JSON only."
)

_DIAGNOSE_SYSTEM = (
    "You diagnose robot task failures. Given the task, the failed step, and "
    "the scene state, name the anomaly (one of object_moved, object_missing, "
    "object_dropped, drawer_jammed, grasp_lost, none, unknown) and an action "
    "(retry_skill, replan, abort). Reply with the diagnosis JSON only."
)


class OvGenaiVlmEngine:
    """openvino-genai VLM engine (§10.3 VlmEngine protocol).

    All generations are grammar-constrained to the call's JSON schema; the
    parse runs on a single-worker thread pool (the async seam) with one
    repair retry; boundary calls are blocking with a wall-clock generation
    cap. Images enter through the image provider wired by the runtime - the
    protocol signatures stay text-only by contract."""

    def __init__(
        self,
        model_dir: Path,
        device: str = "CPU",
        *,
        image_provider: Callable[[], np.ndarray | None] | None = None,
        parse_max_new_tokens: int = 350,
        boundary_max_new_tokens: int = 48,
        budget_s: float = VLM_BUDGET_S,
    ) -> None:
        self._pipe = genai.VLMPipeline(str(model_dir), device)
        self._tokenizer = self._pipe.get_tokenizer()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._image_provider = image_provider
        self._parse_tokens = parse_max_new_tokens
        self._boundary_tokens = boundary_max_new_tokens
        self._budget_s = budget_s
        self._future = None
        self._result: TaskGraph | None = None
        self._error: str | None = None
        self._warmup()

    def _warmup(self) -> None:
        """Compile the pipeline graphs once so no later call pays cold-start."""
        with self._lock:
            self._pipe.generate("hi", generation_config=genai.GenerationConfig(max_new_tokens=1))

    def submit_parse(self, instruction: str, summary: SceneSummary) -> None:
        """Begin an async parse of instruction into a TaskGraph."""
        self._error = None
        self._result = None
        self._future = self._executor.submit(self._parse_blocking, instruction, summary)

    @property
    def ready(self) -> bool:
        return self._future is None or self._future.done()

    def result(self) -> TaskGraph | None:
        """The finished TaskGraph, or None if not ready or parsing failed."""
        if self._future is None or not self._future.done():
            return None
        if self._result is None and self._error is None:
            try:
                self._result = self._future.result()
            except Exception as exc:  # noqa: BLE001 - surfaced through last_error
                self._error = f"parse crashed: {exc}"
        return self._result

    def last_error(self) -> str | None:
        return self._error

    def check_preconditions(
        self, graph: TaskGraph, current_step_id: int, summary: SceneSummary
    ) -> PreconditionReport:
        """Blocking precondition check; generation capped at the budget."""
        step = next((s for s in graph.steps if s.id == current_step_id), None)
        step_text = "unknown step"
        if step is not None:
            step_text = f"{step.skill} {step.object or ''} {step.target or ''} with arm {step.arm}"
        user = (
            f"Task: {graph.instruction}\nCurrent step: {step_text}\n"
            f"Scene: {summary.model_dump_json()}\n"
            "Does the precondition for the current step hold?"
        )
        text = self._generate(
            _BOUNDARY_SYSTEM, user, PreconditionReport, self._boundary_tokens, self._budget_s
        )
        try:
            return PreconditionReport.model_validate_json(text)
        except ValueError as exc:
            return PreconditionReport(ok=False, reason=f"vlm check failed: {exc}")

    def diagnose(
        self, graph: TaskGraph, failed_step_id: int, summary: SceneSummary
    ) -> VlmDiagnosis:
        """Blocking failure diagnosis; generation capped at the budget."""
        step = next((s for s in graph.steps if s.id == failed_step_id), None)
        step_text = "unknown step"
        if step is not None:
            step_text = f"{step.skill} {step.object or ''} with arm {step.arm}"
        user = (
            f"Task: {graph.instruction}\nFailed step: {step_text}\n"
            f"Scene: {summary.model_dump_json()}\nWhat went wrong?"
        )
        text = self._generate(
            _DIAGNOSE_SYSTEM, user, VlmDiagnosis, self._boundary_tokens, self._budget_s
        )
        try:
            return VlmDiagnosis.model_validate_json(text)
        except ValueError as exc:
            return VlmDiagnosis(
                anomaly="unknown",
                explanation=f"vlm diagnosis failed: {exc}",
                suggested_action="retry_skill",
            )

    def _parse_blocking(self, instruction: str, summary: SceneSummary) -> TaskGraph | None:
        user = (
            f"Scene: {summary.model_dump_json()}\n"
            f'Instruction: "{instruction}"\nReply with the task-graph JSON only.'
        )
        _grammar_schema(TaskGraph)
        text = self._generate(_PARSE_SYSTEM, user, TaskGraph, self._parse_tokens)
        try:
            return TaskGraph.model_validate_json(text)
        except ValueError as exc:
            error_text = str(exc)
            logger.warning("parse validation failed once: %s", error_text)
        repair = (
            f"{user}\nThe last reply was invalid ({error_text}). Every skill "
            "except home and retract needs its object field filled."
        )
        text = self._generate(_PARSE_SYSTEM, repair, TaskGraph, self._parse_tokens)
        try:
            return TaskGraph.model_validate_json(text)
        except ValueError as exc:
            self._error = f"parse failed after repair: {exc}"
            return None

    def _generate(
        self,
        system: str,
        user: str,
        output_model: type,
        max_new_tokens: int,
        budget_s: float | None = None,
    ) -> str:
        """One grammar-constrained generation; `budget_s` caps wall clock (boundary
        calls only - the async parse is bounded by max_new_tokens alone)."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        config = genai.GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            structured_output_config=genai.StructuredOutputConfig(
                json_schema=_grammar_schema(output_model)
            ),
        )
        images: list = []
        frame = self._image_provider() if self._image_provider is not None else None
        if frame is not None:
            scale = min(VLM_MAX_IMAGE / frame.shape[0], VLM_MAX_IMAGE / frame.shape[1], 1.0)
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            images = [ov.Tensor(np.ascontiguousarray(frame))]
        streamer = None
        if budget_s is not None:
            deadline = time.perf_counter() + budget_s

            def budget_streamer(_chunk: str) -> bool:
                return time.perf_counter() > deadline

            streamer = budget_streamer
        with self._lock:
            output = self._pipe.generate(
                prompt, images=images, generation_config=config, streamer=streamer
            )
        return output.text if hasattr(output, "text") else str(output)


@dataclass(frozen=True)
class EngineBundle:
    """The perception + reasoning engines for one device map."""

    detector: OpenvinoDetector
    vlm: OvGenaiVlmEngine
    device_map: dict[str, str]


def _onnx_has_dynamic_input(path: Path) -> bool:
    """True when the cached ONNX still carries its dynamic spatial input."""
    model = ov.Core().read_model(str(path))
    return any(dim.is_dynamic for dim in model.input(0).partial_shape)


def fetch_standin_yolo(cache_dir: Path | None = None) -> tuple[Path, list[str]]:
    """Export the pretrained yolo11n ONNX into a cache dir; (path, class names).

    ultralytics/torch is imported only here - the deployed detector itself is
    OpenVINO-only. A cached export with a static input (e.g. written by a plain
    `yolo export` without dynamic=True) is re-exported: the rect letterbox and
    the torch-equivalence gate both depend on the dynamic spatial axes."""
    cache = Path(cache_dir or Path("artifacts") / "standin")
    onnx_path = cache / f"{STANDIN_YOLO_TAG}.onnx"
    names_path = cache / f"{STANDIN_YOLO_TAG}_names.json"
    if onnx_path.is_file() and names_path.is_file() and _onnx_has_dynamic_input(onnx_path):
        return onnx_path, json.loads(names_path.read_text(encoding="utf-8"))
    from ultralytics import YOLO

    cache.mkdir(parents=True, exist_ok=True)
    model = YOLO(f"{STANDIN_YOLO_TAG}.pt")
    # dynamic input so the detector can use the rect letterbox that mirrors
    # the torch predict path exactly (the equivalence gate depends on it)
    exported = Path(model.export(format="onnx", imgsz=LETTERBOX_SIZE, dynamic=True, opset=12))
    if exported != onnx_path:
        onnx_path.unlink(missing_ok=True)
        exported.rename(onnx_path)
        Path(f"{STANDIN_YOLO_TAG}.pt").replace(cache / f"{STANDIN_YOLO_TAG}.pt")
    names = [model.names[i] for i in sorted(model.names)]
    names_path.write_text(json.dumps(names), encoding="utf-8")
    return onnx_path, names


def fetch_standin_vlm() -> Path:
    """Local path of the public VLM stand-in (downloaded + tokenizer-fixed)."""
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(STANDIN_VLM_REPO))
    return ensure_genai_tokenizers(model_dir)


def load_bundle(
    device_map: dict[str, str],
    *,
    yolo_model: Path | None = None,
    yolo_class_names: list[str] | None = None,
    vlm_model_dir: Path | None = None,
    cache_dir: Path | None = None,
    image_provider: Callable[[], np.ndarray | None] | None = None,
) -> EngineBundle:
    """Construct the detector + VLM engines per a device map.

    Stand-in models are fetched when no explicit paths are given; project
    artifacts replace them without code changes after the merge checkpoints."""
    if yolo_model is None or yolo_class_names is None:
        yolo_model, yolo_class_names = fetch_standin_yolo(cache_dir)
    if vlm_model_dir is None:
        vlm_model_dir = fetch_standin_vlm()
    detector = OpenvinoDetector(
        yolo_model, yolo_class_names, device_map.get("yolo", "CPU"), cache_dir=cache_dir
    )
    vlm = OvGenaiVlmEngine(
        vlm_model_dir, device_map.get("vlm", "CPU"), image_provider=image_provider
    )
    return EngineBundle(detector=detector, vlm=vlm, device_map=dict(device_map))
