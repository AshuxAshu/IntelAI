# Bimanual VLA Table-Setting — Intel Physical AI Online Challenge

[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![mujoco](https://img.shields.io/badge/MuJoCo-3.2-orange)](https://mujoco.org/)
[![openvino](https://img.shields.io/badge/OpenVINO-2026.3-8a2be2)](https://docs.openvino.ai/)
[![assets](https://img.shields.io/badge/assets-license--tracked-green)](assets/ASSETS_LICENSES.md)

**What this project is.** An end-to-end Physical AI solution for bimanual robotic
manipulation in simulation. Two official **SO-101** arms in **MuJoCo** interpret a
natural-language instruction, reason over camera observations, coordinate both
manipulators, and set a dinner table — opening the cutlery drawer, retrieving
forks and spoons, placing the plate and mug at their settings, handing the water
bottle between arms, and pouring into the mug held by the other arm. The stack
runs the full observe → understand → plan → act → optimize loop on Intel
hardware, with every learned component exported to **OpenVINO IR** and mapped
across the **CPU / Arc iGPU / NPU** of an Intel Core Ultra system.

It is built for the Intel Physical AI Online Challenge: *Bimanual VLA Manipulation
with Multi-Modal Reasoning*, and delivers all five required artifacts — a
reproducible repository, a reproducible MuJoCo simulation, an Intel inference
benchmark script, a 10-seed demonstration, and this technical architecture
summary.

**The one-line architecture.** A *temporally-decoupled VLA*: a slow vision-language
"brain" (~1 Hz, NPU/iGPU) turns the instruction and the scene into a
schema-validated JSON task graph; a deterministic executor schedules that graph
onto the two arms with precondition gates and recovery; and a fast learned
bimanual ACT action expert (~25 Hz, iGPU/CPU) turns it into 12 joint targets.
The simulator doubles as a data factory, generating perfectly-labeled
demonstrations, bounding boxes, and VQA pairs for free.

---

## Tech stack

| Layer | Technology | Role in this project |
| --- | --- | --- |
| Simulation | **MuJoCo 3.2** (`MjSpec` programmatic scenes, EGL offscreen render) | Dual SO-101 physics, cameras, domain randomization |
| Robot model | **Official SO-101 MJCF** (MuJoCo Menagerie, Apache-2.0) | 5-DOF + gripper arms, STS-3215 actuator gains |
| Data format | **LeRobot 0.5.1** | Episode datasets (mp4 + parquet + meta) for imitation training |
| Policy training | **Intel Physical AI Studio** (`physicalai-train`) | Native ACT training on Lightning, XPU acceleration, OpenVINO export |
| Deployment runtime | **Intel `physicalai` runtime** | `RobotRuntime` control loop, `InferenceModel`, async + chunked actions |
| Inference optimization | **OpenVINO ≥ 2026.3**, **NNCF ≥ 3.0**, **openvino-genai** | IR export, FP16/INT8 precision ladder, NPUW heterogeneous compilation |
| Action policy | **ACT** (bimanual, skill-conditioned, ~20 M params) | Imitation-learned closed-loop visuomotor control at 25 Hz |
| Object grounding | **YOLO11n** (Ultralytics) | Object + drawer-state detection on synthetic labels |
| Reasoner | **Qwen3-VL-2B-Instruct** + LoRA SFT | Instruction → JSON task graph, grounded scene QA, re-planning |
| Optimization platform | **Intel Physical AI Studio** | Train → export → quantize → deploy in one toolchain |
| Environment | `uv` + Python 3.12, Docker, GitHub Actions (CPU CI) | Pinned, reproducible, judge-ready |

<table align="center">
<tr>
<td align="center"><a href="https://cdn-uploads.huggingface.co/production/uploads/677ac3710c9718b04aac4c1f/VWzTBcjyuXxMxypIc2gW1.png"><img src="assets/readme/tech_openvino.png" width="256" height="75" alt="Intel OpenVINO"></a><br><b>Intel OpenVINO</b></td>
<td align="center"><a href="https://github.com/qwenlm/qwen3-vl"><img src="assets/readme/tech_qwen3vl.png" width="256" height="75" alt="Qwen3-VL"></a><br><b>Qwen3-VL</b></td>
<td align="center"><a href="https://github.com/open-edge-platform/physical-ai-studio"><img src="assets/readme/tech_physical_ai_studio.png" width="256" height="75" alt="Intel Physical AI Studio"></a><br><b>Intel Physical AI Studio</b></td>
</tr>
</table>

---

## OpenVINO benchmark results

ACT policy inference, one checkpoint exported through the full precision ladder and
timed with Intel's own `InferenceLatencyBenchmark` + `RandomInputSource`
methodology — the same harness `make bench` uses, so both report p50/p95 latency,
throughput and device selection on the same basis. The final column is how far
each variant's actions drift from the FP32 reference on identical inputs — the
quantization-quality check.

| ACT policy | CPU | Arc iGPU | NPU | Action error vs FP32 |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 35.77 ms | – | – | 1.890e-07 |
| OpenVINO FP32 | 23.31 ms | 30.41 ms | – | reference |
| OpenVINO FP16 | 24.64 ms | 51.55 ms | – | 2.130e-04 |
| OpenVINO INT8 (NNCF) | 7.97 ms | 18.20 ms | – | 7.432e-03 |
| OpenVINO INT8 weights | 20.09 ms | 22.57 ms | – | 3.072e-03 |

A dash means the device is not present on the measuring host — it is shown
explicitly rather than dropped. All rows use fixed-shape, per-consumer inputs
(ACT 224 px, YOLO 640 px, VLM 512 px) and zero-copy `ov.Tensor` views over the
renderer's buffers. The INT8 (NNCF) rung is the latency winner on both devices at
7.97 ms CPU / 18.20 ms iGPU, a **2.9× CPU** and **1.7× iGPU** speedup over the
OpenVINO FP32 reference, while staying inside the action-parity gate
(normalized MSE 4.17e-05 < 5e-03).

> **NPU note.** NPU tests will be updated here soon, as soon as our Core Series VM
> request gets approved on Intel's cloud. The NPU column is wired end-to-end
> today: `scripts/npu_preflight.py` compiles every model for the `NPU` device and
> lists unsupported ops, and the device map already carries the Ultra profile
> (`vlm: "NPUW:CPU,NPU"`), so switching the column on is a validation sprint
> rather than a port. On the current dev host the NPU is absent and the column
> prints a dash by design.

Reproduce the whole matrix in one command:

```bash
make bench-ov                        # or: uv run python scripts/benchmark_openvino.py
```

It trains a stand-in checkpoint when none is given, exports every precision rung,
times each on CPU/iGPU/NPU, gates each rung's actions against FP32, and writes
`bench/ov_matrix/` as markdown + CSV + JSON.

---

## Architecture

### Tier decomposition

The system is split at the skill boundary so that each tier can be mapped onto the
right accelerator and the right control rate. This is what makes the Intel story
work: temporal decomposition *is* hardware decomposition.

```
+--------------------------------------------------------------------------+
| TIER 3 - REASON   (runs at skill boundaries, ~1 Hz)                      |
|  Qwen3-VL-2B-Instruct, LoRA-fine-tuned on synthetic sim VQA              |
|  - NL instruction      -> JSON task graph (schema-constrained decoding)  |
|  - Grounded scene QA   (what's where? is precondition X true?)           |
|  - Anomaly explanation + re-plan on scene-state mismatch                 |
|  - Audio input (P1): Speechmatics RT STT (+ local Whisper fallback)      |
|  Deployed: OpenVINO GenAI, INT8 weights -> NPU (Ultra 2/3) / iGPU (i7)   |
+--------------------------------------------------------------------------+
| TIER 2 - PLAN / EXECUTE   (deterministic, in-process, us-scale)          |
|  - Pydantic-validated task graph -> skill scheduler                     |
|  - Arm assignment via workspace reasoning (reassign if unreachable)     |
|  - Pre/postcondition gates between skills; retry & recovery policies    |
|  - Parallel dual-arm skills: pour+hold, hand-off, coordinated reach     |
+--------------------------------------------------------------------------+
| TIER 1 - ACT   (closed-loop visuomotor, 25 Hz)                           |
|  - ONE skill-conditioned bimanual ACT policy (~20 M params)             |
|  - Obs: 2 wrist cams + overhead cam (128 px RGB) + proprio + goal       |
|    vector derived from PERCEPTION, never privileged sim state           |
|  - Act: 12 joint position targets (2x5 DOF + 2 grippers), chunk = 25     |
|  - Trained by imitation on procedurally-generated teacher demos + DR    |
|  Deployed: Physical AI Studio OpenVINO export (FP16/INT8 IR + manifest)  |
|            -> iGPU/CPU/NPU via Intel physicalai runtime                 |
+--------------------------------------------------------------------------+
| TIER 0 - PERCEIVE + SIMULATE                                            |
|  - YOLO11n object grounding (trained on free synthetic sim labels)       |
|  - Overhead depth render -> 3D pose; multi-view + temporal fusion        |
|  - MuJoCo 3.2: dual SO-101, tabletop cutlery caddy, utensils,           |
|    plate/mug/bottle, water fill-level proxy, domain randomization       |
+--------------------------------------------------------------------------+
```

### Hardware mapping

Device assignment is **configuration, not code** (`configs/runtime/device_map.yaml`),
which is what makes the i7 → Core Ultra migration a YAML edit:

| Tier | Model | i7-13620H (dev) | Core Ultra 2/3 | Precision |
| --- | --- | --- | --- | --- |
| Reason (~1 Hz) | Qwen3-VL-2B + LoRA | iGPU | **NPU** (`NPUW:CPU,NPU`) | INT8 weights, FP32 activations |
| Execute (us) | Task-graph executor | CPU (P-core) | CPU (P-core) | — |
| Act (25 Hz) | Bimanual ACT | iGPU (CPU fallback) | iGPU | FP16 / INT8 |
| Perceive | YOLO11n | iGPU | iGPU | INT8 |
| Simulate | MuJoCo physics | CPU (P-core pinned) | CPU (P-core pinned) | FP32 |

`NPUW:CPU,NPU` lets the transformer run on the NPU while unsupported ops fall back
to the CPU within a single compiled model — the heterogeneous-architecture story
the mentors asked for.

### Repository layout

```
src/dinner_table/
  config.py                 pydantic-settings; loads YAML, validates cross-references
  compat/                   thin shims over fast-moving APIs (physicalai, lerobot, nncf)
  scene/                    builder, object catalog + spawn sampler, randomizer,
                            camera rig, contact sheet, water fill-level proxy
  teacher/                  PRIVILEGED oracle: kinematics, damped-least-squares IK,
                            planner, grasp catalog, coroutine skills, graph runner
  data/                     demo generator, Perturber (noise), LeRobot export,
                            YOLO label factory, VQA factory, seed-disjoint splits
  policies/                 conditioning vector, Studio training wrapper,
                            OpenVINO export + NNCF INT8 PTQ + parity checks
  perception/               YOLO train path vs OpenVINO inference path,
                            depth fusion, temporal tracker
  reasoning/                task-graph schema (VLM <-> executor contract), prompts,
                            Qwen3-VL LoRA SFT, openvino-genai runtime, fallback parser
  executor/                 graph executor, scheduler + zone claims, precondition
                            gates, workspace reachability maps, recovery matrix
  runtime/                  MuJoCo Robot + Camera adapters, skill action source,
                            OpenVINO engines, telemetry, HUD, Gradio app, Speechmatics
  bench/                    bench_intel.py, device probe, telemetry sampler, reporting
  eval/                     seed-sweep harness, failure taxonomy, robustness matrix
```

Isolation is enforced by `tests/test_isolation.py`: the privileged teacher and any
simulator ground truth may never be imported by the deployed runtime, executor, or
perception/reasoning inference paths. The learned policy sees only cameras,
proprioception, and a perception-derived goal vector.

### Task-graph executor

The VLM emits a pydantic-validated graph; the executor owns sequencing — the
hardest part of an end-to-end VLA:

```json
{
  "task_id": "dinner_v3",
  "steps": [
    { "id": 1, "skill": "open_drawer", "arm": "A", "object": "drawer_top", "next": 2 },
    { "id": 2, "skill": "pick", "arm": "A", "object": "plate",
      "then": { "skill": "place", "target": "placemat_1", "next": 3 } },
    { "id": 3, "skill": "pick", "arm": "B", "object": "mug",
      "then": { "skill": "hold", "parallel_with": 4 } },
    { "id": 4, "skill": "pour", "arm": "A", "object": "bottle",
      "target": "mug", "amount": 0.6 }
  ],
  "success_conditions": {
    "plate": "on placemat_1",
    "mug_water_level": ">0.5",
    "utensils": "at settings"
  }
}
```

Between steps the executor enforces pre/postconditions, retries with recovery
(re-grasp, re-perceive, re-plan, drawer wiggle), and reassigns arms by workspace
reachability. Recovery depth is bounded at 3 re-plans.

### Bimanual skills

Each skill is one conditioning token for ACT: `home`, `open_drawer`,
`close_drawer`, `pick`, `place`, `handoff`, `hold`, `pour`, `retract`. The
demonstration matrix covers every dual-arm rubric line:

| Dual-arm behaviour | How it is demonstrated |
| --- | --- |
| Hand-off | Arm B lifts the bottle and passes it to arm A in the shared zone |
| Complementary action | Arm B holds the mug steady while arm A tilts the bottle and pours |
| Coordinated concurrent action | Both arms retrieve different utensils from the open drawer at once |
| Shared-workspace reasoning | Arm reassignment when an object spawns only in the other arm's reach |

ACT is conditioned structurally, not textually: a 35-dim vector is packed into
`observation.state` (10 arm joints, 2 gripper apertures, active-arm one-hot,
9-skill one-hot, 9-object one-hot, goal xyz), so the native ACT policy — which
takes no task text — is skill- and goal-aware by construction. Placement targets
are resolved through a frozen `RelativeTarget` vocabulary shared by the teacher
and the deployed runtime, so "put a fork beside the plate" resolves identically in
training and at inference.

---

## Training approach

Training uses **imitation learning from a privileged teacher**, with the simulator
acting as an unlimited, perfectly-labeled data factory — no teleoperation and no
manual annotation anywhere in the pipeline.

**1. Scene and task definition.** Two official SO-101 arms are mounted side by
side on the operator-facing table edge; the scene includes a tabletop cutlery
caddy with a prismatic drawer, mass-accurate plate/mug/bottle/utensils, two
placemats as placement targets, and an overhead + two wrist cameras. Scene
geometry, object masses, and contact parameters are the measured,
reference-proven values for the real SO-101 (see `docs/PLAN_AMENDMENTS.md`).

**2. Privileged teacher.** A damped-least-squares IK solver, a corridor planner
with arm–arm keep-outs, a per-object grasp catalog (plate rim, mug wall, bottle
neck side-grasp, cutlery pinch), and coroutine skill state machines produce
validated multi-step demonstrations at 25 Hz. A `Perturber` injects kicks, slips,
and jitter so the demonstrations contain the recovery behaviour the policy must
learn, and the teacher's task-graph runner verifies each step's postcondition
before advancing.

**3. Domain randomization during generation.** DR is baked into data generation
rather than bolted on afterwards: object mass ×0.5–2.0, friction 0.4–1.2, spawn
pose ±6 cm / ±25°, light intensity and color temperature, and texture/background
variation — each axis a seeded mutation, so eval extremes (±10 cm, ×0.3–3.0 mass,
unseen textures) are strictly outside the training hull.

**4. ACT training via Intel Physical AI Studio.** Native
`physicalai.policies.ACT` (PyTorch Lightning) is trained on the exported LeRobot
dataset through Intel's own CLI. Chunk size and action horizon are 25 (1 s at
25 Hz), with three camera streams and skill conditioning packed into the state
vector. A GPU A/B gate compares native ACT, the LeRobot-wrapper ACT, and native
SmolVLA (language-conditioned) on closed-loop success and exported latency; native
ACT wins on edge latency and SmolVLA survives as the documented "true VLA"
ablation.

**5. Perception and reasoning training.** YOLO11n is trained on synthetic labels
derived from ground-truth geometry (bounding boxes are free from the simulator),
then exported ONNX → OpenVINO. Qwen3-VL-2B receives a LoRA SFT (r=16, vision tower
frozen) on the VQA corpus generated from episode logs, and is gated against a
zero-shot Qwen3-VL-4B baseline on held-out instruction parsing and scene QA.

**6. Diagnostics loop.** Every run logs to Weights & Biases (TensorBoard
fallback): action-MSE/L1/CVAE losses, per-skill closed-loop success, a
failure-stage taxonomy (phase × cause: perception miss, conditioning confusion,
control drift, physics slip), a chunk-position error curve, a
skill-conditioning confusion matrix, and an OOD matrix over DR axes. The dominant
failure mode drives targeted counter-data generation, then retrain and re-eval —
the iteration is recorded symptom → diagnosis → data fix → result in
`docs/TRAINING_DIARY.md`.

### How we leveraged Intel's Physical AI Studio

Physical AI Studio is not just the trainer — it is the spine of the train → export
→ deploy chain, which is why the whole pipeline is Intel-native end to end:

| Studio capability | How this project uses it |
| --- | --- |
| `physicalai.policies.ACT` | The primary bimanual action policy, trained natively (no re-implementation) |
| `LeRobotDataModule` with `data_format="physicalai"` | Loads our local LeRobot dataset directly — no Hub round-trip |
| `physicalai fit --config` CLI | One identical command string on the laptop, in CI, and on the cloud GPU |
| Intel XPU accelerator (`accelerator: xpu`, `xpu_single`) | A duplicate run trained on the Intel iGPU, strengthening the deployment story |
| `policy.export(backend="openvino")` | Produces `act.xml/.bin + manifest.json` (FP16 IR) with letterbox-resize, stats normalization, and action-chunk trimming declared as manifest components |
| Opt-in INT8 weight-compression post-export hook | One rung of the precision ladder (`physicalai-train[nncf]`), re-saved over the same manifest |
| The `physicalai` runtime (`InferenceModel`, `RobotRuntime`) | Loads the Studio export and executes it in a fixed-rate loop, with async execution and chunked action queues for free |
| Universal configs | `configs/physicalai/act_dinner.yaml` (CUDA) and `act_dinner_xpu.yaml` (Intel XPU) |

Our MuJoCo simulation plugs into Intel's runtime as a first-class citizen: the
`Robot` protocol and `Camera` ABC are explicitly implementable by a simulator, so
`MuJoCoBimanualRobot`, `MuJoCoCamera`, and `SkillExecutorSource` join the same
`RobotRuntime` that would drive real hardware. `tests/test_studio_roundtrip.py`
pins the Studio → runtime contract in CI on CPU: smoke-train, export, load, and
assert action parity against the Torch checkpoint.

---

## Quick start

Everything is one command. The mentor-facing path is container-first so no
dependency resolution is needed on your side; the native path is for development.

### Option A — Docker (recommended for evaluation)

```bash
git clone <repo-url> && cd IntelAI
docker build -t dinner-table .
docker run --rm -it --device /dev/dri dinner-table \
  bash -lc "make smoke && make scene SEED=42"
```

The image installs the full pinned environment (`uv sync --frozen`, CPU + dev
extras) including MuJoCo, OpenVINO, and both `physicalai` packages, so the judges
can reproduce the simulation and the benchmark without touching the dependency
graph. `--device /dev/dri` exposes the Intel iGPU for the GPU inference path; drop
it to run CPU-only.

### Option B — native (`uv`)

```bash
uv sync                       # installs Python 3.12 + the pinned lockfile
make smoke                    # headless MuJoCo step + PNG (Phase 0 gate)
```

### The commands that matter

| Command | What it does |
| --- | --- |
| `make scene SEED=42` | Randomized scene, multi-view contact sheet |
| `make teacher` | Per-skill × DR success matrix (teacher oracle) |
| `make dataset` | Teacher demos → LeRobot dataset + YOLO labels + VQA corpora |
| `make train MODEL=act` | `physicalai fit` on the local LeRobot dataset; logs to W&B, checkpoints to HF Hub |
| `make eval-policy` | Closed-loop policy skill eval + failure taxonomy |
| `make eval SEEDS=0-9` | Full-stack 10-seed campaign → `docs/RESULTS.md` |
| `make bench DEVICES=auto` | Deliverable 3: latency, throughput, device and precision per model |
| `make bench-ov` | The OpenVINO precision matrix in one command |
| `make run` | `physicalai run --config configs/runtime/dinner_demo.yaml` |
| `make demo COMMAND="..." SEED=3` | One HUD-overlaid episode |

The canonical instruction from the problem statement drives the flagship graph:

```bash
make demo COMMAND="Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A." SEED=3
```

### Reproduce the 10-seed demonstration (deliverable 4)

```bash
uv run python scripts/demo/render_demo_videos.py --seeds 0-9 --ov-device GPU
uv run python scripts/demo/probe_runs.py --seeds 0-9        # outcomes, no video (fast)
```

Output lands in `artifacts/demo/videos/` with a `manifest.json` recording every
step's outcome, frame count, sim time, and the live OpenVINO statistics observed
during that episode. Failures are marked on camera, never hidden.

### What "working" looks like

- `make eval SEEDS=0-9` reports the full-stack campaign: **≥ 70% first-attempt**
  task success and **≥ 90% with the recovery ladder enabled**, with per-skill
  success and timing in the report.
- `make bench-ov` prints the precision matrix above and writes
  `bench/ov_matrix/`; every rung passes its action-parity gate.
- The demonstration video shows the command, the randomized scene, the perception
  overlay, coordinated dual-arm action including the hand-off and pour, and the
  final table state.

---

## Verification and reproducibility

The suite runs the real thing almost everywhere — the only mocks are network
access (artifacts snapshotted) and `time.sleep` in watchdog tests. Real MuJoCo
physics, real OpenVINO CPU inference on every run, real iGPU where present, and
conditional-skip with an explicit reason only for absent hardware.

```bash
uv run pytest tests -m "fast or slow"        # what CI runs on every push
uv run pytest -m nightly                     # training smokes, parity gates, recovery
uv run pytest tests/bench/test_ov_matrix.py -m fast   # OpenVINO plumbing, no benchmark run
```

Key gates: `test_ov_parity` (OpenVINO FP32 vs PyTorch: ACT rel. error < 1e-3),
`test_quant_parity` (ACT action-MSE < 5e-3 normalized), `test_quant_success_parity`
(closed-loop success Δ ≥ −3 pp), `test_policy_no_privilege` (no simulator ground
truth leaks into policy inputs), `test_e2e_10seeds`, `test_recovery_suite`,
`test_studio_roundtrip`, and `test_readme_commands` (every fenced command block in
this README is executed as a doc-test).

Determinism: pinned seeds for numpy/torch/random/MuJoCo, `OMP_NUM_THREADS=1` in
eval mode, and golden-file hashes for the scene, dataset samples, and FP32
inference outputs.

### Optimization protocol

Every optimization is a recorded experiment, never a blind flag. A golden FP32
baseline is measured first, each candidate runs identical seeds and metrics, and
quality gates decide the verdict: closed-loop success Δ ≥ −3 pp, placement error
Δ ≤ +1 cm, VLM QA Δ ≥ −2 pp, JSON validity 100%. A technique must also buy at
least 15% latency or throughput to be worth its risk; the recovery ladder is FP16,
then sensitivity-ranked mixed precision, then QAT, and if the gate still fails the
aggressive configuration is rejected — the gate outranks the speedup. Verdicts are
appended to `docs/OPTIMIZATION.md`.

---

## Robustness and evaluation campaign

`make eval SEEDS=0-9` sweeps ten fixed seeds through the full stack and emits
per-skill success, failure taxonomy, and a DR-axis OOD matrix. The robustness
matrix additionally exercises extreme physical DR (mass, friction, placement),
visual DR (unseen textures and backgrounds, relighting), paraphrase variants of
the instruction, and mid-episode perturbations (object nudged mid-carry, object
knocked from the gripper, drawer friction jam, instruction swapped mid-stream,
nonexistent-object refusal).

---

## Licenses and attribution

Third-party assets are recorded one row per file in
`assets/ASSETS_LICENSES.md` and gated by `scripts/check_licenses.py`, which fails
if any asset lacks a license row. The official SO-101 MJCF and meshes are
Apache-2.0 (MuJoCo Menagerie) and vendored with provenance. The reference
approach in `example-approach/` (MIT) informed the scene geometry and grasp
recipes; what was adopted and what was deliberately not adopted is documented in
`docs/PLAN_AMENDMENTS.md`.

## Documentation map

| Document | Contents |
| --- | --- |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Full end-to-end specification and phase plan |
| [`docs/PLAN_AMENDMENTS.md`](docs/PLAN_AMENDMENTS.md) | Measured deviations from the plan, with evidence |
| [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) | Optimization protocol and per-experiment verdicts |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Reproducing the OpenVINO matrix and reading the table |
| [`docs/TRAINING_DIARY.md`](docs/TRAINING_DIARY.md) | Per-iteration training diagnosis log |
| [`scripts/demo/README.md`](scripts/demo/README.md) | How the per-seed demonstration videos are built |
