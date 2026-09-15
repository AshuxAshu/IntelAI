# OpenVINO optimization record

Every optimization is a recorded experiment, never a blind flag. The protocol: a
golden baseline (FP32 closed-loop on 10 fixed seeds) is measured first, then each
candidate runs the identical seeds and metrics, and the non-negotiable quality
gates decide the verdict. A technique must also buy at least 15% latency or
throughput to be worth its risk.

Quality gates: closed-loop success delta >= -3 pp; placement error delta <= +1 cm;
VLM QA delta >= -2 pp; JSON validity stays 100%. When a gate fails, the recovery
ladder is FP16, then sensitivity-ranked mixed precision, then QAT; if the gate
still fails, the aggressive configuration is rejected - the gate outranks the
speedup.

## Experiment verdicts

One row per optimization experiment. `latency delta` and `quality delta` are
measured against the golden baseline on identical seeds; `verdict` is keep or
reject.

| technique | device | latency delta | quality delta | verdict |
| --- | --- | --- | --- | --- |

## Technique inventory

Applied techniques are checked and reference their verdict rows above.

### Tier A - applied by default (P0)

- [ ] A1 NNCF INT8 PTQ calibrated on our sim data (ACT, YOLO, VLM)
- [ ] A2 Preprocessing as manifest components, in-graph fusion as optimization (all)
- [ ] A3 Fixed-shape, per-consumer inputs (ACT 128 px, YOLO 640 px, VLM 512 px)
- [ ] A4 Model caching via cache_dir (all)
- [ ] A5 performance_mode=LATENCY, single stream (ACT, YOLO)
- [ ] A6 physicalai AsyncExecution + ChunkedActionQueue vs SyncExecution (ACT)
- [ ] A7 openvino-genai VLMPipeline instead of naive transformers (VLM)
- [ ] A8 Zero-copy ov.Tensor over numpy render buffers (all)
- [ ] A9 Event-triggered perception (YOLO at 5-10 Hz or on demand)
- [ ] A10 Hybrid-core pinning (physics + executor on P-cores, telemetry on E-cores)
- [ ] A11 Reuse of physicalai InferenceLatencyBenchmark + RandomInputSource (bench)

### Tier B - heterogeneous and advanced (P1)

- [ ] B1 NPUW partitioned compilation, NPUW:CPU,NPU / NPUW:GPU,NPU (VLM)
- [ ] B2 HETERO:GPU,CPU documented alternative (VLM)
- [ ] B3 Latency-hiding default skills during VLM boundary calls (system)
- [ ] B4 Sensitivity-ranked mixed precision (ACT, VLM)
- [ ] B5 execution_mode=ACCURACY on GPU where fp16 rounding hurts (ACT)
- [ ] B6 JSON-grammar-constrained decoding (VLM)
- [ ] B7 Prefix-stable prompts + bounded max_new_tokens + downscaled VLM input (VLM)
- [ ] B8 enable_profiling per-layer hot-spot audit (all)

### Tier C - gated experiments / stretch (P1-P2)

- [ ] C1 NNCF QAT for ACT (gate: INT8 PTQ fails the closed-loop parity gate)
- [ ] C2 INT8/INT4 weight compression (VLM; gate: VLM QA parity)
- [ ] C3 Speculative decoding with a 135M draft model (VLM; lossless for greedy)
- [ ] C4 Two power profiles via powerprofilesctl (benchmark: perf and perf-per-watt)
