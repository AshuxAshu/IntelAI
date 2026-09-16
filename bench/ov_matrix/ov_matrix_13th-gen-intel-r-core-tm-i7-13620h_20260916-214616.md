# OpenVINO optimization matrix

- host cpu: 13th Gen Intel(R) Core(TM) i7-13620H
- os: Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44
- openvino 2026.1.0 | physicalai 0.1.1 | nncf 3.3.0
- devices: CPU, GPU.0, GPU.1
- checkpoint: artifacts/act_public/experiments/lightning_logs/version_0/checkpoints/epoch=0-step=50.ckpt

| ACT policy | CPU | iGPU | NPU | Action error vs FP32 |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 38.60 ms | – | – | 1.890e-07 |
| OpenVINO FP32 | 24.43 ms | 46.42 ms | – | reference |
| OpenVINO FP16 | 30.03 ms | 51.34 ms | – | 2.130e-04 |
| OpenVINO INT8 (NNCF) | 10.27 ms | 35.14 ms | – | 7.432e-03 |
| OpenVINO INT8 weights | 23.29 ms | 83.85 ms | – | 3.072e-03 |

## Detail

| row | device | p50 ms | p95 ms | throughput ips | cold compile s | model MB | rss MB | weights | acts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | CPU | 38.599 | 45.332 | 25.87 | 0.6 | – | 26.0 | – | – |
| PyTorch FP32 | iGPU | – | – | – | – | – | – | – | – |
| PyTorch FP32 | NPU | – | – | – | – | – | – | – | – |
| OpenVINO FP32 | CPU | 24.429 | 32.76 | 40.38 | 0.27 | 130.8 | 288.1 | FP32 | FP32 |
| OpenVINO FP32 | iGPU | 46.423 | 113.596 | 20.98 | 0.87 | 130.8 | 160.8 | FP32 | FP32 |
| OpenVINO FP32 | NPU | – | – | – | – | 130.8 | – | FP32 | FP32 |
| OpenVINO FP16 | CPU | 30.028 | 37.554 | 32.8 | 0.3 | 65.7 | 226.2 | FP16 | FP32 |
| OpenVINO FP16 | iGPU | 51.337 | 92.241 | 19.19 | 0.76 | 65.7 | 26.0 | FP16 | FP32 |
| OpenVINO FP16 | NPU | – | – | – | – | 65.7 | – | FP16 | FP32 |
| OpenVINO INT8 (NNCF) | CPU | 10.265 | 14.074 | 94.95 | 0.53 | 37.4 | 106.0 | INT8 | FP32 |
| OpenVINO INT8 (NNCF) | iGPU | 35.136 | 48.818 | 28.52 | 1.01 | 37.4 | 38.3 | INT8 | FP32 |
| OpenVINO INT8 (NNCF) | NPU | – | – | – | – | 37.4 | – | INT8 | FP32 |
| OpenVINO INT8 weights | CPU | 23.292 | 29.098 | 42.77 | 0.39 | 33.5 | 192.2 | UINT8 | FP32 |
| OpenVINO INT8 weights | iGPU | 83.85 | 95.941 | 12.55 | 0.94 | 33.5 | 24.9 | UINT8 | FP32 |
| OpenVINO INT8 weights | NPU | – | – | – | – | 33.5 | – | UINT8 | FP32 |

## Parity gates

| row | rel err vs FP32 | norm MSE vs FP32 | rel err vs PyTorch | verdict |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 1.890e-07 | 2.080e-14 | – | n/a (baseline) |
| OpenVINO FP32 | 0.000e+00 | 0.000e+00 | 1.890e-07 | reference |
| OpenVINO FP16 | 2.130e-04 | 3.740e-08 | 2.130e-04 | pass (rel err 2.13e-04 < 0.005) |
| OpenVINO INT8 (NNCF) | 7.432e-03 | 4.167e-05 | 7.432e-03 | pass (norm MSE 4.17e-05 < 0.005) |
| OpenVINO INT8 weights | 3.072e-03 | 3.004e-06 | 3.072e-03 | pass (norm MSE 3.00e-06 < 0.005) |

## Notes

- NPU is not available on this host
- the action-parity column is each rung against the OpenVINO FP32 reference on identical inputs; it is the quantization-quality check, not a task-success rate
- checkpoint: artifacts/act_public/experiments/lightning_logs/version_0/checkpoints/epoch=0-step=50.ckpt
- inputs: state[1, 2], images[1, 3, 224, 224]
