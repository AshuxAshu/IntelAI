# Intel benchmark results

- schema host cpu: 13th Gen Intel(R) Core(TM) i7-13620H
- os: Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44
- ram: 15.25 GB
- devices: CPU, GPU.0, GPU.1
- npu available: False
- power profile: None
- openvino 2026.1.0 | physicalai 0.1.1 | physicalai-train 0.1.0 | nncf 3.3.0

| device | full name | type | architecture | execution units |
| --- | --- | --- | --- | --- |
| CPU | 13th Gen Intel(R) Core(TM) i7-13620H | INTEGRATED | intel64 | - |
| GPU.0 | Intel(R) UHD Graphics (iGPU) | INTEGRATED | GPU: vendor=0x8086 arch=v12.3.0 | 64 |
| GPU.1 | NVIDIA GeForce RTX 4050 Laptop GPU (dGPU) | DISCRETE | GPU: vendor=0x10de arch=v8.9.0 | 20 |

## ACT policy

| backend | device | precision | p50 ms | p90 ms | p95 ms | p99 ms | throughput ips | cold compile s | weights | activations | rss MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openvino | CPU | FP32 | 17.608 | 19.067 | 19.758 | 22.149 | 56.4 | 0.38 | FP32 | FP32 | 348.6 |
| openvino | GPU.0 | FP32 | 20.402 | 22.263 | 24.179 | 27.514 | 49.12 | 0.67 | FP32 | FP32 | 165.5 |
| openvino | CPU | FP16 | 18.496 | 20.261 | 21.161 | 21.822 | 53.62 | 0.65 | FP16 | FP32 | 228.9 |
| openvino | GPU.0 | FP16 | 18.66 | 22.462 | 27.543 | 140.396 | 40.22 | 0.57 | FP16 | FP32 | 25.6 |
| openvino | CPU | INT8 | 6.113 | 7.137 | 7.949 | 11.005 | 159.81 | 0.58 | INT8 | INT8 | 126.2 |
| openvino | GPU.0 | INT8 | 19.348 | 22.238 | 23.564 | 24.73 | 55.15 | 0.86 | INT8 | INT8 | 38.4 |
| pytorch | cpu | FP32 | 33.242 | 38.804 | 44.041 | 194.986 | 25.96 | 1.09 | FP32 | FP32 | 826.0 |

### PyTorch comparison

| backend | device | precision | p50 ms | speedup vs pytorch-cpu |
| --- | --- | --- | --- | --- |
| openvino | CPU | FP32 | 17.608 | 1.89 |
| openvino | GPU.0 | FP32 | 20.402 | 1.63 |
| openvino | CPU | FP16 | 18.496 | 1.8 |
| openvino | GPU.0 | FP16 | 18.66 | 1.78 |
| openvino | CPU | INT8 | 6.113 | 5.44 |
| openvino | GPU.0 | INT8 | 19.348 | 1.72 |
| pytorch | cpu | FP32 | 33.242 | 1.0 |

## YOLO detector

| backend | device | precision | p50 ms | p90 ms | p95 ms | p99 ms | throughput ips | cold compile s | weights | activations | rss MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openvino | CPU | FP32 | 20.1 | 22.92 | 25.383 | 29.771 | 48.28 | 0.24 | FP32 | FP32 | 9.0 |
| openvino | GPU.0 | FP32 | 37.56 | 44.598 | 46.179 | 50.845 | 27.16 | 0.53 | FP32 | FP32 | 113.7 |

## VLM generation

| device | precision | prefill tok/s | decode tok/s | e2e 128 tok s |
| --- | --- | --- | --- | --- |
| CPU | INT8 | 128.27 | 14.16 | 11.844 |
| GPU.0 | - | - | - | - |

## Failed cells

| model | device | precision | error |
| --- | --- | --- | --- |
| vlm | GPU.0 | None | Exception from src/inference/src/cpp/infer_request.cpp:224:
Check 'args[i].index < data.inputs.size() && data.inputs[args[i].index]' failed at src/plugins/intel_gpu/src/runtime/ocl/ocl_stream.cpp:92:
The allocated input memory is necessary to set kernel arguments.

 |
