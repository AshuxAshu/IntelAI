#!/usr/bin/env python
"""One command to reproduce the OpenVINO optimization table.

    uv run python scripts/benchmark_openvino.py

Trains the public stand-in checkpoint when none exists, exports the full
precision ladder from it (FP32 / FP16 / INT8 weights / INT8 PTQ), times every
rung on every Intel device this host exposes (CPU, iGPU, NPU), checks each
rung's action output against the FP32 reference, and writes the table to
bench/ov_matrix/ as markdown, CSV, and JSON.

Point it at the real policy once one is trained:

    uv run python scripts/benchmark_openvino.py --ckpt experiments/.../checkpoints/last.ckpt

Full option list: --help (the CLI lives in dinner_table.bench.ov_matrix).
"""

from __future__ import annotations

import sys

from dinner_table.bench.ov_matrix import main

if __name__ == "__main__":
    sys.exit(main())
