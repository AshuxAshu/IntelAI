# Reproducing the OpenVINO optimization matrix

This is the short version for someone who has just cloned the repository and
wants the optimization table on their own Intel machine. You do not need to
train anything, download a dataset by hand, or pick an export format - one
command does the whole chain and prints the result.

## What you need

- **Linux or Windows** on an Intel machine. (macOS is not supported for the
  benchmark: the Intel stack has no macOS wheels at all, so `uv sync` there
  cannot install `openvino`/`physicalai`.)
- **`uv`**, which also installs the right Python for you. It is often not
  preinstalled - if `uv` is "not recognized" / "command not found", install it:

  ```powershell
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

  ```sh
  # Linux / macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  Alternatives: `winget install --id=astral-sh.uv -e` on Windows, or
  `pip install uv` anywhere. Restart your terminal afterwards so the new
  `uv` is on `PATH`. Check with `uv --version`.
- An Intel CPU with OpenVINO support. An iGPU and/or NPU are picked up
  automatically when present; on a laptop without an NPU the NPU column prints
  a dash, it does not fail the run.

Windows notes: `make` is not available, so run the Python command directly (the
`make bench-ov` equivalent). The benchmark needs no MuJoCo rendering, so no GPU
setup beyond the Intel graphics driver is required. `uv sync` resolves
correctly on Windows - `torch` is taken from PyPI there instead of the
Linux-only CPU wheel index, and `torchcodec` is skipped because it has no
Windows build.

## Run it

```sh
uv sync
uv run python scripts/benchmark_openvino.py
```

The first run trains a small public stand-in checkpoint (about a minute) and
exports the four precision rungs (about half a minute), then times everything.
Later runs reuse both. The table is printed and also written to
`bench/ov_matrix/` as markdown, CSV, and JSON.

Equivalent make target (Linux only): `make bench-ov`.

## If something fails

**`uv: The term 'uv' is not recognized` (Windows) / `uv: command not found`.**
`uv` is not installed or not on `PATH`. Install it and reopen the terminal -
see "What you need" above.

**`PermissionError: [WinError 32] ... used by another process` while exporting
or quantizing.** Windows locks a file while OpenVINO has it memory-mapped, so a
rung that reads an IR and then saves over it can hit this if something else
(Antivirus, OneDrive/Dropbox sync, a Windows indexer) has the file open. The
code already reads IRs through in-memory buffers to avoid the lock and retries
the move briefly. If it still happens: close anything syncing the project
folder, or run from a plain local directory such as `C:\work\IntelAI` rather
than a cloud-synced one, then re-run with `--force`.

**`Incorrect weights in bin file!`** An IR is missing its sibling `.bin`. Delete
the offending rung directory under the exports root and re-run with `--force`.

## What the table means

Rows are policy variants. Columns are the devices OpenVINO exposes on this host
(`CPU`, `iGPU`, `NPU`). The last column is how far each variant's actions drift
from the FP32 reference on identical inputs - the quantization-quality check.
A fast row with a bad last column is a row you should not ship.

Timing uses Intel's own `InferenceLatencyBenchmark` + `RandomInputSource`, the
same methodology as `make bench`, so numbers from the two agree.

## Known limits (read before quoting numbers)

- **The checkpoint is a stand-in.** With no `--ckpt`, the matrix trains on the
  public `lerobot/pusht` dataset. Architecture and export pipeline are the real
  thing; the task is not ours. Point `--ckpt` at a trained dinner-table
  checkpoint to benchmark the policy you actually deploy.
- **There is no task-success column yet.** The quality column is action parity,
  not "mug placed". A policy-in-the-loop success number needs a learned policy
  driving the simulation, which is not implemented - the demos are driven by
  privileged teacher skills. `--closed-loop teacher` records the teacher-oracle
  system smoke rate and labels it precision-independent; never present it as
  quantization evidence.
- **Absent devices are shown, not hidden.** A dash means the device is not on
  this host. It does not mean the rung was skipped.

## Useful flags

```sh
uv run python scripts/benchmark_openvino.py --help
uv run python scripts/benchmark_openvino.py --ckpt path/to/last.ckpt   # real policy
uv run python scripts/benchmark_openvino.py --force                    # re-export rungs
uv run python scripts/benchmark_openvino.py --iters 500                # longer timing
uv run python scripts/benchmark_openvino.py --closed-loop teacher --seeds 10
```

## Verifying the plumbing without a benchmark run

```sh
uv run pytest tests/bench/test_ov_matrix.py -m fast
```
