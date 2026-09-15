"""Live interactive viewer mirroring the running teacher simulation.

Run this in a second terminal while the tests/teacher runs execute:

    uv run python scripts/live_viewer.py

The viewer rebuilds the exact scene (deterministic per seed + DR profile)
that the current test is driving, then follows the physics state published
under artifacts/live/. It auto-reloads whenever the test moves to a new
seed or profile. Controls (MuJoCo native viewer):

  - left-drag / right-drag : orbit / pan
  - scroll                 : zoom
  - double-click           : focus on a body
  - camera dropdown        : overhead, wrist_A, wrist_B, demo_cam

The window title shows the current seed, DR profile, skill, and phase. When
a skill fails, the physics keeps the last (failure) state so you can inspect
the exact frame; the console also prints a line per skill outcome.

Set DINNER_LIVE_DIR to mirror a non-default publisher directory.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

import numpy as np

LIVE_DIR = Path(os.environ.get("DINNER_LIVE_DIR", "artifacts/live"))
POLL_S = 0.02


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_scene(seed: int, profile: str):
    from dinner_table.scene.builder import Scene

    return Scene(seed=seed, dr_profile=profile)


def _open_state(nq: int, nv: int) -> np.ndarray:
    size = 3 + nq + nv
    return np.memmap(LIVE_DIR / "state.bin", dtype=np.float64, mode="r",
                     shape=(size,))


def main() -> int:
    print(f"Waiting for a live publisher under {LIVE_DIR}/ ...")
    print("Start any test (e.g. a teacher pick run) and the viewer will attach.")
    meta = None
    while meta is None:
        meta = _read_json(LIVE_DIR / "meta.json")
        if meta is None:
            time.sleep(0.5)
    seed, profile = int(meta["seed"]), str(meta["profile"])
    scene = _build_scene(seed, profile)
    state = _open_state(int(meta["nq"]), int(meta["nv"]))
    viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
    last_meta = meta
    print(f"Attached: seed {seed}, profile {profile}. Camera dropdown lists all named views.")

    last_status = None
    while viewer.is_running():
        meta = _read_json(LIVE_DIR / "meta.json")
        if meta is not None and (
            int(meta["seed"]) != int(last_meta["seed"])
            or str(meta["profile"]) != str(last_meta["profile"])
        ):
            viewer.close()
            seed, profile = int(meta["seed"]), str(meta["profile"])
            scene = _build_scene(seed, profile)
            state = _open_state(int(meta["nq"]), int(meta["nv"]))
            viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
            last_meta = meta
            print(f"Switched to seed {seed}, profile {profile}.")
            continue

        nq, nv = int(state[1]), int(state[2])
        if nq == scene.model.nq:
            scene.data.qpos[:] = state[3 : 3 + nq]
            scene.data.qvel[:] = state[3 + nq : 3 + nq + nv]
            scene.data.time = float(state[0])
            mujoco.mj_forward(scene.model, scene.data)

        status = _read_json(LIVE_DIR / "status.json")
        if status != last_status and status is not None:
            last_status = status
            line = (f"seed {status['seed']} [{status['profile']}] "
                    f"{status['skill']}({status['skill_object']}) -> {status['phase']}")
            print(line, flush=True)
            try:
                viewer.user_scn.window_title = line
            except AttributeError:
                pass

        viewer.sync()
        time.sleep(POLL_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
