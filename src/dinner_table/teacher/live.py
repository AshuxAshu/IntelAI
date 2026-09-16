"""Live state publisher: mirrors the running teacher simulation to disk.

Writes a tiny shared-memory file (np.memmap) plus a status JSON under
``artifacts/live/`` every physics step, so ``scripts/live_viewer.py`` can
render exactly what the current test is doing in a separate process. The
publisher is failure-tolerant and adds negligible overhead; set
``DINNER_LIVE=0`` to disable.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

LIVE_DIR = Path(os.environ.get("DINNER_LIVE_DIR", "artifacts/live"))
BUFFER_FLOATS = 512  # header (5) + qpos (62) + qvel (56) + slack
DISABLED = os.environ.get("DINNER_LIVE", "1") == "0"


class LivePublisher:
    """Publishes (seed, profile, time, qpos, qvel) and skill status per step."""

    def __init__(self, scene) -> None:
        self.active = not DISABLED
        self.scene = scene
        self._mm = None
        self._status = {}
        self._last_status_write = 0.0
        if not self.active:
            return
        try:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            self._mm = np.memmap(LIVE_DIR / "state.bin", dtype=np.float64,
                                 mode="w+", shape=(BUFFER_FLOATS,))
            self._publish_meta()
        except (OSError, ValueError):
            # ValueError: two processes sharing LIVE_DIR can map the file
            # between another's create and resize ("mmap length is greater
            # than file size"). Publishing is a debug convenience, so a lost
            # race silently disables it rather than killing the episode.
            self.active = False

    def _publish_meta(self) -> None:
        (LIVE_DIR / "meta.json").write_text(
            json.dumps({
                "seed": int(self.scene.seed),
                "profile": str(self.scene.dr_profile_name),
                "nq": int(self.scene.model.nq),
                "nv": int(self.scene.model.nv),
                "written_at": time.time(),
            }),
            encoding="utf-8",
        )

    def note_skill(self, skill) -> None:
        """Record the current skill name/phase for the viewer title bar."""
        if not self.active or skill is None:
            return
        entry = {
            "seed": int(self.scene.seed),
            "profile": str(self.scene.dr_profile_name),
            "skill": type(skill).__name__,
            "skill_object": getattr(skill, "object_name", None) or "-",
            "phase": getattr(skill, "phase", "-"),
        }
        if entry != self._status:
            self._status = entry
            try:
                (LIVE_DIR / "status.json").write_text(json.dumps(entry), encoding="utf-8")
            except OSError:
                pass

    def step(self) -> None:
        """Mirror one physics state; called every control tick."""
        if not self.active:
            return
        m, d = self.scene.model, self.scene.data
        mm = self._mm
        mm[0] = float(d.time)
        mm[1] = float(m.nq)
        mm[2] = float(m.nv)
        mm[3 : 3 + m.nq] = d.qpos
        mm[3 + m.nq : 3 + m.nq + m.nv] = d.qvel
        mm.flush()
