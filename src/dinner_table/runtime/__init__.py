"""Runtime package: action source, telemetry, HUD, and sim adapters.

The dinner_demo.yaml class_paths (dinner_table.runtime.SkillExecutorSource,
…) resolve against these names. The MuJoCo adapters import physicalai, which
has no macOS wheels, so they resolve lazily: importing this package never
fails, and accessing an adapter on a host without the Intel stack raises the
original ImportError with context.
"""

from __future__ import annotations

from dinner_table.runtime.hud import render_hud
from dinner_table.runtime.skill_source import SkillExecutorSource
from dinner_table.runtime.telemetry import HudTelemetryCallback, TickRecord

__all__ = [
    "HudTelemetryCallback",
    "MuJoCoBimanualRobot",
    "MuJoCoCamera",
    "SkillExecutorSource",
    "TickRecord",
    "render_hud",
]

_LAZY = {
    "MuJoCoBimanualRobot": "dinner_table.runtime.mujoco_robot",
    "MuJoCoCamera": "dinner_table.runtime.mujoco_camera",
}


def __getattr__(name: str):  # PEP 562 lazy adapters
    if name in _LAZY:
        import importlib

        try:
            module = importlib.import_module(_LAZY[name])
        except ImportError as exc:
            raise ImportError(
                f"{name} needs the Intel stack (physicalai), which is not installed on this host"
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
