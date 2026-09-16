"""Separate drawer-driven fixture motion from free-object displacement."""
from __future__ import annotations

import numpy as np
import pytest

from dinner_table.scene.builder import Scene
from dinner_table.teacher.context import TeacherContext
from dinner_table.teacher.skills import OpenDrawer, Pick, Place


@pytest.mark.slow
@pytest.mark.parametrize("seed", [2, 3, 6])
def test_unpicked_fork_tracks_drawer_during_place(seed):
    scene = Scene(seed=seed, dr_profile="dr_train")
    scene.hold_safe()
    ctx = TeacherContext(scene)
    for skill in (OpenDrawer("A"), Pick("A", "fork_1")):
        ctx.begin(180.0)
        for action in skill.run(ctx):
            ctx.step(action)
    before = scene.object_pose("fork_2")[0]
    opening = ctx.drawer_opening()
    relative_before = before + np.array([0.0, opening, 0.0])
    ctx.begin(180.0)
    for action in Place("A", "fork_1", "placemat_1").run(ctx):
        ctx.step(action)
        relative_now = scene.object_pose("fork_2")[0] + np.array([
            0.0, ctx.drawer_opening(), 0.0,
        ])
        assert np.linalg.norm(relative_now - relative_before) <= 0.004
    after = scene.object_pose("fork_2")[0]
    assert opening - ctx.drawer_opening() > 0.10
    assert np.linalg.norm(after - before) > 0.10
