"""Final pickup verification must not accept an out-of-limit carried pose."""
from __future__ import annotations

import numpy as np
import pytest

from dinner_table.scene.builder import Scene
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.grasp_catalog import GraspCatalog
from dinner_table.teacher.skills import Pick


@pytest.mark.slow
@pytest.mark.parametrize("seed", [2, 3, 14])
def test_plate_pick_rejects_excessive_post_lift_tilt(seed):
    scene = Scene(seed=seed, dr_profile="dr_train")
    scene.hold_safe()
    ctx = TeacherContext(scene)
    ctx.begin(180.0)
    skill = Pick("A", "plate")
    with pytest.raises(SkillFailed) as failure:
        for action in skill.run(ctx):
            ctx.step(action)
    assert (failure.value.skill, failure.value.phase, failure.value.cause) == (
        "pick", "verify", "missed_grasp",
    )
    assert ctx.object_upright("plate") < np.cos(
        np.deg2rad(GraspCatalog.CARRY_TILT_DEG)
    )
    assert min(ctx.finger_forces("A", "plate")) > 0.08
    assert ctx.object_external_force("A", "plate") == 0.0
