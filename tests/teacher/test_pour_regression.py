"""Real-physics seed-zero pouring regression, without task-graph retries."""

import numpy as np
import pytest

from dinner_table.scene.objects import BOTTLE_MOUTH_Z
from dinner_table.teacher import pour_planner
from dinner_table.teacher.kinematics import arm_q

from .test_bimanual_skills import _pour_episode


@pytest.mark.slow
@pytest.mark.parametrize("mass_scale", [1.0, 2.0], ids=["nominal", "double"])
def test_held_pour_seed_zero(mass_scale, monkeypatch):
    original = pour_planner.plan_return
    calls = []

    def audited_return(ctx, arm, bottle):
        before = [array.copy() for array in
                  (ctx.data.qpos, ctx.data.qvel, ctx.data.xfrc_applied, ctx.data.qfrc_applied)]
        geometry = pour_planner._PourGeometry(ctx, arm, bottle)
        q = arm_q(ctx.data, arm)
        mouth, rotation, base = geometry.pose(q)
        body = ctx.model.body(bottle).id
        live_rotation = ctx.data.xmat[body].reshape(3, 3)
        np.testing.assert_allclose(base, ctx.data.xpos[body], atol=1e-9)
        np.testing.assert_allclose(rotation, live_rotation, atol=1e-9)
        np.testing.assert_allclose(
            mouth, ctx.data.xpos[body] + BOTTLE_MOUTH_Z * live_rotation[:, 2], atol=1e-9)
        assert geometry.clear(q), "recovery starts in predicted external contact"
        try:
            path = original(ctx, arm, bottle)
            np.testing.assert_allclose(path.joints[0], q)
            calls.append(True)
            return path
        finally:
            for expected, actual in zip(before, (ctx.data.qpos, ctx.data.qvel,
                                                ctx.data.xfrc_applied, ctx.data.qfrc_applied)):
                np.testing.assert_array_equal(expected, actual)

    monkeypatch.setattr(pour_planner, "plan_return", audited_return)
    result = _pour_episode(0, mass_scale, True)
    assert calls, result.outcomes
    assert not result.collisions, result.collisions
    assert result.outcomes.get("pour") == "ok", result.outcomes
    assert result.flow_events > 0
