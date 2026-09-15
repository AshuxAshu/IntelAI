"""Executor state-machine and recovery tests: the six S6 failure modes,
happy path, aborts, gate, and conditioning - all with scripted protocol fakes
and deterministic (inline) VLM call execution. No physics, no models."""

from __future__ import annotations

import numpy as np
import pytest

from dinner_table.contracts.geometry import HOME_JOINTS, TABLE_TOP_HEIGHT
from dinner_table.executor import graph_executor
from dinner_table.executor.graph_executor import (
    MAX_STEP_RETRIES,
    SAFE_JOINT_LIMITS,
    VELOCITY_LIMIT_PER_TICK,
    Executor,
    ExecutorError,
    PolicyObservation,
    TaskAborted,
)
from dinner_table.perception.interfaces import (
    Detection,
    Detector,
    ObjectPose3D,
    Tracker,
)
from dinner_table.reasoning.interfaces import VlmEngine
from dinner_table.reasoning.schema import (
    PreconditionReport,
    RelativeTarget,
    Step,
    TaskGraph,
    VlmDiagnosis,
)

pytestmark = pytest.mark.fast

JOINTS = np.asarray(HOME_JOINTS["A"] + HOME_JOINTS["B"], dtype=np.float64)
FRAME = np.zeros((4, 4, 3), dtype=np.uint8)
DEPTH = np.zeros((4, 4), dtype=np.float32)


@pytest.fixture(autouse=True)
def _inline_vlm_calls(monkeypatch):
    monkeypatch.setattr(graph_executor, "CALL_RUNNER", graph_executor._inline_runner)


class FakeWorld:
    """Mutable scripted perception state the fakes read from."""

    def __init__(self, poses=None, drawer_open=False):
        self.poses = dict(poses or {})
        self.drawer_open = drawer_open

    def set(self, name, x, y, z=TABLE_TOP_HEIGHT, held_by=None):
        self.poses[name] = ObjectPose3D(name=name, position=np.array([x, y, z]), held_by=held_by)

    def drop(self, name):
        self.poses.pop(name, None)


class ScriptedDetector:
    def __init__(self, world):
        self.world = world

    def detect(self, overhead_rgb):
        label = "drawer_open" if self.world.drawer_open else "drawer_closed"
        return [Detection(label=label, xyxy=(0.0, 0.0, 1.0, 1.0), confidence=0.9)]

    def ground(self, snapshot, target):
        return self.world.poses.get(target)


class ScriptedTracker:
    def __init__(self):
        self.latest = {}

    def update(self, poses):
        self.latest = dict(poses)
        return dict(poses)

    def fuse_held(self, poses, gripper_apertures):
        return dict(poses)


class ScriptedVlm:
    def __init__(self, graphs=None):
        self.graphs = list(graphs or [])
        self.checks = []
        self.parse_calls = 0

    def submit_parse(self, instruction, summary):
        self.parse_calls += 1

    @property
    def ready(self):
        return True

    def result(self):
        return self.graphs.pop(0) if self.graphs else None

    def last_error(self):
        return None

    def check_preconditions(self, graph, current_step_id, summary):
        return self.checks.pop(0) if self.checks else PreconditionReport(ok=True, reason="ok")

    def diagnose(self, graph, failed_step_id, summary):
        return VlmDiagnosis(anomaly="none", explanation="", suggested_action="retry_skill")


def make_executor(world, graphs=None):
    vlm = ScriptedVlm(graphs)
    ex = Executor(vlm, ScriptedDetector(world), ScriptedTracker())
    ex.begin_default_skill()
    return ex, vlm


def drive(ex, world, ticks=1):
    for _ in range(ticks):
        grounded = ex.needs_grounding()
        ex.tick(
            PolicyObservation(
                joints=JOINTS,
                overhead_rgb=FRAME if grounded else None,
                overhead_depth=DEPTH if grounded else None,
                timestamp=0.0,
            )
        )


def _graph(*steps, instruction="do the task"):
    return TaskGraph(task_id="t", instruction=instruction, steps=list(steps))


def _pick(i, arm, obj):
    return Step(id=i, skill="pick", arm=arm, object=obj)


def _place(i, arm, obj, target):
    return Step(id=i, skill="place", arm=arm, object=obj, target=target)


class TestProtocols:
    def test_scripted_fakes_satisfy_the_contracts(self):
        assert isinstance(ScriptedVlm(), VlmEngine)
        assert isinstance(ScriptedDetector(FakeWorld()), Detector)
        assert isinstance(ScriptedTracker(), Tracker)


class TestHappyPath:
    def test_pick_then_place_completes(self):
        world = FakeWorld()
        world.set("plate", -0.2, 0.0)
        ex, _ = make_executor(world)
        ex.adopt_graph(_graph(_pick(1, "A", "plate"), _place(2, "A", "plate", "placemat_1")))
        drive(ex, world, 1)
        assert ex.state()["mode"] == "running"
        world.set("plate", -0.2, 0.0, held_by="A")
        drive(ex, world, 25)
        assert ex.state()["postcondition"] == "pass"
        assert ex.state()["group_index"] == 1
        drive(ex, world, 1)
        assert ex.state()["mode"] == "running"
        world.set("plate", -0.06, -0.095, held_by=None)
        drive(ex, world, 25)
        assert ex.done()
        assert ex.state()["terminal_reason"] == "completed"
        assert ex.state()["recovered"] is None

    def test_default_mode_before_plan(self):
        ex, _ = make_executor(FakeWorld())
        drive(ex, FakeWorld(), 3)
        assert ex.state()["mode"] == "default"
        skill, arm, obj, goal = ex.conditioning()
        assert (skill, arm, obj) == ("home", "A", None)
        np.testing.assert_allclose(goal, np.zeros(3))
        assert not ex.done()


class TestRecoveryMatrix:
    def test_missed_grasp_retries_then_succeeds(self):
        world = FakeWorld()
        world.set("plate", -0.2, 0.0)
        ex, _ = make_executor(world)
        ex.adopt_graph(_graph(_pick(1, "A", "plate")))
        drive(ex, world, 1)
        drive(ex, world, 24)
        drive(ex, world, 1)  # poll tick: plate still free -> failure
        state = ex.state()
        assert state["recovered"] == "missed_grasp"
        assert state["retries_left"][1] == MAX_STEP_RETRIES - 1
        assert state["postcondition"].startswith("fail:")
        world.set("plate", -0.2, 0.0, held_by="A")
        drive(ex, world, 25)
        assert ex.done()
        assert ex.state()["terminal_reason"] == "completed"

    def test_dropped_rolls_back_to_pick(self):
        world = FakeWorld()
        world.set("plate", -0.2, 0.0)
        ex, _ = make_executor(world)
        ex.adopt_graph(_graph(_pick(1, "A", "plate"), _place(2, "A", "plate", "placemat_1")))
        drive(ex, world, 1)
        world.set("plate", -0.2, 0.0, held_by="A")
        drive(ex, world, 25)
        drive(ex, world, 1)  # place boundary -> running
        world.set("plate", -0.2, 0.0, held_by=None)  # dropped mid-place, far from goal
        drive(ex, world, 25)  # poll: misplaced -> dropped -> rollback
        state = ex.state()
        assert state["recovered"] == "dropped"
        assert state["group_index"] == 0  # rolled back to the pick group
        drive(ex, world, 1)  # pick boundary again
        world.set("plate", -0.2, 0.0, held_by="A")
        drive(ex, world, 25)
        drive(ex, world, 1)  # place boundary
        world.set("plate", -0.06, -0.095, held_by=None)
        drive(ex, world, 25)
        assert ex.done()
        assert ex.state()["terminal_reason"] == "completed"

    def test_object_moved_boundary_replans(self):
        world = FakeWorld()
        world.set("plate", -0.2, 0.0, held_by="B")
        replacement = _graph(_pick(1, "A", "mug"), instruction="pick the mug instead")
        ex, vlm = make_executor(world, graphs=[replacement])
        ex.adopt_graph(_graph(_pick(1, "A", "plate")))
        drive(ex, world, 1)  # boundary failure 1 (re-perceive)
        drive(ex, world, 1)  # boundary failure 2 -> object_moved -> replan
        state = ex.state()
        assert state["recovered"] == "object_moved"
        assert state["replan_depth"] == 1
        assert vlm.parse_calls == 1
        drive(ex, world, 1)  # replan result adopted -> boundary
        world.set("mug", 0.0, -0.22)
        drive(ex, world, 1)  # mug pick boundary -> running
        world.set("mug", 0.0, -0.22, held_by="A")
        drive(ex, world, 25)
        assert ex.done()
        assert ex.state()["terminal_reason"] == "completed"

    def test_drawer_jammed_wiggles_then_retries(self):
        world = FakeWorld(drawer_open=False)
        ex, _ = make_executor(world)
        ex.adopt_graph(_graph(Step(id=1, skill="open_drawer", arm="A", object="drawer_top")))
        drive(ex, world, 1)
        drive(ex, world, 24)
        drive(ex, world, 1)  # poll: drawer still closed -> jammed
        state = ex.state()
        assert state["recovered"] == "drawer_jammed"
        assert state["wiggle"] is True
        first = ex.gate(np.full(12, 0.5))
        drive(ex, world, 3)
        second = ex.gate(np.full(12, 0.5))
        assert any(first[:5] != second[:5])  # active arm oscillates
        assert all(first[6:] == second[6:])  # idle arm holds still
        world.drawer_open = True
        drive(ex, world, 25)
        assert ex.done()
        assert ex.state()["terminal_reason"] == "completed"
        assert ex.state()["wiggle"] is False

    def test_fumble_retracts_then_re_picks(self):
        world = FakeWorld()
        world.set("bottle", 0.0, -0.22)
        ex, _ = make_executor(world)
        ex.adopt_graph(
            _graph(
                _pick(1, "B", "bottle"),
                Step(id=2, skill="handoff", arm="B", object="bottle", target="hand_of_A"),
            )
        )
        drive(ex, world, 1)
        world.set("bottle", 0.0, -0.22, held_by="B")
        drive(ex, world, 25)  # pick passes
        drive(ex, world, 1)  # handoff boundary -> running
        world.set("bottle", 0.0, -0.22, held_by=None)  # fumbled mid-transfer
        drive(ex, world, 25)  # poll: held by neither -> fumble
        state = ex.state()
        assert state["recovered"] == "fumble"
        assert state["mode"] == "retract_pause"
        drive(ex, world, 25)  # retract pause -> rollback to pick
        assert ex.state()["group_index"] == 0
        drive(ex, world, 1)  # pick boundary
        world.set("bottle", 0.0, -0.22, held_by="B")
        drive(ex, world, 25)
        drive(ex, world, 1)  # handoff boundary
        world.set("bottle", 0.0, -0.22, held_by="A")
        drive(ex, world, 25)
        assert ex.done()
        assert ex.state()["terminal_reason"] == "completed"

    def test_spilled_repours_after_vlm_boundary_check(self):
        world = FakeWorld()
        world.set("bottle", -0.14, -0.10)
        world.set("mug", 0.0, -0.22)
        ex, vlm = make_executor(world)
        ex.adopt_graph(
            _graph(
                _pick(1, "A", "bottle"),
                _pick(2, "B", "mug"),
                Step(id=3, skill="hold", arm="B", object="mug", parallel_group=1),
                Step(
                    id=4,
                    skill="pour",
                    arm="A",
                    object="bottle",
                    target="mug",
                    amount=0.6,
                    parallel_group=1,
                ),
            )
        )
        drive(ex, world, 1)
        world.set("bottle", -0.14, -0.10, held_by="A")
        drive(ex, world, 25)
        drive(ex, world, 1)
        world.set("mug", 0.0, -0.22, held_by="B")
        drive(ex, world, 25)
        drive(ex, world, 1)  # hold+pour boundary -> running
        drive(ex, world, 90)  # pour duration elapses; verification pending
        vlm.checks.append(PreconditionReport(ok=False, reason="mug water level low - spilled"))
        drive(ex, world, 2)  # verify submitted inline, then resolved -> spilled
        state = ex.state()
        assert state["recovered"] == "spilled"
        assert state["group_index"] == 2  # back at the hold+pour group
        drive(ex, world, 1)  # group boundary again
        drive(ex, world, 90)  # re-pour duration
        drive(ex, world, 2)  # second VLM check returns ok -> complete
        assert ex.done()
        assert ex.state()["terminal_reason"] == "completed"


class TestAborts:
    def test_replan_depth_exceeded_aborts(self):
        world = FakeWorld()
        world.set("plate", -0.2, 0.0, held_by="B")
        failing = _graph(_pick(1, "A", "plate"))
        ex, vlm = make_executor(world, graphs=[failing, failing, failing, failing])
        ex.adopt_graph(_graph(_pick(1, "A", "plate")))
        with pytest.raises(TaskAborted) as excinfo:
            for _ in range(60):
                drive(ex, world, 1)
        assert "replan_depth_exceeded" in excinfo.value.reason
        assert ex.done()
        assert ex.state()["terminal_reason"].startswith("aborted:replan_depth_exceeded")
        assert vlm.parse_calls == 3  # the fourth request aborts before submitting

    def test_parse_failure_aborts_after_resubmit(self):
        world = FakeWorld()
        world.set("plate", -0.2, 0.0, held_by="B")
        ex, vlm = make_executor(world, graphs=[])
        ex.adopt_graph(_graph(_pick(1, "A", "plate")))
        with pytest.raises(TaskAborted) as excinfo:
            for _ in range(30):
                drive(ex, world, 1)
        assert "parse_failed" in excinfo.value.reason
        assert ex.state()["terminal_reason"].startswith("aborted:parse_failed")
        assert vlm.parse_calls == 2  # initial replan + one resubmit


class TestGate:
    def _ticked(self, world):
        ex, _ = make_executor(world)
        drive(ex, world, 1)
        return ex

    def test_clamps_joint_limits_and_velocity(self):
        ex = self._ticked(FakeWorld())
        out = ex.gate(np.full(12, 10.0))
        for i, (lo, hi) in enumerate(SAFE_JOINT_LIMITS):
            assert lo - 1e-9 <= out[i] <= hi + 1e-9
            expected = JOINTS[i] + (0.10 if i % 6 == 5 else VELOCITY_LIMIT_PER_TICK)
            assert abs(out[i] - expected) < 1e-9

    def test_default_mode_both_arms_follow(self):
        ex = self._ticked(FakeWorld())
        out = ex.gate(JOINTS + 0.05)
        np.testing.assert_allclose(out, JOINTS + 0.05, atol=1e-9)

    def test_single_arm_skill_holds_idle_arm(self):
        world = FakeWorld()
        world.set("plate", -0.2, 0.0)
        ex, _ = make_executor(world)
        ex.adopt_graph(_graph(_pick(1, "A", "plate")))
        drive(ex, world, 1)  # running, arm A active
        out = ex.gate(JOINTS + 0.05)
        np.testing.assert_allclose(out[6:12], JOINTS[6:12], atol=1e-9)  # B holds
        np.testing.assert_allclose(out[0:6], JOINTS[0:6] + 0.05, atol=1e-9)  # A follows

    def test_wrong_shape_raises(self):
        ex = self._ticked(FakeWorld())
        with pytest.raises(ExecutorError):
            ex.gate(np.zeros(11))


class TestConditioning:
    def test_running_pick_uses_tracker_goal(self):
        world = FakeWorld()
        world.set("plate", -0.2, 0.0)
        ex, _ = make_executor(world)
        ex.adopt_graph(_graph(_pick(1, "A", "plate")))
        drive(ex, world, 1)
        skill, arm, obj, goal = ex.conditioning()
        assert (skill, arm, obj) == ("pick", "A", "plate")
        np.testing.assert_allclose(goal, [-0.2, 0.0, TABLE_TOP_HEIGHT])

    def test_relative_place_resolves_anchor_goal(self):
        world = FakeWorld()
        world.set("fork_1", -0.1, 0.0)
        world.set("plate", -0.2, 0.0)  # anchor for the relative target
        ex, _ = make_executor(world)
        ex.adopt_graph(
            _graph(
                _pick(1, "A", "fork_1"),
                Step(
                    id=2,
                    skill="place",
                    arm="A",
                    object="fork_1",
                    target=RelativeTarget(relation="beside", anchor="plate"),
                ),
            )
        )
        drive(ex, world, 1)
        world.set("fork_1", -0.1, 0.0, held_by="A")
        drive(ex, world, 25)  # pick passes
        drive(ex, world, 1)  # place boundary -> running
        skill, arm, obj, goal = ex.conditioning()
        assert (skill, arm, obj) == ("place", "A", "fork_1")
        np.testing.assert_allclose(goal, [-0.34, 0.0, TABLE_TOP_HEIGHT])  # beside: -0.14 x from A-side plate

    def test_unresolvable_goal_falls_back_to_home(self):
        world = FakeWorld()
        world.set("fork_1", -0.1, 0.0)  # anchor plate never visible
        ex, _ = make_executor(world)
        ex.adopt_graph(
            _graph(
                _pick(1, "A", "fork_1"),
                Step(
                    id=2,
                    skill="place",
                    arm="A",
                    object="fork_1",
                    target=RelativeTarget(relation="beside", anchor="plate"),
                ),
            )
        )
        drive(ex, world, 1)
        world.set("fork_1", -0.1, 0.0, held_by="A")
        drive(ex, world, 25)
        drive(ex, world, 1)  # place boundary -> running (held check passes)
        skill, arm, obj, goal = ex.conditioning()
        assert (skill, arm, obj) == ("home", "A", None)
        np.testing.assert_allclose(goal, np.zeros(3))
        assert ex.needs_grounding()  # perception requested to resolve the anchor
