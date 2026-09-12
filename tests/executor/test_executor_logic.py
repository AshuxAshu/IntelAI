"""Executor logic tests: zones, reassignment, grouping, predicates, claims."""

from __future__ import annotations

import numpy as np
import pytest

from dinner_table.contracts.geometry import HOME_JOINTS, TABLE_TOP_HEIGHT
from dinner_table.executor.preconditions import (
    WorldState,
    check_postcondition,
    check_precondition,
)
from dinner_table.executor.scheduler import group_steps
from dinner_table.executor.workspace import WorkspaceError, ZoneClaims, zone_for_step, zone_of
from dinner_table.perception.interfaces import ObjectPose3D
from dinner_table.reasoning.schema import Step, TaskGraph

pytestmark = pytest.mark.fast


def _pose(name: str, x: float, y: float, z: float = TABLE_TOP_HEIGHT, held_by: str | None = None):
    return ObjectPose3D(name=name, position=np.array([x, y, z]), held_by=held_by)


def _world(
    poses: dict[str, ObjectPose3D] | None = None,
    drawer_open: bool = False,
    multiview_ok: tuple[str, ...] = (),
    joints: dict[str, np.ndarray] | None = None,
) -> WorldState:
    return WorldState(
        poses=poses or {},
        drawer_open=drawer_open,
        multiview_ok=frozenset(multiview_ok),
        joints=joints,
    )


ZONE_CASES = [
    ((0.0, 0.0), "shared"),
    ((0.14, -0.14), "shared"),
    ((-0.1, 0.05), "shared"),
    ((0.1, 0.2), "shared"),  # A/B footprint overlap
    ((0.16, 0.0), "A"),
    ((0.4, 0.24), "A"),
    ((0.45, 0.0), "A"),  # inclusive x_max boundary
    ((-0.16, 0.0), "B"),
    ((-0.44, -0.2), "B"),
    ((0.5, 0.0), "out"),
    ((0.0, 0.3), "out"),
    ((0.0, 0.51), "out"),  # drawer region
]


class TestZoneContainment:
    @pytest.mark.parametrize(("xy", "expected"), ZONE_CASES)
    def test_zone_of_classifies(self, xy, expected):
        assert zone_of(np.array([xy[0], xy[1], TABLE_TOP_HEIGHT])) == expected

    def test_zone_for_step_static_mapping(self):
        assert (
            zone_for_step(Step(id=1, skill="place", arm="A", object="plate", target="placemat_1"))
            == "A"
        )
        assert (
            zone_for_step(Step(id=1, skill="place", arm="B", object="plate", target="placemat_2"))
            == "B"
        )
        assert (
            zone_for_step(Step(id=1, skill="open_drawer", arm="A", object="drawer_top")) == "shared"
        )
        assert (
            zone_for_step(
                Step(id=1, skill="pour", arm="A", object="bottle", target="mug", amount=0.5)
            )
            == "shared"
        )
        assert zone_for_step(Step(id=1, skill="pick", arm="A", object="plate")) == "out"
        assert zone_for_step(Step(id=1, skill="retract", arm="B")) == "out"


class TestReassignment:
    def _pick_place_graph(self, arm: str, target: str) -> TaskGraph:
        return TaskGraph(
            task_id="t",
            instruction="place the plate",
            steps=[
                Step(id=1, skill="pick", arm=arm, object="plate"),
                Step(id=2, skill="place", arm=arm, object="plate", target=target),
            ],
        )

    def test_object_in_other_zone_swaps_arm(self):
        graph = self._pick_place_graph("B", "placemat_1")
        poses = {"plate": _pose("plate", 0.30, 0.0)}
        groups, log = group_steps(graph, poses)
        assert groups[0][0].arm == "A"
        assert groups[1][0].arm == "A"  # place follows the reassigned pick's holder
        assert log == [
            "reassigned step 1 to arm A: goal in A",
            "reassigned step 2 to arm A: follows object holder",
        ]

    def test_object_in_shared_zone_unchanged(self):
        graph = self._pick_place_graph("B", "placemat_2")  # B-side target, B arm: consistent
        poses = {"plate": _pose("plate", 0.0, 0.0)}
        groups, log = group_steps(graph, poses)
        assert all(group[0].arm == "B" for group in groups)
        assert log == []

    def test_shared_object_with_far_goal_stays_on_holder(self):
        # Plate in the shared zone, placement in A's exclusive zone: the place
        # must NOT swap to A (arm B holds the plate) - the chain wins and the
        # mismatch surfaces as a warning line instead.
        graph = self._pick_place_graph("B", "placemat_1")
        poses = {"plate": _pose("plate", 0.0, 0.0)}
        groups, log = group_steps(graph, poses)
        assert all(group[0].arm == "B" for group in groups)
        assert log == ["step 2: placement goal in A unreachable by arm B - left for recovery"]

    def test_handoff_swap_flips_target_hand(self):
        graph = TaskGraph(
            task_id="t",
            instruction="hand off the bottle",
            steps=[
                Step(id=1, skill="pick", arm="B", object="bottle"),
                Step(id=2, skill="handoff", arm="B", object="bottle", target="hand_of_A"),
            ],
        )
        poses = {"bottle": _pose("bottle", 0.30, 0.05)}
        groups, log = group_steps(graph, poses)
        assert groups[0][0].arm == "A"  # pick reassigned by object position
        assert groups[1][0].arm == "A"  # handoff source follows the holder
        assert groups[1][0].target == "hand_of_B"
        assert log == [
            "reassigned step 1 to arm A: goal in A",
            "reassigned step 2 to arm A: follows object holder",
        ]

    def test_relative_place_goal_feasibility_warning(self):
        # Fork in B's zone, anchor plate in A's zone: the relative goal
        # resolves into A's exclusive zone while B holds the fork - warning,
        # no chain-breaking swap.
        graph = TaskGraph(
            task_id="t",
            instruction="put a fork beside the plate",
            steps=[
                Step(id=1, skill="pick", arm="B", object="fork_1"),
                Step(
                    id=2,
                    skill="place",
                    arm="B",
                    object="fork_1",
                    target={"relation": "right_of", "anchor": "plate"},
                ),
            ],
        )
        poses = {
            "fork_1": _pose("fork_1", -0.30, 0.0),
            "plate": _pose("plate", 0.30, 0.0),
        }
        groups, log = group_steps(graph, poses)
        assert all(group[0].arm == "B" for group in groups)
        assert groups[1][0].target.relation == "right_of"  # target itself is preserved
        assert log == ["step 2: placement goal in A unreachable by arm B - left for recovery"]

    def test_relative_place_goal_reachable_no_warning(self):
        graph = TaskGraph(
            task_id="t",
            instruction="put a fork beside the plate",
            steps=[
                Step(id=1, skill="pick", arm="B", object="fork_1"),
                Step(
                    id=2,
                    skill="place",
                    arm="B",
                    object="fork_1",
                    target={"relation": "right_of", "anchor": "plate"},
                ),
            ],
        )
        poses = {
            "fork_1": _pose("fork_1", -0.10, 0.0),
            "plate": _pose("plate", 0.0, 0.0),  # anchor shared -> goal (0.14, 0) shared
        }
        groups, log = group_steps(graph, poses)
        assert all(group[0].arm == "B" for group in groups)
        assert log == []

    def test_invisible_object_no_reassignment(self):
        graph = self._pick_place_graph("B", "placemat_2")  # consistent B-side instruction
        groups, log = group_steps(graph, {})
        assert all(group[0].arm == "B" for group in groups)
        assert log == []


class TestParallelGrouping:
    def test_canonical_graph_groups(self):
        graph = TaskGraph(
            task_id="t",
            instruction="pour water into the mug",
            steps=[
                Step(id=1, skill="pick", arm="B", object="mug"),
                Step(id=2, skill="hold", arm="B", object="mug", parallel_group=1),
                Step(
                    id=3,
                    skill="pour",
                    arm="A",
                    object="bottle",
                    target="mug",
                    amount=0.6,
                    parallel_group=1,
                ),
            ],
        )
        poses = {"mug": _pose("mug", -0.2, 0.0)}  # mug in B's zone: no reassignment
        groups, log = group_steps(graph, poses)
        assert [[step.id for step in group] for group in groups] == [[1], [2, 3]]
        assert log == []


class TestPreconditions:
    def test_pick_pass(self):
        step = Step(id=1, skill="pick", arm="A", object="plate")
        world = _world({"plate": _pose("plate", 0.2, 0.0)}, multiview_ok=("plate",))
        report = check_precondition(step, world)
        assert report.ok and report.reason

    def test_pick_fail_held(self):
        step = Step(id=1, skill="pick", arm="A", object="plate")
        world = _world({"plate": _pose("plate", 0.2, 0.0, held_by="B")})
        report = check_precondition(step, world)
        assert not report.ok and report.reason

    def test_pick_fail_multiview(self):
        step = Step(id=1, skill="pick", arm="A", object="plate")
        world = _world({"plate": _pose("plate", 0.2, 0.0)})
        report = check_precondition(step, world)
        assert not report.ok and report.reason

    def test_pick_fail_invisible(self):
        step = Step(id=1, skill="pick", arm="A", object="plate")
        report = check_precondition(step, _world())
        assert not report.ok and report.reason

    def test_place_precondition_pass_and_fail(self):
        step = Step(id=1, skill="place", arm="A", object="plate", target="placemat_1")
        held_by_a = _world({"plate": _pose("plate", 0.2, 0.0, held_by="A")})
        held_by_b = _world({"plate": _pose("plate", 0.2, 0.0, held_by="B")})
        assert check_precondition(step, held_by_a).ok
        report = check_precondition(step, held_by_b)
        assert not report.ok and report.reason

    def test_open_drawer_precondition_pass_and_fail(self):
        step = Step(id=1, skill="open_drawer", arm="A", object="drawer_top")
        assert check_precondition(step, _world(drawer_open=False)).ok
        report = check_precondition(step, _world(drawer_open=True))
        assert not report.ok and report.reason

    def test_close_drawer_precondition_pass_and_fail(self):
        step = Step(id=1, skill="close_drawer", arm="A", object="drawer_top")
        assert check_precondition(step, _world(drawer_open=True)).ok
        report = check_precondition(step, _world(drawer_open=False))
        assert not report.ok and report.reason

    def test_handoff_precondition_pass_and_fail(self):
        step = Step(id=1, skill="handoff", arm="B", object="bottle", target="hand_of_A")
        assert check_precondition(step, _world({"bottle": _pose("bottle", 0, 0, held_by="B")})).ok
        report = check_precondition(step, _world({"bottle": _pose("bottle", 0, 0, held_by="A")}))
        assert not report.ok and report.reason

    def test_hold_precondition_pass_and_fail(self):
        step = Step(id=1, skill="hold", arm="B", object="mug")
        assert check_precondition(step, _world({"mug": _pose("mug", 0, 0, held_by="B")})).ok
        report = check_precondition(step, _world({"mug": _pose("mug", 0, 0)}))
        assert not report.ok and report.reason

    def test_pour_precondition_pass_and_fail(self):
        step = Step(id=1, skill="pour", arm="A", object="bottle", target="mug", amount=0.6)
        good = _world(
            {
                "bottle": _pose("bottle", 0, 0, held_by="A"),
                "mug": _pose("mug", 0, 0, held_by="B"),
            }
        )
        assert check_precondition(step, good).ok
        bad = _world(
            {
                "bottle": _pose("bottle", 0, 0, held_by="A"),
                "mug": _pose("mug", 0, 0),
            }
        )
        report = check_precondition(step, bad)
        assert not report.ok and report.reason

    def test_home_precondition_always_ok(self):
        step = Step(id=1, skill="home", arm="A")
        assert check_precondition(step, _world()).ok

    def test_retract_precondition_always_ok(self):
        step = Step(id=1, skill="retract", arm="B")
        assert check_precondition(step, _world()).ok


class TestPostconditions:
    def test_pick_pass_and_fail(self):
        step = Step(id=1, skill="pick", arm="A", object="plate")
        assert check_postcondition(step, _world({"plate": _pose("plate", 0.2, 0, held_by="A")})).ok
        report = check_postcondition(step, _world({"plate": _pose("plate", 0.2, 0)}))
        assert not report.ok and report.reason

    def test_place_pass(self):
        step = Step(id=1, skill="place", arm="A", object="plate", target="placemat_1")
        goal = np.array([0.22, 0.10, TABLE_TOP_HEIGHT])
        world = _world({"plate": _pose("plate", 0.22, 0.10)})
        assert check_postcondition(step, world, goal=goal).ok

    def test_place_fail_too_far(self):
        step = Step(id=1, skill="place", arm="A", object="plate", target="placemat_1")
        goal = np.array([0.22, 0.10, TABLE_TOP_HEIGHT])
        world = _world({"plate": _pose("plate", 0.27, 0.10)})  # 5 cm off
        report = check_postcondition(step, world, goal=goal)
        assert not report.ok and report.reason

    def test_place_fail_still_held(self):
        step = Step(id=1, skill="place", arm="A", object="plate", target="placemat_1")
        goal = np.array([0.22, 0.10, TABLE_TOP_HEIGHT])
        world = _world({"plate": _pose("plate", 0.22, 0.10, held_by="A")})
        report = check_postcondition(step, world, goal=goal)
        assert not report.ok and report.reason

    def test_drawer_postconditions(self):
        open_step = Step(id=1, skill="open_drawer", arm="A", object="drawer_top")
        close_step = Step(id=1, skill="close_drawer", arm="A", object="drawer_top")
        assert check_postcondition(open_step, _world(drawer_open=True)).ok
        assert not check_postcondition(open_step, _world(drawer_open=False)).ok
        assert check_postcondition(close_step, _world(drawer_open=False)).ok
        assert not check_postcondition(close_step, _world(drawer_open=True)).ok

    def test_handoff_pass_and_fail(self):
        step = Step(id=1, skill="handoff", arm="B", object="bottle", target="hand_of_A")
        assert check_postcondition(step, _world({"bottle": _pose("bottle", 0, 0, held_by="A")})).ok
        report = check_postcondition(step, _world({"bottle": _pose("bottle", 0, 0, held_by="B")}))
        assert not report.ok and report.reason

    def test_hold_and_pour_posts_are_delegated(self):
        hold = Step(id=1, skill="hold", arm="B", object="mug")
        pour = Step(id=1, skill="pour", arm="A", object="bottle", target="mug", amount=0.6)
        assert check_postcondition(hold, _world()).ok
        assert check_postcondition(pour, _world()).ok

    def test_home_postcondition_pass_and_fail(self):
        step = Step(id=1, skill="home", arm="A")
        at_home = np.asarray(HOME_JOINTS["A"], dtype=np.float64)
        away = at_home.copy()
        away[0] += 0.5
        assert check_postcondition(step, _world(joints={"A": at_home})).ok
        report = check_postcondition(step, _world(joints={"A": away}))
        assert not report.ok and report.reason

    def test_retract_postcondition_pass_and_fail(self):
        step = Step(id=1, skill="retract", arm="B")
        at_home = np.asarray(HOME_JOINTS["B"], dtype=np.float64)
        away = at_home.copy()
        away[1] -= 0.4
        assert check_postcondition(step, _world(joints={"B": at_home})).ok
        report = check_postcondition(step, _world(joints={"B": away}))
        assert not report.ok and report.reason


class TestZoneClaims:
    def test_shared_zone_mutual_exclusion(self):
        claims = ZoneClaims()
        assert claims.claim("A", "shared") is True
        assert claims.claim("B", "shared") is False
        claims.release("A")
        assert claims.claim("B", "shared") is True

    def test_same_arm_reclaim_is_idempotent(self):
        claims = ZoneClaims()
        assert claims.claim("A", "shared") is True
        assert claims.claim("A", "shared") is True

    def test_unclaimable_zone_is_noop(self):
        claims = ZoneClaims()
        assert claims.claim("A", "out") is True
        assert claims.claim("B", "out") is True

    def test_distinct_arm_zones_do_not_conflict(self):
        claims = ZoneClaims()
        assert claims.claim("A", "A") is True
        assert claims.claim("B", "B") is True

    def test_release_only_frees_that_arm(self):
        claims = ZoneClaims()
        claims.claim("A", "A")
        claims.claim("A", "shared")
        claims.release("A")
        assert claims.claim("B", "shared") is True

    def test_invalid_arm_rejected(self):
        claims = ZoneClaims()
        with pytest.raises(WorkspaceError):
            claims.claim("C", "shared")
