"""Contract self-tests: frozen interfaces behave exactly as specified (S10)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from dinner_table.contracts.geometry import ACTION_DIM, JOINT_NAMES
from dinner_table.policies.conditioning import STATE_DIM, build_state
from dinner_table.reasoning.schema import Step, TaskGraph

pytestmark = pytest.mark.fast


def _canonical_graph() -> TaskGraph:
    return TaskGraph(
        task_id="t1",
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


class TestTaskGraph:
    def test_canonical_graph_parses(self):
        assert [s.id for s in _canonical_graph().steps] == [1, 2, 3]

    def test_place_without_pick_rejected(self):
        with pytest.raises(ValidationError):
            TaskGraph(
                task_id="t",
                instruction="x",
                steps=[
                    Step(id=1, skill="place", arm="A", object="plate", target="placemat_1"),
                ],
            )

    def test_hallucinated_key_rejected(self):
        with pytest.raises(ValidationError):
            Step(id=1, skill="home", arm="A", nonexistent_field=1)

    def test_unknown_skill_rejected(self):
        with pytest.raises(ValidationError):
            Step(id=1, skill="teleport", arm="A")

    def test_parallel_group_same_arm_rejected(self):
        with pytest.raises(ValidationError):
            TaskGraph(
                task_id="t",
                instruction="x",
                steps=[
                    Step(id=1, skill="hold", arm="B", object="mug", parallel_group=1),
                    Step(
                        id=2, skill="pour", arm="B", object="bottle", target="mug", parallel_group=1
                    ),
                ],
            )

    def test_pour_without_hold_rejected(self):
        with pytest.raises(ValidationError):
            TaskGraph(
                task_id="t",
                instruction="x",
                steps=[
                    Step(id=1, skill="pour", arm="A", object="bottle", target="mug", amount=0.5),
                ],
            )

    def test_non_increasing_ids_rejected(self):
        with pytest.raises(ValidationError):
            TaskGraph(
                task_id="t",
                instruction="x",
                steps=[
                    Step(id=2, skill="home", arm="A"),
                    Step(id=1, skill="retract", arm="B"),
                ],
            )

    def test_relative_target_parses(self):
        graph = TaskGraph(
            task_id="t",
            instruction="put a fork beside the plate",
            steps=[
                Step(id=1, skill="pick", arm="A", object="fork_1"),
                Step(
                    id=2,
                    skill="place",
                    arm="A",
                    object="fork_1",
                    target={"relation": "beside", "anchor": "plate"},
                ),
            ],
        )
        assert graph.steps[1].target.relation == "beside"

    def test_relative_target_bad_relation_rejected(self):
        with pytest.raises(ValidationError):
            Step(
                id=1,
                skill="place",
                arm="A",
                object="fork_1",
                target={"relation": "above", "anchor": "plate"},
            )

    def test_handoff_rejects_relative_target(self):
        with pytest.raises(ValidationError):
            TaskGraph(
                task_id="t",
                instruction="x",
                steps=[
                    Step(id=1, skill="pick", arm="B", object="bottle"),
                    Step(
                        id=2,
                        skill="handoff",
                        arm="B",
                        object="bottle",
                        target={"relation": "beside", "anchor": "plate"},
                    ),
                ],
            )


class TestConditioning:
    def test_state_layout_exact(self):
        joints = np.arange(12, dtype=np.float64)
        state = build_state(joints, "pick", "A", "mug", np.array([0.1, 0.2, 0.4]))
        assert state.shape == (STATE_DIM,) and STATE_DIM == 35
        # non-gripper gather: indices [0,1,2,3,4,6,7,8,9,10] of JOINT_NAMES order
        np.testing.assert_allclose(state[0:10], joints[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]])
        assert state[10] == 5.0  # A.gripper slot holds joints[5]
        assert state[11] == 11.0  # B.gripper slot holds joints[11]
        assert state[12] == 1.0 and state[13] == 0.0  # arm one-hot A
        assert state[14 + 2] == 1.0  # "pick" is SKILLS[2]
        assert state[23 + 1] == 1.0  # "mug" is index 1
        np.testing.assert_allclose(state[32:35], [0.1, 0.2, 0.4])

    def test_state_wrong_joint_shape_raises(self):
        with pytest.raises(ValueError):
            build_state(np.zeros(11), "home", "A", None, np.zeros(3))


class TestGeometry:
    def test_joint_names_canonical(self):
        assert len(JOINT_NAMES) == 12
        assert JOINT_NAMES[0] == "A.shoulder_pan"
        assert JOINT_NAMES[5] == "A.gripper"
        assert JOINT_NAMES[11] == "B.gripper"
        assert ACTION_DIM == 12
