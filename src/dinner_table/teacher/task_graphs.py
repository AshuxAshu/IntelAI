"""Canonical task graphs the teacher demonstrates and the data engine samples.

The frozen ``PlacementTarget`` vocabulary names only the two placemats and the
drawer tray, so cutlery settings are expressed as ``RelativeTarget`` placements
anchored on the plate. That is also the form the judges' spatial-relational
commands take ("put a fork beside the plate"), so the teacher demonstrates the
exact resolution path the deployed runtime uses.
"""

from __future__ import annotations

from dinner_table.reasoning.schema import RelativeTarget, Step, TaskGraph

FORK_SETTING = RelativeTarget(relation="left_of", anchor="plate")
SPOON_SETTING = RelativeTarget(relation="left_of", anchor="fork_1")

CANONICAL_INSTRUCTION = (
    "Open and close the top drawer, pick up the plate with arm A, place it on the table, "
    "pick up the mug with arm B, pour water into the mug with arm A."
)

_CANONICAL = TaskGraph(
    task_id="dinner_canonical",
    instruction=CANONICAL_INSTRUCTION,
    steps=[
        Step(id=1, skill="open_drawer", arm="A", object="drawer_top"),
        Step(id=2, skill="close_drawer", arm="A", object="drawer_top"),
        Step(id=3, skill="pick", arm="A", object="plate"),
        Step(id=4, skill="place", arm="A", object="plate", target="placemat_1"),
        Step(id=5, skill="pick", arm="B", object="mug"),
        Step(id=6, skill="pick", arm="A", object="bottle"),
        Step(id=7, skill="hold", arm="B", object="mug", parallel_group=1),
        Step(id=8, skill="pour", arm="A", object="bottle", target="mug",
             amount=0.6, parallel_group=1),
    ],
)

_FULL = TaskGraph(
    task_id="dinner_full",
    instruction=(
        "Set the dinner table: open the drawer, lay the plate and the cutlery, "
        "set the mug, pass the bottle between the arms, pour water into the mug "
        "while the other arm steadies it, then close the drawer."
    ),
    steps=[
        Step(id=1, skill="open_drawer", arm="A", object="drawer_top"),
        Step(id=2, skill="pick", arm="A", object="plate"),
        Step(id=3, skill="place", arm="A", object="plate", target="placemat_1"),
        Step(id=4, skill="pick", arm="A", object="fork_1"),
        Step(id=5, skill="place", arm="A", object="fork_1", target=FORK_SETTING),
        Step(id=6, skill="pick", arm="A", object="spoon_1"),
        Step(id=7, skill="place", arm="A", object="spoon_1", target=SPOON_SETTING),
        Step(id=8, skill="close_drawer", arm="A", object="drawer_top"),
        Step(id=9, skill="pick", arm="A", object="bottle"),
        Step(id=10, skill="handoff", arm="A", object="bottle", target="hand_of_B"),
        Step(id=11, skill="handoff", arm="B", object="bottle", target="hand_of_A"),
        Step(id=12, skill="pick", arm="B", object="mug"),
        Step(id=13, skill="hold", arm="B", object="mug", parallel_group=1),
        Step(id=14, skill="pour", arm="A", object="bottle", target="mug",
             amount=0.6, parallel_group=1),
        Step(id=15, skill="place", arm="B", object="mug", target="placemat_2"),
    ],
)

_DRAWER = TaskGraph(
    task_id="drawer_cycle",
    instruction="Open the top drawer, then close it again.",
    steps=[
        Step(id=1, skill="open_drawer", arm="A", object="drawer_top"),
        Step(id=2, skill="close_drawer", arm="A", object="drawer_top"),
    ],
)

_PLATE = TaskGraph(
    task_id="plate_setting",
    instruction="Pick up the plate with arm A and put it on the left place setting.",
    steps=[
        Step(id=1, skill="pick", arm="A", object="plate"),
        Step(id=2, skill="place", arm="A", object="plate", target="placemat_1"),
    ],
)

_MUG = TaskGraph(
    task_id="mug_setting",
    instruction="Pick up the mug with arm B and put it on the right place setting.",
    steps=[
        Step(id=1, skill="pick", arm="B", object="mug"),
        Step(id=2, skill="place", arm="B", object="mug", target="placemat_2"),
    ],
)

_CUTLERY = TaskGraph(
    task_id="cutlery_pair",
    instruction="Open the drawer and lay a fork and a spoon beside the plate.",
    steps=[
        Step(id=1, skill="open_drawer", arm="A", object="drawer_top"),
        Step(id=2, skill="pick", arm="A", object="plate"),
        Step(id=3, skill="place", arm="A", object="plate", target="placemat_1"),
        Step(id=4, skill="pick", arm="A", object="fork_1"),
        Step(id=5, skill="place", arm="A", object="fork_1", target=FORK_SETTING),
        Step(id=6, skill="pick", arm="A", object="spoon_1"),
        Step(id=7, skill="place", arm="A", object="spoon_1", target=SPOON_SETTING),
        Step(id=8, skill="close_drawer", arm="A", object="drawer_top"),
    ],
)

_RELAY = TaskGraph(
    task_id="bottle_relay",
    instruction="Pick up the bottle with arm A, pass it to arm B, and pass it back.",
    steps=[
        Step(id=1, skill="pick", arm="A", object="bottle"),
        Step(id=2, skill="handoff", arm="A", object="bottle", target="hand_of_B"),
        Step(id=3, skill="handoff", arm="B", object="bottle", target="hand_of_A"),
        Step(id=4, skill="place", arm="A", object="bottle",
             target=RelativeTarget(relation="left_of", anchor="mug")),
    ],
)

_POUR = TaskGraph(
    task_id="pour_only",
    instruction="Hold the mug with arm B and pour water into it from the bottle with arm A.",
    steps=[
        Step(id=1, skill="pick", arm="B", object="mug"),
        Step(id=2, skill="pick", arm="A", object="bottle"),
        Step(id=3, skill="hold", arm="B", object="mug", parallel_group=1),
        Step(id=4, skill="pour", arm="A", object="bottle", target="mug",
             amount=0.6, parallel_group=1),
    ],
)

CANONICAL_GRAPHS: dict[str, TaskGraph] = {
    "dinner_canonical": _CANONICAL,
    "dinner_full": _FULL,
    "drawer_cycle": _DRAWER,
    "plate_setting": _PLATE,
    "mug_setting": _MUG,
    "cutlery_pair": _CUTLERY,
    "bottle_relay": _RELAY,
    "pour_only": _POUR,
}
