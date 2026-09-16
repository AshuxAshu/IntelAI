"""Canonical demonstration runs: named, ordered skill sequences.

A "run" is one scene, one TeacherContext, and an ordered list of steps. It is
what the per-seed demonstration videos render: each entry groups whatever
combines safely into one continuous episode.

Step = (skill, arm, object, target):
  pick / place use ``object`` and ``target``; open_drawer / close_drawer / hold
  use ``arm`` (and ``object`` for hold). ``target`` is a PlacementTarget name or
  an (x, y) world point.

Composition is dictated by measurement, not preference. Discovered per-seed on
this HEAD (scripts/demo/probe_runs.py):

  * plate -> placemat_1 and mug -> placemat_2 SUCCEED, and they also succeed as
    ONE episode (different arms, landing sites 0.33 m apart) -- so the two are
    combined into the single "table_setting" run.
  * bottle pick succeeds from any seed, but bottle PLACE fails on every target
    tried (a 6x4 grid plus all named settings): the neck pinch cannot be
    re-oriented over the table. The bottle therefore gets a pick-and-lift run.
  * open_drawer + pick fork + place fork succeeds; Place's cutlery path
    servo-closes the drawer itself, so a trailing close_drawer must NOT be
    added (it fails pregrass/already_closed because the drawer is shut).
  * open_drawer + plate place FAILS (the open drawer's front wall overlaps
    placemat_1), which is why the drawer run cannot include the plate.
  * spoon place is unstable (verify/misplaced, or the path clips the plate).
"""

from __future__ import annotations

# Arm assignments follow the scene layout: A works the left/cutlery half (it
# owns the drawer and placemat_1), B the right half (placemat_2). The bottle
# spawns at x=+0.11 m, i.e. on B's side of centre, but only A can grasp it.
PLATE_STEPS = [("pick", "A", "plate", None),
               ("place", "A", "plate", "placemat_1")]
MUG_STEPS = [("pick", "B", "mug", None),
             ("place", "B", "mug", "placemat_2")]

RUNS: dict[str, list[tuple]] = {
    # Both place settings in one continuous episode: the headline run.
    "table_setting": PLATE_STEPS + MUG_STEPS,
    # Each setting on its own, so a single-object view is available too.
    "plate": PLATE_STEPS,
    "mug": MUG_STEPS,
    # Bottle: pick and hold aloft. Placement is not supported on this HEAD, and
    # a trailing Hold step is also rejected (Hold's fumble audit reads the
    # settled rim pinch as a lost grasp), so the pick's own end-hold is the
    # demonstration.
    "bottle": [
        ("pick", "A", "bottle", None),
    ],
    # Drawer cycle, both utensils in one episode: open, retrieve the fork and
    # lay it out (Place's cutlery path servo-closes the drawer itself), re-open,
    # then retrieve the spoon. Spoon PLACEMENT fails on every target tried, so
    # this run ends holding it -- the drawer cycle and the retrieval are the
    # demonstrated behaviour.
    "drawer": [
        ("open_drawer", "A", None, None),
        ("pick", "A", "fork_1", None),
        ("place", "A", "fork_1", "fork_setting"),
        ("open_drawer", "A", None, None),
        ("pick", "A", "spoon_1", None),
    ],
    # The fork alone, end to end, including the drawer closing.
    "drawer_fork": [
        ("open_drawer", "A", None, None),
        ("pick", "A", "fork_1", None),
        ("place", "A", "fork_1", "fork_setting"),
    ],
    # --- Attempt runs: known-unsupported on this HEAD, rendered so the
    # limitation is visible on camera rather than only asserted in prose. Their
    # videos are expected to carry the FAILED marking.
    "attempt_spoon_place": [
        ("open_drawer", "A", None, None),
        ("pick", "A", "spoon_1", None),
        ("place", "A", "spoon_1", "spoon_setting"),
    ],
    "attempt_bottle_place": [
        ("pick", "A", "bottle", None),
        ("place", "A", "bottle", "spoon_setting"),
    ],
}

# Runs that are expected to fail on every seed; the manifest marks them
# `expected_failure` so a red FAILED overlay is read as the documented
# limitation, not as a regression.
EXPECTED_FAILURE: frozenset[str] = frozenset(
    {"attempt_spoon_place", "attempt_bottle_place"}
)

# The object each run centres on, for manifests (None = multi-object episode).
PRIMARY_OBJECT: dict[str, str | None] = {
    "table_setting": None,
    "plate": "plate",
    "mug": "mug",
    "bottle": "bottle",
    "drawer": "fork_1+spoon_1",
    "drawer_fork": "fork_1",
    "attempt_spoon_place": "spoon_1",
    "attempt_bottle_place": "bottle",
}

# What the run demonstrates, for the manifest/report (never burned into video).
RUN_TITLE: dict[str, str] = {
    "table_setting": "plate + mug placed on their settings",
    "plate": "plate placed on placemat_1",
    "mug": "mug placed on placemat_2",
    "bottle": "bottle picked and held aloft",
    "drawer": "drawer opened, fork retrieved and laid out, re-opened, spoon retrieved",
    "drawer_fork": "drawer opened, fork retrieved and laid out (drawer closed)",
    "attempt_spoon_place": "ATTEMPT (unsupported): place the spoon on spoon_setting",
    "attempt_bottle_place": "ATTEMPT (unsupported): place the bottle",
}
