"""Bounded-grammar fallback parser (A14).

Port of the reference's ``parse_command``
(``example-approach/simulation_lab/language.py``, MIT — see
``example-approach/simulation_lab/NOTICE.md``), re-expressed to emit our
:class:`TaskGraph`. Keep the reference behaviors exactly: normalization, alias
table, 8-clause cap, negation refusal, verb/destination grammar, ``"it"``
chaining, and compound-clause refusal.

Deliberate extensions beyond the reference (forced by our schema and the A14
corpus): ASR filler words (um/uh/er/ah) are stripped, the ASR
sentence-boundary join covers the full verb set, bare ``beside`` is accepted
as a relative relation, and ``with arm A|B`` / ``to arm A|B`` are accepted
alongside the spec'd ``left|right arm`` forms.
Front/behind relations, center spots, and objects outside our world (glass,
side plate) are explicit refusals — our ``RelativeTarget`` / ``PlacementTarget``
vocabularies cannot express them. Bare ``fork``/``spoon`` resolve to the first
instance (``fork_1``/``spoon_1``); the executor's preconditions verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dinner_table.reasoning.schema import RelativeTarget, Step, TaskGraph

TASK_ID = "fallback"


class FallbackError(ValueError):
    """Base for explicit fallback outcomes that are not graphs."""


class FallbackRefusal(FallbackError):
    """The instruction is refused with a human-readable message."""


class FallbackCancel(FallbackError):
    """Stop/cancel control flow."""

    control = "cancel"


ALIASES = {
    "water bottle": "bottle",
    "pottle": "bottle",
    "flasche": "bottle",
    "teller": "plate",
    "cup": "mug",
    "tasse": "mug",
    "gabel": "fork",
    "löffel": "spoon",
    "loeffel": "spoon",
    "schublade": "drawer",
    "side plate": "side_plate",
    "gold plate": "side_plate",
    "blue plate": "plate",
}
ITEMS = (
    "side_plate",
    "bottle",
    "plate",
    "mug",
    "glass",
    "fork",
    "spoon",
    "fork_1",
    "fork_2",
    "spoon_1",
    "spoon_2",
)
FILLERS = ("um", "uh", "er", "ah")
OBJECTS = {
    "plate": "plate",
    "mug": "mug",
    "bottle": "bottle",
    "fork": "fork_1",
    "spoon": "spoon_1",
    "fork_1": "fork_1",
    "fork_2": "fork_2",
    "spoon_1": "spoon_1",
    "spoon_2": "spoon_2",
}
FORK_HOME = RelativeTarget(relation="left_of", anchor="plate")
SPOON_HOME = RelativeTarget(relation="left_of", anchor="fork_1")
HOME_TARGETS = {
    "plate": "placemat_1",
    "mug": "placemat_2",
    "fork_1": FORK_HOME,
    "fork_2": FORK_HOME,
    "spoon_1": SPOON_HOME,
    "spoon_2": SPOON_HOME,
}
RELATIONS = {"left": "left_of", "right": "right_of", "beside": "beside"}
SET_TABLE_STEPS = [
    Step(id=1, skill="open_drawer", arm="A", object="drawer_top"),
    Step(id=2, skill="pick", arm="A", object="plate"),
    Step(id=3, skill="place", arm="A", object="plate", target="placemat_1"),
    Step(id=4, skill="pick", arm="A", object="fork_1"),
    Step(id=5, skill="place", arm="A", object="fork_1", target=FORK_HOME),
    Step(id=6, skill="pick", arm="A", object="spoon_1"),
    Step(id=7, skill="place", arm="A", object="spoon_1", target=SPOON_HOME),
    Step(id=8, skill="close_drawer", arm="A", object="drawer_top"),
    Step(id=9, skill="pick", arm="A", object="bottle"),
    Step(id=10, skill="handoff", arm="A", object="bottle", target="hand_of_B"),
    Step(id=11, skill="handoff", arm="B", object="bottle", target="hand_of_A"),
    Step(id=12, skill="pick", arm="B", object="mug"),
    Step(id=13, skill="hold", arm="B", object="mug", parallel_group=1),
    Step(
        id=14,
        skill="pour",
        arm="A",
        object="bottle",
        target="mug",
        amount=0.6,
        parallel_group=1,
    ),
    Step(id=15, skill="place", arm="B", object="mug", target="placemat_2"),
]


@dataclass
class _Intent:
    kind: str  # drawer_open | dinner_place
    object: str | None = None  # resolved ObjectName
    arm: str = "auto"  # A | B | auto
    destination: dict | None = None


def _resolve_object(item: str) -> str:
    try:
        return OBJECTS[item]
    except KeyError:
        raise FallbackRefusal(f"Unknown object: {item}.") from None


def _normalize(instruction: str) -> tuple[str, str]:
    original = instruction.strip()
    if not 1 <= len(original) <= 500:
        raise FallbackRefusal("Use a short table-setting instruction (1–500 characters).")
    s = re.sub(r"\s+", " ", original.lower()).strip(" .!?")
    s = re.sub(r"\s+([,.!?;])", r"\1", s)
    for filler in FILLERS:
        s = re.sub(r"\b" + filler + r"\b", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.!?")
    for phrase, value in sorted(ALIASES.items(), key=lambda r: -len(r[0])):
        s = re.sub(r"\b" + re.escape(phrase) + r"\b", value, s)
    joinables = "|".join((*ITEMS, "drawer"))
    s = re.sub(
        r"\b(put|place|move|take|pick up|bring|hand|pass|transfer|open)[.!?]\s+"
        rf"(?=(?:the )?(?:{joinables})\b)",
        r"\1 ",
        s,
    )
    return original, s


def _parse_clause(clause: str, context: str | None) -> tuple[_Intent, str]:
    clause = re.sub(r"^(?:and then|then)[,\s]+", "", clause).strip(" ,.!?")
    clause = re.sub(r"^please,\s*", "", clause)
    if re.fullmatch(r"(?:please )?open (?:the )?(?:top )?drawer", clause):
        return _Intent("drawer_open"), context
    if not re.match(
        r"^(?:please )?(?:put|place|move|take|pick up|transfer|hand|pass|bring)\b", clause
    ):
        raise FallbackRefusal(
            "Try “place the plate”, “open the drawer”, "
            "or “put the bottle in front of the right plate”."
        )
    found = re.search(r"\b(" + "|".join(ITEMS) + r")\b", clause)
    item = found.group(1) if found else context if re.search(r"\bit\b", clause) else None
    if item is None:
        raise FallbackRefusal("Name the item to move.")
    obj = _resolve_object(item)
    tail = clause[found.end() :] if found else clause
    if re.search(r"\b(under|underneath|over|above|inside)\b", tail):
        raise FallbackRefusal("That spatial relation is not supported.")
    arm_match = re.search(r"\b(?:with|using) (?:the )?(left|right) arm\b", tail)
    arm_letter = re.search(r"\b(?:with|using) arm (a|b)\b", tail)
    if arm_match:
        arm = "A" if arm_match.group(1) == "left" else "B"
        tail = tail[: arm_match.start()] + tail[arm_match.end() :]
    elif arm_letter:
        arm = arm_letter.group(1).upper()
        tail = tail[: arm_letter.start()] + tail[arm_letter.end() :]
    else:
        arm = "auto"
    destination: dict = {"kind": "default"}
    relative = re.search(
        r"\b(in front of|infront of|behind|to the left of|to the right of|next to|beside) "
        r"(?:the )?(?:(left|right) )?(" + "|".join(ITEMS) + r")\b",
        tail,
    )
    if relative:
        relation = {
            "in front of": "front",
            "infront of": "front",
            "behind": "behind",
            "to the left of": "left",
            "to the right of": "right",
            "next to": "beside",
            "beside": "beside",
        }[relative.group(1)]
        if relation not in RELATIONS:
            raise FallbackRefusal("Only left/right/beside placements are supported.")
        if relative.group(2):
            raise FallbackRefusal("Ambiguous reference: name a single object.")
        reference = _resolve_object(relative.group(3))
        if reference == obj:
            raise FallbackRefusal("The destination must refer to another item.")
        destination = {"kind": "relative", "relation": relation, "reference": reference}
    elif spot := re.search(
        r"\b(?:(far) )?(left|right|center|middle) (?:spot|place|setting|side)\b", tail
    ):
        side = spot.group(2)
        if side in ("center", "middle"):
            raise FallbackRefusal("Only left/right spots are supported.")
        destination = {"kind": "spot", "side": side}
    elif (
        re.search(r"\b(?:to|in|on|into|under|over)\b", tail)
        and not re.search(
            r"\b(?:its|the) (?:spot|place|setting|table)\b|\bout of (?:the )?drawer\b", tail
        )
        and not (
            re.search(r"\b(hand|pass|transfer)\b", clause)
            and re.search(r"\b(?:(left|right) arm|arm (a|b))\b", tail)
        )
    ):
        raise FallbackRefusal(
            "I cannot resolve that destination. Use a left/right spot or a relation to a named item."
        )
    handoff_recipient = None
    if re.search(r"\b(hand|pass|transfer)\b", clause):
        recipient = re.search(r"\b(?:(left|right) arm|arm (a|b))\b", tail)
        if recipient:
            side = recipient.group(1) or ("left" if recipient.group(2) == "a" else "right")
            handoff_recipient = "hand_of_A" if side == "left" else "hand_of_B"
            destination = {"kind": "handoff", "recipient": handoff_recipient}
    # Drawer retrieval is one intent; the executor inserts opening only if needed.
    # (from_drawer has no schema field: the executor's drawer-open precondition
    # covers it, so it is intentionally dropped here.)
    compound = re.search(r"\band\s+(?:put|place|move|take|pick|transfer)\b", tail)
    # “take X out of the drawer and put it ...” describes one move.
    if compound and not re.search(r"\band\s+(?:put|place)\s+it\b", tail):
        raise FallbackRefusal("Separate different movements with “then”.")
    return _Intent("dinner_place", obj, arm, destination), item


def _intent_to_steps(intent: _Intent, first_id: int) -> list[Step]:
    if intent.kind == "drawer_open":
        return [Step(id=first_id, skill="open_drawer", arm="A", object="drawer_top")]
    dest = intent.destination or {"kind": "default"}
    if dest["kind"] == "handoff":
        recipient = dest["recipient"]
        source = intent.arm if intent.arm != "auto" else ("B" if recipient == "hand_of_A" else "A")
        if source == recipient[-1]:
            raise FallbackRefusal("Handoff source and destination arms must differ.")
        return [
            Step(id=first_id, skill="pick", arm=source, object=intent.object),
            Step(
                id=first_id + 1, skill="handoff", arm=source, object=intent.object, target=recipient
            ),
        ]
    arm = intent.arm if intent.arm != "auto" else "A"
    if dest["kind"] == "default" and intent.object not in HOME_TARGETS:
        return [Step(id=first_id, skill="pick", arm=arm, object=intent.object)]
    if dest["kind"] == "relative":
        target = RelativeTarget(relation=RELATIONS[dest["relation"]], anchor=dest["reference"])
    elif dest["kind"] == "spot":
        target = "placemat_1" if dest["side"] == "left" else "placemat_2"
    else:
        target = HOME_TARGETS[intent.object]
    return [
        Step(id=first_id, skill="pick", arm=arm, object=intent.object),
        Step(id=first_id + 1, skill="place", arm=arm, object=intent.object, target=target),
    ]


def template_parse(instruction: str) -> TaskGraph | None:
    """Parse one instruction with the bounded grammar.

    Returns None only for empty input. Unsupported language raises
    :class:`FallbackRefusal` with a message; stop/cancel raises
    :class:`FallbackCancel`. Never guesses a graph.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    original, s = _normalize(instruction)
    if re.search(r"\b(don't|do not|never|nicht)\b", s):
        raise FallbackRefusal("Negated movement instructions are not executed. Say stop to cancel.")
    if s in ("stop", "cancel", "stop moving", "cancel the task", "halt", "stopp"):
        raise FallbackCancel("Stop requested by user.")
    if s in (
        "set the table",
        "set up the table",
        "set the dinner table",
        "arrange the table",
        "deck den tisch",
    ):
        return TaskGraph(task_id=TASK_ID, instruction=original, steps=list(SET_TABLE_STEPS))
    clauses = re.split(r"\s*(?:;|,?\s+(?:and then|then)|\.\s+)\s*", s)
    if len(clauses) > 8:
        raise FallbackRefusal("Use at most eight task steps.")
    steps: list[Step] = []
    context: str | None = None
    for clause in clauses:
        clause = re.sub(r"^(?:and then|then)[,\s]+", "", clause).strip(" ,.!?")
        if not clause:
            continue
        intent, context = _parse_clause(clause, context)
        steps.extend(_intent_to_steps(intent, len(steps) + 1))
    if not steps:
        raise FallbackRefusal("No supported action found.")
    return TaskGraph(task_id=TASK_ID, instruction=original, steps=steps)
