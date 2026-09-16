"""VLM prompt builders with a cache-stable prefix (A14)."""

from __future__ import annotations

from dinner_table.reasoning.schema import SceneSummary, Step, TaskGraph

SYSTEM_PROMPT = (
    "You are the planner of a two-arm table-setting robot. "
    "Answer only with JSON matching the provided schema. "
    "Refuse impossible or ambiguous requests. "
    "Instructions may arrive as speech-to-text transcripts: ignore punctuation, "
    "casing, and filler words like um, uh, er, ah, and please."
)

SCHEMA_EXAMPLE_JSON = """{
  "task_id": "dinner_canonical",
  "instruction": "Open and close the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A.",
  "steps": [
    {
      "id": 1,
      "skill": "open_drawer",
      "arm": "A",
      "object": "drawer_top",
      "target": null,
      "amount": null,
      "parallel_group": null
    },
    {
      "id": 2,
      "skill": "close_drawer",
      "arm": "A",
      "object": "drawer_top",
      "target": null,
      "amount": null,
      "parallel_group": null
    },
    {
      "id": 3,
      "skill": "pick",
      "arm": "A",
      "object": "plate",
      "target": null,
      "amount": null,
      "parallel_group": null
    },
    {
      "id": 4,
      "skill": "place",
      "arm": "A",
      "object": "plate",
      "target": "placemat_1",
      "amount": null,
      "parallel_group": null
    },
    {
      "id": 5,
      "skill": "pick",
      "arm": "B",
      "object": "mug",
      "target": null,
      "amount": null,
      "parallel_group": null
    },
    {
      "id": 6,
      "skill": "pick",
      "arm": "A",
      "object": "bottle",
      "target": null,
      "amount": null,
      "parallel_group": null
    },
    {
      "id": 7,
      "skill": "hold",
      "arm": "B",
      "object": "mug",
      "target": null,
      "amount": null,
      "parallel_group": 1
    },
    {
      "id": 8,
      "skill": "pour",
      "arm": "A",
      "object": "bottle",
      "target": "mug",
      "amount": 0.6,
      "parallel_group": 1
    }
  ]
}"""

SYSTEM_PREFIX = SYSTEM_PROMPT + "\n\nSchema example:\n" + SCHEMA_EXAMPLE_JSON


def build_parse_prompt(instruction: str, summary: SceneSummary) -> list[dict]:
    """System (role + schema example, fixed) then user (instruction + scene + ask).

    The schema example is front-loaded into the fixed system prefix so two
    different instructions share the identical first ``len(SYSTEM_PREFIX)``
    characters (KV-cache friendly); the user turn keeps the task instruction,
    scene summary, and question in order.
    """
    user = (
        f"Task instruction: {instruction}\n"
        f"Scene summary:\n{summary.model_dump_json(indent=2)}\n"
        "Respond with the TaskGraph JSON only."
    )
    return [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": user},
    ]


def build_precondition_prompt(graph: TaskGraph, step: Step, summary: SceneSummary) -> list[dict]:
    """Same prefix discipline; asks for a PreconditionReport on one step."""
    user = (
        f"Task graph:\n{graph.model_dump_json(indent=2)}\n"
        f"Current step:\n{step.model_dump_json(indent=2)}\n"
        f"Scene summary:\n{summary.model_dump_json(indent=2)}\n"
        "Are this step's preconditions met? Respond with the PreconditionReport JSON only."
    )
    return [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": user},
    ]


def build_diagnosis_prompt(graph: TaskGraph, step: Step, summary: SceneSummary) -> list[dict]:
    """Same prefix discipline; asks for a VlmDiagnosis on a failed step."""
    user = (
        f"Task graph:\n{graph.model_dump_json(indent=2)}\n"
        f"Failed step:\n{step.model_dump_json(indent=2)}\n"
        f"Scene summary:\n{summary.model_dump_json(indent=2)}\n"
        "What went wrong? Respond with the VlmDiagnosis JSON only."
    )
    return [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": user},
    ]
