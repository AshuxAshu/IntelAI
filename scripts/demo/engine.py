"""Execute a demonstration run against the teacher skills; no rendering.

Shared by the capability probe and the video renderer so both agree on what a
run is and how its per-step outcome is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dinner_table.scene.builder import Scene
from dinner_table.teacher.context import SkillFailed, TeacherContext
from dinner_table.teacher.skills import CloseDrawer, Hold, OpenDrawer, Pick, Place

# Generous: a cutlery run with a fail-and-retry is the longest sequence here.
RUN_TIMEOUT_S = 900.0


@dataclass
class StepResult:
    skill: str
    arm: str
    object: str | None
    target: object
    outcome: str = "pending"      # success | failed | skipped
    phase: str = ""
    cause: str = ""

    @property
    def label(self) -> str:
        if self.object:
            return f"{self.skill} {self.object}"
        return self.skill


@dataclass
class RunResult:
    run: str
    seed: int
    profile: str
    steps: list[StepResult] = field(default_factory=list)
    success: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "run": self.run,
            "seed": self.seed,
            "profile": self.profile,
            "success": self.success,
            "error": self.error,
            "steps": [
                {
                    "skill": s.skill, "arm": s.arm, "object": s.object,
                    "target": str(s.target) if s.target is not None else None,
                    "outcome": s.outcome, "phase": s.phase, "cause": s.cause,
                }
                for s in self.steps
            ],
        }


def build_skill(step: tuple):
    """Instantiate the teacher skill a run step names."""
    skill, arm, obj, target = step
    if skill == "pick":
        return Pick(arm, obj)
    if skill == "place":
        return Place(arm, obj, target)
    if skill == "open_drawer":
        return OpenDrawer(arm)
    if skill == "close_drawer":
        return CloseDrawer(arm)
    if skill == "hold":
        return Hold(arm, obj)
    raise ValueError(f"unknown skill in run step: {skill}")


def make_scene(seed: int, profile: str) -> Scene:
    """A settled scene for this seed, ready to be driven by skills."""
    scene = Scene(seed=seed, dr_profile=profile)
    scene.hold_safe()
    return scene


def execute_run(run_name: str, steps: list[tuple], seed: int,
                profile: str = "dr_train", scene: Scene | None = None,
                on_phase=None, on_step=None, on_tick=None):
    """Drive every step in order on ONE scene; never raises.

    ``on_tick()`` is called after every physics step (the renderer's frame
    hook), and ``on_phase(skill, phase, scene)`` whenever a skill's phase
    changes; ``on_step(index, result)`` runs after each step completes.
    """
    scene = scene if scene is not None else make_scene(seed, profile)
    ctx = TeacherContext(scene)
    ctx.begin(RUN_TIMEOUT_S)
    result = RunResult(run=run_name, seed=seed, profile=profile)

    step_results: list[StepResult] = []
    for idx, step in enumerate(steps):
        skill_name, arm, obj, target = step
        rec = StepResult(skill=skill_name, arm=arm, object=obj, target=target)
        step_results.append(rec)
        skill = build_skill(step)
        last_phase = None
        try:
            for action in skill.run(ctx):
                ctx.step(action)
                if on_phase is not None and skill.phase != last_phase:
                    last_phase = skill.phase
                    on_phase(skill_name, skill.phase, scene)
                if on_tick is not None:
                    on_tick()
            rec.outcome = "success"
        except SkillFailed as exc:
            rec.outcome = "failed"
            rec.phase, rec.cause = exc.phase, exc.cause
            if on_tick is not None:
                on_tick()
            if on_step is not None:
                on_step(idx, rec)
            # A failed step ends the run: later steps assume its postcondition.
            for later in steps[idx + 1:]:
                step_results.append(StepResult(
                    skill=later[0], arm=later[1], object=later[2],
                    target=later[3], outcome="skipped",
                ))
            break
        if on_step is not None:
            on_step(idx, rec)

    result.steps = step_results
    result.success = all(s.outcome == "success" for s in step_results)
    return result, scene, ctx
