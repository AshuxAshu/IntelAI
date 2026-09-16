"""Mid-episode perturbations so demonstrations include teacher recovery.

The teacher is closed-loop: it re-plans from live state and retries failed
steps. Firing a physical event mid-episode (a kicked object, a slipped grip, a
stiff drawer) records the recovery in the action history, so the student learns
correction instead of trajectory replay. Every event is precomputed in
``schedule`` (the run stays a pure function of the episode seed) and annotated
onto the frame it fires on for the VQA factory and the failure taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.reasoning.schema import TaskGraph

GRASPABLE_OBJECTS = ("plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2")


class PerturberError(DinnerTableError):
    """Exception raised for un-schedulable or un-appliable perturbation events."""


@dataclass(frozen=True)
class Event:
    """One precomputed perturbation: what fires, on which tick, with what payload."""

    name: str
    tick: int
    object: str | None = None
    arm: str | None = None
    vector: tuple[float, ...] = ()


class Perturber:
    """Schedule mid-episode physical events and apply them to a live context."""

    EVENTS = (
        "object_kick",
        "gripper_slip",
        "waypoint_jitter",
        "mid_skill_retarget",
        "drawer_friction_spike",
    )
    KICK_IMPULSE_NS = 0.05
    KICK_DV_MAX = 0.5
    SLIP_SIGMA_DEG = 1.5
    JITTER_SIGMA_CM = 2.0
    RETARGET_CM = 3.0
    DAMPING_FACTOR = (2.0, 3.0)
    FIRST_TICK = (50, 600)
    SECOND_TICK = (400, 1500)
    SECOND_EVENT_PROBABILITY = 0.25

    def __init__(self, events: list[Event] | None = None) -> None:
        """Attach an event list; ``schedule`` builds one for an episode."""
        self._events = []
        if events is not None:
            self._events = list(events)
        self._fired = 0
        self._rng: np.random.Generator | None = None

    def schedule(
        self, rng: np.random.Generator, graph: TaskGraph, probability: float = 0.35
    ) -> list[Event]:
        """Precompute this episode's events; ``probability`` is the per-episode rate."""
        self._rng = rng
        self._events = []
        self._fired = 0
        if rng.random() >= probability:
            return []
        objects = [o for o in self._graph_objects(graph) if o in GRASPABLE_OBJECTS]
        touches_drawer = any(s.skill in ("open_drawer", "close_drawer") for s in graph.steps)
        first = int(rng.integers(self.FIRST_TICK[0], self.FIRST_TICK[1] + 1))
        self._events.append(self._draw(rng, graph, objects, touches_drawer, first))
        if rng.random() < self.SECOND_EVENT_PROBABILITY:
            second = int(rng.integers(self.SECOND_TICK[0], self.SECOND_TICK[1] + 1))
            self._events.append(self._draw(rng, graph, objects, touches_drawer, second))
        return list(self._events)

    def poll(self, ctx) -> list[Event]:
        """Fire every event due at the context's current tick; returns fired events."""
        fired = []
        while self._fired < len(self._events) and self._events[self._fired].tick <= ctx.ticks():
            event = self._events[self._fired]
            self._fired += 1
            self.apply(event, ctx.data, ctx)
            fired.append(event)
        return fired

    def apply(self, event: Event, data, ctx) -> None:
        """Apply one precomputed event to the live simulation."""
        if event.name == "object_kick":
            self._kick(ctx, event)
        elif event.name == "gripper_slip":
            ctx.perturb_gripper(event.arm, event.vector[0])
        elif event.name == "waypoint_jitter":
            if self._rng is None:
                raise PerturberError("waypoint_jitter needs the schedule-time rng")
            ctx.jitter_teacher_waypoints(self.JITTER_SIGMA_CM, self._rng)
        elif event.name == "mid_skill_retarget":
            target = event.object
            if target in (ctx.carrying.get("A"), ctx.carrying.get("B")):
                unheld = [
                    o
                    for o in GRASPABLE_OBJECTS
                    if o not in (ctx.carrying.get("A"), ctx.carrying.get("B"))
                ]
                if not unheld:
                    return
                target = unheld[0]
            ctx.shift_object(target, event.vector[0], event.vector[1])
        elif event.name == "drawer_friction_spike":
            ctx.set_joint_damping("drawer_slide", event.vector[0])
        else:
            raise PerturberError(f"unknown perturbation event: {event.name}")

    def _kick(self, ctx, event: Event) -> None:
        # NOTE: a velocity kick, not the sketched persistent-force call: 0.05 N
        # sits below every object's static friction (a resting object would not
        # move at all) while never terminating (a held object would be dragged
        # until the grasp breaks). The impulse 0.05 N s slides the bottle ~3 cm.
        bid = ctx.object_body(event.object)
        mass = float(ctx.model.body_mass[bid])
        direction = np.array(event.vector, dtype=np.float64)
        dv = min(self.KICK_IMPULSE_NS / mass, self.KICK_DV_MAX)
        adr = int(ctx.model.jnt_dofadr[int(ctx.model.body_jntadr[bid])])
        ctx.data.qvel[adr : adr + 3] += direction * dv

    def _graph_objects(self, graph: TaskGraph) -> list[str]:
        names = []
        for step in graph.steps:
            if step.object is not None and step.object not in names:
                names.append(step.object)
            if step.skill == "pour" and step.target == "mug" and "mug" not in names:
                names.append("mug")
        return names

    def _graph_arms(self, graph: TaskGraph) -> list[str]:
        arms = []
        for step in graph.steps:
            if step.arm not in arms:
                arms.append(step.arm)
        return arms

    def _draw(
        self,
        rng: np.random.Generator,
        graph: TaskGraph,
        objects: list[str],
        touches_drawer: bool,
        tick: int,
    ) -> Event:
        choices = ["gripper_slip", "waypoint_jitter"]
        if objects:
            choices += ["object_kick", "mid_skill_retarget"]
        if touches_drawer:
            choices.append("drawer_friction_spike")
        name = str(rng.choice(choices))
        if name == "object_kick":
            target = str(rng.choice(objects))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            direction = (float(np.cos(angle)), float(np.sin(angle)), 0.0)
            return Event(name, tick, object=target, vector=direction)
        if name == "gripper_slip":
            arm = str(rng.choice(self._graph_arms(graph)))
            delta = float(rng.normal(0.0, np.deg2rad(self.SLIP_SIGMA_DEG)))
            return Event(name, tick, arm=arm, vector=(delta,))
        if name == "mid_skill_retarget":
            target = str(rng.choice(objects))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            dist = self.RETARGET_CM / 100.0
            return Event(
                name,
                tick,
                object=target,
                vector=(float(np.cos(angle) * dist), float(np.sin(angle) * dist)),
            )
        if name == "drawer_friction_spike":
            lo, hi = self.DAMPING_FACTOR
            return Event(name, tick, vector=(float(rng.uniform(lo, hi)),))
        return Event(name, tick)
