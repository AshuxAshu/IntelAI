"""Failure taxonomy over episode / rollout logs (A12 / B12).

Pure-logic module: no MuJoCo, no torch. Inputs are plain dicts, either loaded
from ``demos/logs/*.json`` or produced by :meth:`EpisodeLog.to_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RolloutFrame:
    """One frame of a policy rollout: EpisodeFrame plus the predicted action."""

    tick: int
    joints: list
    action: list
    predicted_action: list
    skill: str
    phase: str
    goal_xyz: list
    event: str
    ctrl: float
    sat: float
    step_id: str


@dataclass(frozen=True)
class RolloutLog:
    """EpisodeLog layout plus a predicted action per frame.

    The teacher records :class:`EpisodeLog`; the B12 eval harness records this.
    """

    episode_id: str
    seed: int
    profile: str
    steps: list
    frames: list
    success: bool
    success_reason: str


@dataclass(frozen=True)
class TaxonomyReport:
    """Frozen aggregate over a set of logs. JSON shape is frozen."""

    episodes: int
    successes: int
    skills: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    episodes_with_event: dict = field(default_factory=dict)
    failures_by_event: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "skills": self.skills,
            "failures": self.failures,
            "episodes_with_event": self.episodes_with_event,
            "failures_by_event": self.failures_by_event,
            "profiles": self.profiles,
        }


def _asdict(log) -> dict:
    return log.to_dict() if hasattr(log, "to_dict") else log


def _profile(log: dict) -> str:
    return str(log.get("profile", log.get("dr_profile", "?")))


def attribute(logs: list) -> TaxonomyReport:
    """Aggregate per-skill / per-phase failure counts with causes.

    DR-trigger correlation: failures are split by whether the failed episode
    saw at least one perturbation event (non-empty frame ``event``), and by
    noise ``profile``.
    """
    skills: dict = {}
    failures: dict = {}
    with_ev = {"episodes": 0, "failures": 0}
    without_ev = {"episodes": 0, "failures": 0}
    by_event: dict = {}
    profiles: dict = {}
    successes = 0
    for raw in logs:
        log = _asdict(raw)
        if log.get("success"):
            successes += 1
        events = {f.get("event", "") for f in log.get("frames", [])} - {""}
        ep_failed = False
        for rec in log.get("steps", []):
            skill = rec.get("skill", "?")
            slot = skills.setdefault(skill, {"attempts": 0, "successes": 0})
            slot["attempts"] += 1
            if rec.get("outcome", "") == "success":
                slot["successes"] += 1
            else:
                ep_failed = True
                key = (skill, rec.get("phase_at_failure", "?"), rec.get("failure_cause", "?"))
                failures[key] = failures.get(key, 0) + 1
        bucket = with_ev if events else without_ev
        bucket["episodes"] += 1
        if ep_failed:
            bucket["failures"] += 1
            for ev in events:
                by_event[ev] = by_event.get(ev, 0) + 1
        prof = profiles.setdefault(_profile(log), {"episodes": 0, "failures": 0})
        prof["episodes"] += 1
        if ep_failed:
            prof["failures"] += 1
    skills_out = {
        name: {
            "attempts": s["attempts"],
            "successes": s["successes"],
            "rate": s["successes"] / s["attempts"] if s["attempts"] else 0.0,
        }
        for name, s in sorted(skills.items())
    }
    failures_out = [
        {"skill": k[0], "phase": k[1], "cause": k[2], "count": v}
        for k, v in sorted(failures.items())
    ]
    return TaxonomyReport(
        episodes=len(logs),
        successes=successes,
        skills=skills_out,
        failures=failures_out,
        episodes_with_event={"with_event": with_ev, "without_event": without_ev},
        failures_by_event=dict(sorted(by_event.items())),
        profiles=dict(sorted(profiles.items())),
    )


def chunk_error_curve(rollout: dict, chunk_size: int = 100) -> list:
    """Mean-squared predicted-vs-realized action error by chunk position.

    Chunks are implicit: frame ``i`` sits at position ``i % chunk_size``.
    Returns one MSE value per position, averaged over all chunks.
    """
    frames = rollout.get("frames", [])
    if not frames:
        return [0.0] * chunk_size
    acc = [0.0] * chunk_size
    cnt = [0] * chunk_size
    for i, fr in enumerate(frames):
        pos = i % chunk_size
        pred = fr["predicted_action"]
        act = fr["action"]
        mse = sum((p - a) ** 2 for p, a in zip(pred, act)) / len(act)
        acc[pos] += mse
        cnt[pos] += 1
    return [a / c if c else 0.0 for a, c in zip(acc, cnt)]


def confusion(rollouts: list) -> dict:
    """Count executed-vs-intended skill mismatches per frame.

    Intended skill = the step record's skill for the frame's ``step_id``.
    Executed skill = the frame's skill, reduced to its primary (the part
    before any ``+`` composite marker). Each mismatch counts one vote for
    ``"intended->executed"``.
    """
    votes: dict = {}
    for raw in rollouts:
        log = _asdict(raw)
        intended = {rec.get("step_id"): rec.get("skill", "?") for rec in log.get("steps", [])}
        for fr in log.get("frames", []):
            want = intended.get(fr.get("step_id"), "?")
            got = str(fr.get("skill", "?")).split("+")[0]
            if got != want:
                key = f"{want}->{got}"
                votes[key] = votes.get(key, 0) + 1
    return dict(sorted(votes.items()))
