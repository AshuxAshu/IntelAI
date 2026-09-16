"""Teacher execution context: stepping, motion playback, forces, and audits.

The machinery is ported from the reference solution's proven teacher (see
docs/PLAN_AMENDMENTS.md): quintic time-scaled waypoint playback with a
velocity-gradient duration floor, Cartesian-densified moves at 2 mm spacing,
contact-audited paths in a scratch MjData (carried objects teleported along
the gripper), force-monitored grasps (both jaws), and software torque
saturation of the gripper position servo.
"""

from __future__ import annotations

from contextlib import contextmanager

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import CONTROL_HZ, DRAWER_TRAVEL, PHYSICS_HZ
from dinner_table.teacher.ik import solve_ik
from dinner_table.teacher.kinematics import arm_q, site_pose
from dinner_table.teacher.live import LivePublisher

TICK = 1.0 / CONTROL_HZ
ARM_NAMES = ("A", "B")
ARM_JOINT_SUFFIXES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CONTACT_PENETRATION_TOL = -0.0008
JAW_FORCE_MIN = 0.08
SUPPORT_BODIES = ("table", "drawer_top", "cabinet", "world")
SKILL_TIMEOUT_S = 75.0


class SkillFailed(DinnerTableError):
    """Raised when a skill cannot complete; phase and cause are attributable."""

    def __init__(self, skill: str, phase: str, cause: str) -> None:
        self.skill = skill
        self.phase = phase
        self.cause = cause
        super().__init__(f"{skill} failed in phase {phase}: {cause}")


def _smooth(a: float) -> float:
    a = float(np.clip(a, 0.0, 1.0))
    return a**3 * (10.0 + a * (-15.0 + 6.0 * a))


class TeacherContext:
    """Drives the scene for one episode: stepping, motion, audits, forces."""

    def __init__(self, scene) -> None:
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.scratch = mujoco.MjData(self.model)
        self.live = LivePublisher(scene)
        self.carrying: dict[str, str | None] = {"A": None, "B": None}
        self.allowed: dict[str, set[str]] = {"A": set(), "B": set()}
        self.drawer_neutral = False  # hold the drawer servo error at zero (physical pulls)
        self._hold: dict[str, np.ndarray] = {}
        self._grip_now: dict[str, float] = {"A": 0.43, "B": 0.43}
        # Active torque-limited closes: arm -> (torque_limit N m, ctrl target rad).
        # The saturated ctrl advances per PHYSICS STEP (the reference engine's
        # bandwidth); a 25 Hz update is ~8x too slow at our 500 Hz timestep.
        self._grip_close: dict[str, tuple[float, float] | None] = {"A": None, "B": None}
        self._bad_grip_since: dict[str, float | None] = {"A": None, "B": None}
        self._deadline = np.inf
        self._cache_ids()

    def _cache_ids(self) -> None:
        m = self.model
        self._jadr = {}
        self._gadr = {}
        self._gact = {}
        for arm in ARM_NAMES:
            self._jadr[arm] = {
                s: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}.{s}")])
                for s in ARM_JOINT_SUFFIXES + ("gripper",)
            }
            self._gadr[arm] = {
                s: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}.{s}")])
                for s in ARM_JOINT_SUFFIXES + ("gripper",)
            }
            self._gact[arm] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}.gripper")
        self._arm_geoms: dict[str, set[int]] = {arm: set() for arm in ARM_NAMES}
        self._jaw_geoms: dict[str, set[int]] = {arm: set() for arm in ARM_NAMES}
        for i in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or ""
            for arm in ARM_NAMES:
                if body.startswith(f"{arm}."):
                    self._arm_geoms[arm].add(i)
        # Jaw geoms are classified by BODY, not name: the official model's
        # jaw collision meshes are unnamed, and a name-prefix match would
        # leave them out — their contact force on a clamped object would
        # then read as external support and fail the pick verify (measured:
        # 0.13 N from the mesh at the 2.94 N m clamp).
        for arm in ARM_NAMES:
            jaw_bodies: set[int] = set()
            for i in range(m.ngeom):
                gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                if gname.startswith(f"{arm}.fixed_jaw") or gname.startswith(
                    f"{arm}.moving_jaw"
                ):
                    jaw_bodies.add(int(m.geom_bodyid[i]))
            for i in range(m.ngeom):
                if int(m.geom_bodyid[i]) in jaw_bodies:
                    self._jaw_geoms[arm].add(i)
        drawer_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        self._drawer_qadr = int(m.jnt_qposadr[drawer_jnt])
        self._drawer_act = next(
            i for i in range(m.nu)
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == "drawer_actuator"
        )

    # ---- observation ------------------------------------------------------

    def object(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.scene.object_pose(name)

    def object_body(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise SkillFailed("observe", "observe", f"unknown object {name}")
        return bid

    def object_upright(self, name: str) -> float:
        _, quat = self.object(name)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
        return float(mat.reshape(3, 3)[2, 2])

    def object_speed(self, name: str) -> float:
        """Linear speed (m/s) of a free-floating object's body origin."""
        bid = self.object_body(name)
        adr = int(self.model.jnt_dofadr[int(self.model.body_jntadr[bid])])
        return float(np.linalg.norm(self.data.qvel[adr:adr + 3]))

    def drawer_opening(self) -> float:
        return float(self.data.qpos[self._drawer_qadr])

    def is_drawer_open(self) -> bool:
        return self.drawer_opening() >= 0.88 * DRAWER_TRAVEL

    def _contact_force(self, index: int) -> float:
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, self.data, index, wrench)
        return max(0.0, float(wrench[0]))

    def finger_forces(self, arm: str, object_name: str) -> tuple[float, float]:
        """Normal forces (N) between the object and the fixed/moving jaws."""
        bid = self.object_body(object_name)
        fixed = moving = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            if bid not in (b1, b2):
                continue
            other = int(c.geom2 if b1 == bid else c.geom1)
            if other not in self._jaw_geoms[arm]:
                continue
            f = self._contact_force(i)
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if ".fixed_jaw" in gname:
                fixed += f
            else:
                moving += f
        return fixed, moving

    def object_external_force(self, arm: str, object_name: str) -> float:
        """Normal force (N) on the object from anything other than the jaws."""
        bid = self.object_body(object_name)
        jaw_bodies = {int(self.model.geom_bodyid[g]) for g in self._jaw_geoms[arm]}
        total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            if bid not in (b1, b2):
                continue
            other = b2 if b1 == bid else b1
            if other in jaw_bodies:
                continue
            total += self._contact_force(i)
        return total

    def object_support_force(self, object_name: str) -> float:
        """Normal force (N) supporting the object from the table or drawer."""
        bid = self.object_body(object_name)
        total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            if bid not in (b1, b2):
                continue
            other = b2 if b1 == bid else b1
            oname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other) or "world"
            if oname in SUPPORT_BODIES:
                total += self._contact_force(i)
        return total

    def grasp_verified(self, arm: str, object_name: str, max_tilt_deg: float,
                       expect_site_at: np.ndarray | None = None,
                       check_upright: bool = True) -> bool:
        """Both jaws loaded and the ee at the grasp point; upright only when
        the object's orientation matters (capsules roll when squeezed).

        ``expect_site_at`` is the planned grasp position (hollow vessels put
        the body origin far from the gripper, so proximity is checked against
        the plan, not the object center).
        """
        fixed, moving = self.finger_forces(arm, object_name)
        if min(fixed, moving) <= JAW_FORCE_MIN:
            return False
        site_pos, _ = site_pose(self.data, f"{arm}.ee")
        reference = np.asarray(expect_site_at) if expect_site_at is not None else self.object(object_name)[0]
        limit = 0.02 if expect_site_at is not None else 0.06
        if float(np.linalg.norm(reference - site_pos)) > limit:
            return False
        if check_upright and self.object_upright(object_name) < float(
            np.cos(np.deg2rad(max_tilt_deg))
        ):
            return False
        return True

    # ---- action helpers ----------------------------------------------------

    def begin(self, skill_timeout: float = SKILL_TIMEOUT_S) -> None:
        """Start a skill: snapshot the hold pose and arm the deadline."""
        self._deadline = float(self.data.time) + skill_timeout
        for arm in ARM_NAMES:
            q = [float(self.data.qpos[self._jadr[arm][s]]) for s in ARM_JOINT_SUFFIXES]
            grip = float(self.data.qpos[self._jadr[arm]["gripper"]])
            self._hold[arm] = np.append(q, self.scene._ctrl_to_aperture(arm, grip))
            self._grip_now[arm] = float(self._hold[arm][5])

    def action(self, arm: str, q_arm: np.ndarray, aperture: float) -> np.ndarray:
        """Build a 12-dim merged target: acting arm moves, other arm holds."""
        out = np.zeros(12, dtype=np.float64)
        other = self._hold["B" if arm == "A" else "A"]
        if arm == "A":
            out[0:5], out[5], out[6:12] = q_arm, aperture, other
        else:
            out[0:6], out[6:11], out[11] = other, q_arm, aperture
        return out

    def step(self, action: np.ndarray) -> None:
        """Apply one 25 Hz action, advance physics, and run safety audits.

        Arm-joint targets come from the action; an active torque-limited
        gripper close and the neutral drawer servo are re-derived every
        physics substep.
        """
        self.scene.set_targets(action)
        for _ in range(int(round(TICK * PHYSICS_HZ))):
            if self.drawer_neutral:
                self.hold_drawer_neutral()
            for arm in ARM_NAMES:
                close = self._grip_close[arm]
                if close is None:
                    continue
                torque_limit, ctrl_des = close
                act = self._gact[arm]
                kp = float(self.model.actuator_gainprm[act, 0])
                kv = -float(self.model.actuator_biasprm[act, 2])
                qadr, vadr = self._jadr[arm]["gripper"], self._gadr[arm]["gripper"]
                qpos = float(self.data.qpos[qadr])
                qvel = float(self.data.qvel[vadr])
                torque = kp * (ctrl_des - qpos) - kv * qvel
                self.data.ctrl[act] = qpos + float(
                    np.clip(torque, -torque_limit, torque_limit)
                ) / kp
            mujoco.mj_step(self.model, self.data)
        for arm in ARM_NAMES:
            if self._grip_close[arm] is not None:
                qadr = self._jadr[arm]["gripper"]
                self._grip_now[arm] = float(
                    self.scene._ctrl_to_aperture(arm, float(self.data.qpos[qadr]))
                )
        if self.data.time > self._deadline:
            raise SkillFailed("skill", "timeout", "skill exceeded its time budget")
        self._audit_contacts()
        self.live.step()

    def _audit_contacts(self) -> None:
        for arm in ARM_NAMES:
            carried = self.carrying[arm]
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.dist > CONTACT_PENETRATION_TOL:
                    continue
                g1, g2 = int(c.geom1), int(c.geom2)
                geoms = self._arm_geoms[arm]
                if g1 not in geoms and g2 not in geoms:
                    continue
                other = int(g2 if g1 in geoms else g1)
                other_body = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[other])
                ) or "world"
                # Self-collisions INSIDE the arm (the official model's jaw vs
                # camera-mount graze at folded poses) are the robot's own
                # structural overlap, not a world collision.
                if other_body.startswith(f"{arm}."):
                    continue
                if carried is not None and other_body == carried:
                    continue
                if other_body in self.allowed[arm]:
                    continue
                raise SkillFailed("skill", "collision", f"arm {arm} contacted {other_body}")
            if carried is not None:
                fixed, moving = self.finger_forces(arm, carried)
                if min(fixed, moving) <= JAW_FORCE_MIN:
                    # Transient unloadings during motion are tolerated before
                    # failing: tall objects pivoting on a high grasp point
                    # (the bottle) re-seat after the lift transient.
                    now = float(self.data.time)
                    if self._bad_grip_since[arm] is None:
                        self._bad_grip_since[arm] = now
                    if now - self._bad_grip_since[arm] > 0.40:
                        raise SkillFailed("skill", "carry", f"lost verified grasp on {carried}")
                else:
                    self._bad_grip_since[arm] = None

    # ---- motion -------------------------------------------------------------

    def play(self, arm: str, points: np.ndarray, grip_from: float, grip_to: float,
             duration: float):
        """Quintic playback through joint waypoints; duration floored by gradient.

        Yields 12-dim merged targets; the caller steps physics per yield.
        """
        points = np.asarray(points, dtype=np.float64)
        gradient = float(np.abs(np.diff(points, axis=0)).max()) * (len(points) - 1)
        duration = max(duration, 1.875 * gradient / 0.8)
        started = float(self.data.time)
        self._grip_now[arm] = float(grip_to)
        while float(self.data.time) < started + duration:
            frac = _smooth((float(self.data.time) - started) / duration)
            pos = frac * (len(points) - 1)
            i = min(int(pos), len(points) - 2)
            q = points[i] + (points[i + 1] - points[i]) * (pos - i)
            grip = grip_from + (grip_to - grip_from) * frac
            yield self.action(arm, q, grip)
        yield self.action(arm, points[-1], grip_to)

    def play_joint(self, arm: str, q_goal: np.ndarray, duration: float):
        """Checked joint-space interpolation from the live pose to q_goal."""
        q_start = arm_q(self.data, arm)
        n = max(8, int(np.max(np.abs(np.asarray(q_goal) - q_start)) / 0.035) + 2)
        points = np.linspace(q_start, q_goal, n)
        grip = self._grip_now[arm]
        yield from self.play(arm, points, grip, grip, duration)

    def plan_ik(self, arm: str, target_pos, approach, lateral=None, seed=None) -> np.ndarray:
        """Solve one IK pose on the scratch data (live state is never touched)."""
        self.scratch.qpos[:] = self.data.qpos
        q0 = np.asarray(seed if seed is not None else arm_q(self.scratch, arm), dtype=np.float64)
        return solve_ik(self.model, self.scratch, f"{arm}.ee", target_pos, approach, q0,
                        target_lateral=lateral)

    def plan_cartesian(self, arm: str, start_pos: np.ndarray, end_pos: np.ndarray,
                       approach: np.ndarray, lateral: np.ndarray | None,
                       q_start: np.ndarray | None = None) -> np.ndarray:
        """IK-densified Cartesian waypoints (2 mm spacing, warm-chained scratch solves)."""
        dist = float(np.linalg.norm(end_pos - start_pos))
        count = max(3, int(dist / 0.002) + 2)
        self.scratch.qpos[:] = self.data.qpos
        q = np.asarray(q_start if q_start is not None else arm_q(self.scratch, arm),
                       dtype=np.float64)
        points = []
        for a in np.linspace(0.0, 1.0, count):
            target = start_pos + (end_pos - start_pos) * a
            q = solve_ik(self.model, self.scratch, f"{arm}.ee", target, approach, q,
                         target_lateral=lateral)
            points.append(q.copy())
        return np.asarray(points)

    def play_cartesian(self, arm: str, end_pos: np.ndarray, approach: np.ndarray,
                       lateral: np.ndarray | None, duration: float):
        """Plan (scratch IK + contact check) and play a Cartesian move."""
        start_pos, _ = site_pose(self.data, f"{arm}.ee")
        points = self.plan_cartesian(arm, start_pos, np.asarray(end_pos), approach, lateral)
        self.check_path(arm, points)
        grip = self._grip_now[arm]
        yield from self.play(arm, points, grip, grip, duration)

    def check_path(self, arm: str, points: np.ndarray, drawer_follow: bool = False) -> None:
        """Reject a waypoint path if any arm contact other than the carried
        object appears in a scratch MjData rollout (carried object teleported).

        ``drawer_follow`` advances the drawer joint proportionally along the
        path: a physical drawer pull drags the drawer along with the arm, so
        auditing every waypoint against the closed drawer would falsely
        reject the valid late-pull configurations.
        """
        scratch = self.scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0
        scratch.ctrl[:] = self.data.ctrl
        carried = self.carrying[arm]
        carry_ref = None
        if carried is not None:
            jid = int(self.model.body_jntadr[self.object_body(carried)])
            # Teleport only free-floating bodies (a 7-qpos pos+quat block). A
            # carried slide-joint body (the drawer during a physical pull) is
            # advanced by drawer_follow instead: writing the 7-qpos block at
            # its 1-DOF address would corrupt the joints that follow it in
            # the qpos vector (arm A's pose) and audit a garbage configuration.
            if int(self.model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                jadr = int(self.model.jnt_qposadr[jid])
                site_pos, site_rot = site_pose(self.data, f"{arm}.ee")
                obj_pos, obj_quat = self.object(carried)
                obj_mat = np.zeros(9, dtype=np.float64)
                mujoco.mju_quat2Mat(obj_mat, np.asarray(obj_quat, dtype=np.float64))
                carry_ref = (site_rot.T @ (obj_pos - site_pos), site_rot.T @ obj_mat.reshape(3, 3), jadr)
        for wi, q in enumerate(points):
            if drawer_follow:
                frac = wi / max(len(points) - 1, 1)
                scratch.qpos[self._drawer_qadr] = frac * DRAWER_TRAVEL
            for k, s in enumerate(ARM_JOINT_SUFFIXES):
                scratch.qpos[self._jadr[arm][s]] = q[k]
            scratch.qpos[self._jadr[arm]["gripper"]] = float(
                self.data.ctrl[self._gact[arm]]
            )
            mujoco.mj_forward(self.model, scratch)
            if carry_ref is not None:
                rel_pos, rel_rot, jadr = carry_ref
                s_pos, s_rot = site_pose(scratch, f"{arm}.ee")
                scratch.qpos[jadr : jadr + 3] = s_pos + s_rot @ rel_pos
                quat = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, (s_rot @ rel_rot).ravel())
                scratch.qpos[jadr + 3 : jadr + 7] = quat
                mujoco.mj_forward(self.model, scratch)
            for i in range(scratch.ncon):
                c = scratch.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                geoms = self._arm_geoms[arm]
                if g1 not in geoms and g2 not in geoms:
                    continue
                other = int(g2 if g1 in geoms else g1)
                other_body = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[other])
                ) or "world"
                if other_body.startswith(f"{arm}."):
                    continue  # arm-internal self-collision; see _audit_contacts
                if other_body == carried or other_body in self.allowed[arm]:
                    continue
                raise SkillFailed("skill", "path_blocked", f"planned path contacts {other_body}")

    def close_gripper(self, arm: str, torque_limit: float, aperture: float = 0.0,
                      seconds: float = 9.0, object_name: str | None = None,
                      q_hold: np.ndarray | None = None):
        """Torque-saturated gripper close (software clamp on the servo).

        The saturation runs per physics substep inside ``step``; this generator
        yields hold-still arm targets until both jaws are loaded against
        ``object_name`` (when given) or the duration elapses.
        """
        ctrl_des = self.scene._aperture_to_ctrl(arm, aperture)
        self._grip_close[arm] = (float(torque_limit), float(ctrl_des))
        held = np.asarray(q_hold if q_hold is not None else arm_q(self.data, arm),
                          dtype=np.float64).copy()
        end = float(self.data.time) + seconds
        try:
            while float(self.data.time) < end:
                yield self.action(arm, held, self._grip_now[arm])
                # No early exit on first contact: the saturated servo must keep
                # pressing for the whole window so the grasp is a real squeeze,
                # not a first-touch rest (the reference closes full-duration).
        finally:
            self._grip_close[arm] = None

    @contextmanager
    def grip_saturation(self, arm: str, torque_limit: float, aperture: float = 0.0):
        """Keep the torque-saturated close active across other motions.

        ``close_gripper`` clears the saturation when its generator exits, after
        which a plain position servo at the resting aperture exerts no
        steady-state squeeze — a dragged grasp (the drawer pull) then slips.
        Wrapping the motion keeps the servo pressing at ``torque_limit``.
        """
        self._grip_close[arm] = (
            float(torque_limit),
            float(self.scene._aperture_to_ctrl(arm, aperture)),
        )
        try:
            yield
        finally:
            self._grip_close[arm] = None

    def hold(self, arm: str, q_target: np.ndarray, seconds: float):
        """Hold a commanded joint target while the position servo converges.

        The quintic playback ends when the clock reaches the duration, but the
        arm is still several mm short of the final waypoint; measurements and
        phase checks that read the lagged pose fail spuriously (the deep
        drawer columns' lift).
        """
        q = np.asarray(q_target, dtype=np.float64)
        end = float(self.data.time) + seconds
        while float(self.data.time) < end:
            yield self.action(arm, q, self._grip_now[arm])

    def open_gripper(self, arm: str, aperture: float = 0.30, seconds: float = 0.8):
        q = arm_q(self.data, arm)
        grip0 = self._grip_now[arm]
        yield from self.play(arm, np.vstack([q, q]), grip0, aperture, seconds)

    def hold_drawer_neutral(self) -> None:
        """Zero the drawer servo error so a physical pull is not fought."""
        self.data.ctrl[self._drawer_act] = float(self.data.qpos[self._drawer_qadr])

    def hold_drawer_open(self) -> None:
        """Hand the drawer back to its servo holding the CURRENT opening.

        While neutral (zero servo error) the drawer is effectively free, and
        the withdrawing arm's incidental drag slides it shut — the follow-up
        CloseDrawer then reports ``already_closed`` (measured). The kp=200
        servo at a fixed target holds it against grazes.
        """
        self.drawer_neutral = False
        self.data.ctrl[self._drawer_act] = float(self.data.qpos[self._drawer_qadr])

    def servo_close_drawer(self, arm: str = "A", seconds: float = 3.0,
                           target_opening: float = 0.0):
        """Slide the drawer toward ``target_opening`` via its position servo
        while an arm holds a retrieved utensil (the physical CloseDrawer
        needs the gripper), or to bring a fully-open drawer's handle back
        into the arm's grasp band.

        The held arm keeps its current target; the saturation in effect
        around this call keeps the grasp alive.
        """
        start = float(self.data.qpos[self._drawer_qadr])
        t0 = float(self.data.time)
        while float(self.data.time) < t0 + seconds:
            frac = _smooth((float(self.data.time) - t0) / seconds)
            self.data.ctrl[self._drawer_act] = start + (target_opening - start) * frac
            yield self.action(arm, arm_q(self.data, arm), self._grip_now[arm])
