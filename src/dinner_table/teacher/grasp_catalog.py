"""Per-object grasp frames for the privileged teacher.

The frames, grip torques, hovers, and carry limits are tuned per object against
the SO-101 gripper's measured jaw envelope:
top-down grasps constrain the approach axis plus a lateral gripper-X
direction, and the bottle is pinched at its neck rather than its body.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import ARM_MOUNTS
from dinner_table.scene.objects import BOTTLE_NECK_R, BOTTLE_NECK_Z, MUG_WALL_R, PLATE_RIM_R

DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)
UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


class GraspCatalogError(DinnerTableError):
    """Exception raised for unknown grasp targets or malformed frames."""


@dataclass(frozen=True)
class GraspFrame:
    """One grasp: where the ee site goes and how the gripper is oriented.

    position: world-frame ee-site target (m). approach: unit vector the site
    axis ``axis_index`` must follow (+Z, the finger direction, for top-down
    grasps; +Y, the jaw-spread axis, for the bottle's side grasp). lateral:
    optional unit vector the site's +X axis must follow (None = unconstrained).
    aperture: normalized open aperture during approach. wrist_roll_seed: roll
    value for the IK seed pose. grip_torque: gripper torque saturation (N m).
    hover_m / lift_m: pre-grasp hover and post-grasp lift offsets (m).
    max_tilt_deg: carried-object upright tolerance.
    """

    position: np.ndarray
    approach: np.ndarray
    lateral: np.ndarray | None
    aperture: float
    wrist_roll_seed: float
    grip_torque: float
    hover_m: float
    lift_m: float
    max_tilt_deg: float
    check_upright: bool = True
    axis_index: int = 2


@dataclass(frozen=True)
class CarryLimits:
    """Transit tolerances a carried object must stay inside.

    max_tilt_deg: how far the object may lean off its carried attitude.
    external_force_max: normal force (N) allowed from anything but the jaws.
    """

    max_tilt_deg: float
    external_force_max: float


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


class GraspCatalog:
    """Compute the proven grasp frame for a named scene object."""

    PLATE_LATERAL = np.array([np.cos(-1.0), np.sin(-1.0), 0.0], dtype=np.float64)
    MUG_LATERAL = np.array([0.55, -np.sqrt(1.0 - 0.55**2), 0.0], dtype=np.float64)
    # Measured SO-101 claw geometry: the jaws converge at +0.9 cm along the
    # site X axis (the fixed tip sits at +1.1 cm, the moving tip sweeps to
    # +0.7 cm). The pinch axis is therefore the SITE X: for a capsule lying
    # along world Y the lateral must be +/-X so the jaws close ACROSS the
    # stick. This orientation also swings the camera bracket toward the
    # drawer's open front (over the front wall), not into its back.
    UTENSIL_LATERAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    PINCH_OFFSET_M = 0.009  # pinch center ahead of the ee site along site X
    # The handle is a horizontal east-west bar; the gripper's X axis follows
    # the drawer's slide axis (site X = -Y world) so the closing faces squeeze
    # across the bar's diameter along the pull direction and the front run-in
    # seats the bar between the open jaws. Our site X negates the gripper X,
    # which sets the -X sign of this lateral;
    # (0, +1, 0) itself does not solve in our roll basin (verified).
    DRAWER_LATERAL = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    # Measured jaw envelope in the ee-site frame: the fixed jaw is static with
    # its tip spheres at site X +0.0119 (outer surface) and its face widening
    # to +0.0155 by site Z -0.027; the moving jaw sweeps from -0.036 (aperture
    # 0.30) to +0.0076 (closed). A top-down grasp lets the moving jaw shove the
    # object against the fixed face, but the bottle's side grasp descends
    # ACROSS the jaws, so the neck must already clear the fixed face on the way
    # down or the descent stalls against it (measured: 11 N at the fixed jaw).
    FIXED_JAW_X = 0.0119
    NECK_CLEARANCE_M = 0.0015  # gap left between the neck wall and the fixed face
    NECK_DEPTH_M = 0.008  # how far past the tip line the neck is seated

    def frame(self, scene, name: str, arm: str) -> GraspFrame:
        """Return the grasp frame for `name` grasped by `arm`, at its live pose."""
        if name == "drawer_top":
            return self._drawer_frame(scene)
        pos, quat = scene.object_pose(name)
        return self.frame_at(name, pos, quat, arm)

    def frame_at(self, name: str, pos, quat, arm: str) -> GraspFrame:
        """Grasp frame for `name` at an ARBITRARY pose, not necessarily its live one.

        Planning a placement that the other arm must then pick up (the relay)
        needs the grasp frame of a pose the object does not hold yet.

        ``lateral`` is the solver-bound site-X direction. NOTE: the site's X
        axis is the gripper X negated, and in our MuJoCo build the robust
        closing arc comes from the opposite roll basin
        (the naive basin's own picks fail on this MuJoCo version — verified), so
        the frames below pin the basin that clamps correctly here.
        """
        if name not in ("plate", "mug", "bottle", "spoon_1", "spoon_2", "fork_1", "fork_2"):
            raise GraspCatalogError(f"no grasp frame for object: {name}")
        pos = np.asarray(pos, dtype=np.float64)
        upright = float(_quat_to_mat(quat)[2, 2])
        if name == "plate":
            position = pos + self.PLATE_LATERAL * PLATE_RIM_R + np.array([0.0, 0.0, 0.009])
            return GraspFrame(position, DOWN, self.PLATE_LATERAL, 0.30, -2.4, 0.7, 0.055, 0.055, 15.0)
        if name == "mug":
            # Wall pinch at the diagonal: the moving jaw
            # presses the wall's inner face, the mug slides until the wall
            # seats against the fixed jaw, and the full-duration saturated
            # close squeezes. The handle is NOT graspable with this gripper:
            # the moving-jaw mesh sweeps the mug wall on the way in.
            position = pos + self.MUG_LATERAL * (MUG_WALL_R - 0.002) + np.array([0.0, 0.0, 0.033])
            return GraspFrame(position, DOWN, self.MUG_LATERAL, 0.30, -2.4, 0.7, 0.055, 0.035, 15.0)
        if name == "bottle":
            if upright < 0.5:
                raise GraspCatalogError("bottle is lying sideways; sideways regrasp unsupported")
            # Neck side grasp: the fingers lie horizontal and
            # close ACROSS the narrow neck, so only the jaw-spread axis (site
            # Y) is pinned — the reach direction is left to the solver, which
            # is what makes the pose solvable at all from a front-edge mount.
            # A top-down wall pinch at the bottle's radius slips: the neck
            # wall is 3 mm thick and the tall body levers straight out of it.
            neck = pos + np.array([0.0, 0.0, BOTTLE_NECK_Z])
            return GraspFrame(neck + self._side_offset(arm, neck), UP, None, 0.30, -1.52,
                              0.5, 0.055, 0.060, 15.0, axis_index=1)
        # Cutlery rolls when squeezed, so the upright check must not apply
        # (a rolled utensil is still grasped). Descent aperture: wide enough
        # that the arm's servo tracking error (~5-10 mm) cannot land a jaw
        # face on the handle, narrow enough to clear the neighboring columns
        # (~3.5 cm apart). The site offset puts the jaw tip band (straddling
        # the site by +/-2.5 mm) across the handle's mid-height; pinching
        # lower puts the COM above the contacts (the utensil pitches out
        # mid-carry) and pinching at the upper band slips the tips off the
        # handle's top edge (both measured).
        position = pos + np.array([0.0, 0.0, 0.007])
        # Close and carry at a firm-but-not-maximal clamp (the servo is
        # commanded CLOSED throughout): 1.5 N m holds the ~0.2 N utensil with
        # a 40x friction margin, while the full 2.94 N m clamp chatters the
        # tip contacts at ~37 N and ratchets the handle's roll until its
        # diagonal wedges and snaps the utensil out of the jaws (measured).
        return GraspFrame(position, DOWN, self.UTENSIL_LATERAL, 0.16, -2.4, 1.5,
                          0.025, 0.035, 30.0, check_upright=False)

    CARRY_TILT_DEG = 15.0
    CUTLERY_CARRY_TILT_DEG = 30.0
    CARRY_EXTERNAL_FORCE_MAX = 0.10

    def carry_limits(self, name: str) -> CarryLimits:
        """Transit tolerances a carried object must stay inside."""
        if name.startswith(("fork", "spoon")):
            return CarryLimits(self.CUTLERY_CARRY_TILT_DEG, self.CARRY_EXTERNAL_FORCE_MAX)
        return CarryLimits(self.CARRY_TILT_DEG, self.CARRY_EXTERNAL_FORCE_MAX)

    def side_reach(self, arm: str, target) -> tuple[np.ndarray, np.ndarray]:
        """Unit reach direction and jaw-spread axis of a side grasp at `target`.

        With the jaw-spread axis pinned horizontal and the reach direction left
        to the solver, the site frame is fixed up to the reach azimuth, which
        the arm's shoulder-pan plane sets: it points from the mount to the
        target. The site X axis is that direction turned a quarter turn about
        world up.
        """
        mount = np.asarray(ARM_MOUNTS[arm], dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        reach = np.array([target[0] - mount[0], target[1] - mount[1], 0.0])
        norm = float(np.linalg.norm(reach))
        if norm < 1e-6:
            raise GraspCatalogError("side grasp target coincides with the arm mount")
        reach = reach / norm
        return reach, np.array([-reach[1], reach[0], 0.0], dtype=np.float64)

    def side_lateral(self, arm: str, target) -> np.ndarray:
        """Jaw-spread axis of a side grasp at `target` (the tilt axis for a pour)."""
        return self.side_reach(arm, target)[1]

    def _side_offset(self, arm: str, target: np.ndarray) -> np.ndarray:
        """Tool-point offset that seats a bottle neck between the open jaws.

        The corrections are built from the reach azimuth alone rather than from
        a pinned lateral, which would over-constrain a 5-DOF solve.
        """
        reach, lateral = self.side_reach(arm, target)
        neck_x = self.FIXED_JAW_X - BOTTLE_NECK_R - self.NECK_CLEARANCE_M
        return -neck_x * lateral + self.NECK_DEPTH_M * reach

    def _drawer_frame(self, scene) -> GraspFrame:
        sid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle")
        if sid == -1:
            raise GraspCatalogError("drawer_handle site missing from scene")
        # The tool point goes directly on the horizontal bar's center; the
        # fingers straddle it vertically and the servo close squeezes across
        # the slide axis. grip_torque is recorded for reference
        # only — the drawer skills close with the plain (force-clamped) servo.
        bar = np.array(scene.data.site_xpos[sid], dtype=np.float64)
        return GraspFrame(bar, DOWN, self.DRAWER_LATERAL, 0.30, -2.4, 0.15, 0.040, 0.040, 15.0)
