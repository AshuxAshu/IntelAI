"""Per-object grasp frames for the privileged teacher.

The frames, grip torques, hovers, and carry limits are ported from the
reference solution's proven grasp strategies (see docs/PLAN_AMENDMENTS.md):
top-down grasps constrain the approach axis plus a lateral gripper-X
direction; the bottle is a side grasp with the gripper's fingers horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import ARM_MOUNTS
from dinner_table.scene.objects import BOTTLE_WALL_R, MUG_WALL_R, PLATE_RIM_R

DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


class GraspCatalogError(DinnerTableError):
    """Exception raised for unknown grasp targets or malformed frames."""


@dataclass(frozen=True)
class GraspFrame:
    """One grasp: where the ee site goes and how the gripper is oriented.

    position: world-frame ee-site target (m). approach: unit vector the site's
    +Z (finger direction) must follow. lateral: optional unit vector the site's
    +X axis must follow (the reference's x_target; None = unconstrained).
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
    # The vertical handle bar is pinched across like a utensil capsule; the
    # (1,0,0) lateral is the only orientation that solves at the drawer's
    # 25 cm forward depth.
    DRAWER_LATERAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def frame(self, scene, name: str, arm: str) -> GraspFrame:
        """Return the grasp frame for `name` grasped by `arm`.

        ``lateral`` is the solver-bound site-X direction. NOTE: the site's X
        axis is the gripper X negated, and in our MuJoCo build the robust
        closing arc comes from the opposite roll basin to the reference's
        (their engine's own picks fail on this MuJoCo version — verified), so
        the frames below pin the basin that clamps correctly here.
        """
        if name == "drawer_top":
            return self._drawer_frame(scene)
        if name not in ("plate", "mug", "bottle", "spoon_1", "spoon_2", "fork_1", "fork_2"):
            raise GraspCatalogError(f"no grasp frame for object: {name}")
        pos, quat = scene.object_pose(name)
        upright = float(_quat_to_mat(quat)[2, 2])
        if name == "plate":
            position = pos + self.PLATE_LATERAL * PLATE_RIM_R + np.array([0.0, 0.0, 0.017])
            return GraspFrame(position, DOWN, self.PLATE_LATERAL, 0.30, -2.4, 0.7, 0.055, 0.055, 15.0)
        if name == "mug":
            # Wall pinch at the diagonal (reference port): the moving jaw
            # presses the wall's inner face, the mug slides until the wall
            # seats against the fixed jaw, and the full-duration saturated
            # close squeezes. The handle is NOT graspable with this gripper:
            # the moving-jaw mesh sweeps the mug wall on the way in.
            position = pos + self.MUG_LATERAL * (MUG_WALL_R - 0.002) + np.array([0.0, 0.0, 0.041])
            return GraspFrame(position, DOWN, self.MUG_LATERAL, 0.30, -2.4, 0.7, 0.055, 0.035, 15.0)
        if name == "bottle":
            if upright < 0.5:
                raise GraspCatalogError("bottle is lying sideways; sideways regrasp unsupported")
            # Wall pinch at the diagonal on the body wall (the mug's
            # mechanism at the bottle's radius). The bottle pick is not yet
            # reliable — the neck side-grasp needs a solver mode that pins
            # only the gripper-Y axis; tracked in PLAN_AMENDMENTS.
            position = (pos + self.MUG_LATERAL * (BOTTLE_WALL_R - 0.002)
                        + np.array([0.0, 0.0, 0.030]))
            return GraspFrame(position, DOWN, self.MUG_LATERAL, 0.30, -2.4, 0.7, 0.055, 0.035, 15.0)
        # Capsules are rotationally symmetric: squeezing them rolls them, so
        # the upright check must not apply (a rolled utensil is still grasped).
        # Descent aperture: wide enough that the arm's servo tracking error
        # (~5-10 mm) cannot land a jaw face on the capsule, narrow enough to
        # clear the neighboring utensil columns (5 cm apart).
        position = pos + np.array([0.0, 0.0, 0.009])
        return GraspFrame(position, DOWN, self.UTENSIL_LATERAL, 0.16, -2.4, 0.5,
                          0.025, 0.035, 30.0, check_upright=False)

    def _drawer_frame(self, scene) -> GraspFrame:
        sid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle")
        if sid == -1:
            raise GraspCatalogError("drawer_handle site missing from scene")
        bar = np.array(scene.data.site_xpos[sid], dtype=np.float64)
        # The site sits 9 mm west of the bar plus the vertical pinch offset:
        # the claw's pinch center is +9 mm along the site X (lateral +X), so
        # this lands the pinch exactly on the vertical bar.
        position = bar + np.array([-self.PINCH_OFFSET_M, 0.0, 0.0055])
        return GraspFrame(position, DOWN, self.DRAWER_LATERAL, 0.30, -2.4, 0.15, 0.025, 0.040, 15.0)
