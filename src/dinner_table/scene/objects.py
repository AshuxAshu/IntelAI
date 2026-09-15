"""Catalog of scene objects and reachability-bounded spawn pose sampler."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import CABINET_X, CABINET_Y, TABLE_TOP_HEIGHT

logger = logging.getLogger(__name__)

DRAWER_TRAY_Z = 0.385  # drawer floor top (TABLE_TOP_HEIGHT + 0.019) + box half-height
UTENSIL_XY_JITTER_M = 0.01
# The bottle's neck grasp sits high in the arm's constrained envelope: the
# verified-feasible region around its anchor is razor-thin, so its spawn
# jitter is minimal (mass/friction DR still fully randomize).
BOTTLE_XY_JITTER_M = 0.015


class SceneObjectError(DinnerTableError):
    """Exception raised for scene object catalog and instantiation failures."""


@dataclass(frozen=True)
class DrProfile:
    """Domain randomization profile parameters governing object spawn jitter and physics scaling."""

    mass_scale: tuple[float, float] = (0.5, 2.0)
    friction_scale: tuple[float, float] = (0.4, 1.2)
    spawn_xy_jitter_m: float = 0.06
    spawn_yaw_jitter_rad: float = 0.44
    gripper_noise_sigma_rad: float = 0.035
    waypoint_jitter_m: float = 0.02
    light_intensity: tuple[float, float] = (0.5, 1.5)
    light_color_k: tuple[int, int] = (3000, 7000)
    light_pos_sigma_m: float = 0.08
    texture_swap_probability: float = 0.9
    camera_jitter_deg: float = 2.0
    drawer_friction_scale: tuple[float, float] = (0.5, 2.0)
    perturb_event_probability: float = 0.35
    holdout_textures: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class ObjectSpec:
    """Specification of a physical object including geometry, anchor, mass, and grasp class."""

    physics: tuple[str, tuple[float, ...]]
    spawn_anchor: tuple[float, float, float]
    spawn_yaw_rad: float
    mass_kg: float
    friction: tuple[float, float, float]
    visual_mesh: str | None
    grasp_class: str


OBJECT_CATALOG: dict[str, ObjectSpec] = {
    "plate": ObjectSpec(
        physics=("cylinder", (0.09, 0.012, 0.0)),
        spawn_anchor=(-0.05, -0.20, TABLE_TOP_HEIGHT + 0.001),
        spawn_yaw_rad=0.0,
        mass_kg=0.065,
        friction=(0.8, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="rim",
    ),
    "mug": ObjectSpec(
        physics=("cylinder", (0.04, 0.04, 0.0)),
        spawn_anchor=(0.27, -0.10, TABLE_TOP_HEIGHT + 0.001),
        spawn_yaw_rad=0.0,
        mass_kg=0.045,
        friction=(0.9, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="handle",
    ),
    "bottle": ObjectSpec(
        physics=("cylinder", (0.03, 0.07, 0.0)),
        spawn_anchor=(0.10, -0.09, TABLE_TOP_HEIGHT + 0.001),
        spawn_yaw_rad=0.0,
        mass_kg=0.080,
        friction=(0.9, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="neck",
    ),
    # Utensils are real-cutlery-length capsules lying along Y (handle toward the
    # arms) in four columns on the drawer floor, all within arm A's reach once
    # the drawer has slid open.
    "spoon_1": ObjectSpec(
        physics=("box", (0.009, 0.055, 0.006)),
        spawn_anchor=(-0.20, CABINET_Y, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.014,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "spoon_2": ObjectSpec(
        physics=("box", (0.009, 0.055, 0.006)),
        spawn_anchor=(-0.165, CABINET_Y, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.014,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "fork_1": ObjectSpec(
        physics=("box", (0.007, 0.055, 0.006)),
        spawn_anchor=(-0.28, CABINET_Y, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.012,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "fork_2": ObjectSpec(
        physics=("box", (0.007, 0.055, 0.006)),
        spawn_anchor=(-0.24, CABINET_Y, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.012,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "drawer_top": ObjectSpec(
        physics=("box", (0.108, 0.072, 0.02)),
        spawn_anchor=(CABINET_X, CABINET_Y, TABLE_TOP_HEIGHT),
        spawn_yaw_rad=0.0,
        mass_kg=0.18,
        friction=(1.0, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="handle",
    ),
}


# Minimums include each object's approach-lane clearance: a bottle spawned
# 10 cm from the mug puts the mug's hover pose inside the bottle's neck.
MIN_PAIRWISE_DISTANCES = {
    ("plate", "mug"): 0.20,
    ("plate", "bottle"): 0.16,
    ("mug", "bottle"): 0.16,
}

# Horizontal furniture footprints (x_min, x_max, y_min, y_max) that free objects
# may never spawn inside: the cutlery caddy body (walls, floor, roof), expanded by
# the spawn jitter plus a small guard. Deep interpenetration with the caddy is the
# one spawn defect that ejects objects violently; resting against the low arm pads
# is a benign contact and is not rejected.
FURNITURE_FOOTPRINTS = ((CABINET_X - 0.21, CABINET_X + 0.21, CABINET_Y - 0.13, CABINET_Y + 0.10),)


def _inside_furniture(xy: np.ndarray) -> bool:
    """True if a horizontal position falls inside any furniture footprint."""
    for x_min, x_max, y_min, y_max in FURNITURE_FOOTPRINTS:
        if x_min <= xy[0] <= x_max and y_min <= xy[1] <= y_max:
            return True
    return False


def sample_spawns(
    rng: np.random.Generator, dr: DrProfile
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Sample bounded spawn positions (m) and yaw quaternions for all catalog objects."""
    for _ in range(100):
        spawns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, spec in OBJECT_CATALOG.items():
            if name == "drawer_top":
                pos = np.array(spec.spawn_anchor, dtype=np.float64)
                yaw = spec.spawn_yaw_rad
            else:
                if name.startswith("spoon") or name.startswith("fork"):
                    jitter_limit = UTENSIL_XY_JITTER_M
                elif name == "bottle":
                    # The neck grasp sits high in the arm's constrained
                    # envelope; the verified-feasible region around the anchor
                    # is tight, so the bottle's spawn jitter is halved (mass
                    # and friction DR still fully randomize).
                    jitter_limit = min(BOTTLE_XY_JITTER_M, dr.spawn_xy_jitter_m)
                else:
                    jitter_limit = dr.spawn_xy_jitter_m
                jitter_x = rng.uniform(-jitter_limit, jitter_limit)
                jitter_y = rng.uniform(-jitter_limit, jitter_limit)
                yaw_jitter = rng.uniform(-dr.spawn_yaw_jitter_rad, dr.spawn_yaw_jitter_rad)
                pos = np.array(
                    [
                        spec.spawn_anchor[0] + jitter_x,
                        spec.spawn_anchor[1] + jitter_y,
                        spec.spawn_anchor[2],
                    ],
                    dtype=np.float64,
                )
                yaw = spec.spawn_yaw_rad + yaw_jitter

            half_yaw = yaw * 0.5
            quat = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)
            spawns[name] = (pos, quat)

        has_overlap = False
        for (o1, o2), min_dist in MIN_PAIRWISE_DISTANCES.items():
            if o1 in spawns and o2 in spawns:
                dist_2d = float(np.linalg.norm(spawns[o1][0][:2] - spawns[o2][0][:2]))
                if dist_2d < min_dist:
                    has_overlap = True
                    break
        # Table objects must clear the caddy; the utensils live inside it by
        # design, so only plate/mug/bottle (and any future tabletop object)
        # are checked against the furniture footprints.
        table_objects = (
            name
            for name in spawns
            if name != "drawer_top" and not name.startswith(("spoon", "fork"))
        )
        if not has_overlap and not any(
            _inside_furniture(spawns[name][0][:2]) for name in table_objects
        ):
            return spawns

    return spawns


# Composite dimensions (Amendment 1): vessels are HOLLOW — base plate plus
# tangential-box ring walls (the reference solution's proven construction) —
# because the reference grasp strategies pinch rim walls and mug handles that
# solid primitives cannot offer. Body origins sit at the object base; the
# grasp catalog's offsets are measured against these radii.
PLATE_RIM_R = 0.061
MUG_WALL_R = 0.025
BOTTLE_WALL_R = 0.028  # body wall centerline radius
BOTTLE_NECK_Z = 0.073  # neck mid-height above the base (10 cm bottle)
SOFT_SOLREF = np.array([0.012, 1.0], dtype=np.float64)


def _z_quat(angle: float) -> np.ndarray:
    half = angle * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _count_geoms(name: str) -> int:
    if name == "plate":
        return 1 + 24
    if name == "mug":
        return 1 + 24 + 24 + 3
    if name == "bottle":
        return 1 + 24 + 24 + 24
    return 1


def _ring(body, name: str, radius: float, thickness: float, height: float,
          z: float, mass_each: float, friction, segments: int = 24) -> None:
    """Tangential overlapping boxes forming a closed ring wall (hollow vessel)."""
    half_y = (radius + thickness / 2.0) * np.tan(np.pi / segments)
    for i in range(segments):
        a = 2.0 * np.pi * i / segments
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.array([thickness / 2.0, half_y, height / 2.0], dtype=np.float64),
            pos=np.array([radius * np.cos(a), radius * np.sin(a), z], dtype=np.float64),
            quat=_z_quat(a),
            mass=mass_each,
            friction=friction,
            solref=SOFT_SOLREF,
            condim=4,
        )


def _cyl(body, name: str, radius: float, half_h: float, z: float,
         mass: float, friction) -> None:
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([radius, half_h, 0.0], dtype=np.float64),
        pos=np.array([0.0, 0.0, z], dtype=np.float64),
        mass=mass,
        friction=friction,
        solref=SOFT_SOLREF,
        condim=4,
    )


def _capsule(body, name: str, radius: float, half_len: float, pos, mass: float,
             friction, quat=None) -> None:
    kwargs = {}
    if quat is not None:
        kwargs["quat"] = quat
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([radius, half_len, 0.0], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mass=mass,
        friction=friction,
        solref=SOFT_SOLREF,
        condim=4,
        **kwargs,
    )


def _build_vessel(body, name: str, mass_kg: float, friction) -> None:
    per = mass_kg / _count_geoms(name)
    if name == "plate":
        # Reference plate scale (radius 0.066 + 10 mm rim wall): the grasp
        # offset 0.061 is measured against THIS rim, and the smaller disc
        # clears the arm's shoulder structure during rim grasps.
        _cyl(body, name, 0.058, 0.003, 0.003, per, friction)
        _ring(body, name, PLATE_RIM_R, 0.010, 0.020, 0.014, per, friction)
    elif name == "mug":
        # Reference mug dimensions (their grasp offsets are measured against
        # this exact wall radius and height; Amendment 1 port).
        _cyl(body, name, 0.025, 0.007, 0.007, per, friction)
        _ring(body, name, 0.025, 0.004, 0.064, 0.046, per, friction)
        _ring(body, name, 0.025, 0.0045, 0.003, 0.0645, per, friction)
        _capsule(body, name, 0.004, 0.012, [0.035, 0.0, 0.019], per, friction)
        _capsule(body, name, 0.004, 0.012, [0.035, 0.0, 0.055], per, friction)
        _capsule(body, name, 0.004, 0.018, [0.047, 0.0, 0.037], per, friction,
                 quat=np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float64))
    elif name == "bottle":
        # 10 cm hollow bottle: base, body wall, shoulder step, narrow neck.
        # The four working grasp recipes (plate/mug/utensils) are unaffected
        # by the bottle; this structure only needs to settle stably.
        _cyl(body, name, BOTTLE_WALL_R, 0.007, 0.007, per, friction)
        _ring(body, name, BOTTLE_WALL_R, 0.004, 0.041, 0.0275, per, friction)
        _ring(body, name, 0.024, 0.008, 0.010, 0.053, per, friction)
        _ring(body, name, 0.013, 0.003, 0.030, 0.073, per, friction)


def instantiate(spec: mujoco.MjSpec, name: str, pose: tuple[np.ndarray, np.ndarray]) -> None:
    """Add a catalog object body and geometry to the MuJoCo specification."""
    if name not in OBJECT_CATALOG:
        raise SceneObjectError(f"unknown catalog object: {name}")
    obj_spec = OBJECT_CATALOG[name]
    pos, quat = pose
    body = spec.worldbody.add_body(name=name, pos=pos, quat=quat)
    if name != "drawer_top":
        body.add_freejoint()
    if name in ("plate", "mug", "bottle"):
        _build_vessel(body, name, obj_spec.mass_kg, obj_spec.friction)
        return
    geom_type_str, geom_size = obj_spec.physics
    geom_type_map = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    }
    mjt_type = geom_type_map[geom_type_str]
    body.add_geom(
        type=mjt_type,
        size=geom_size,
        mass=obj_spec.mass_kg,
        friction=obj_spec.friction,
        solref=SOFT_SOLREF,
        condim=4,
    )
