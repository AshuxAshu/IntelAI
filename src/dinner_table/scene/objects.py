"""Catalog of scene objects and reachability-bounded spawn pose sampler."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import CABINET_X, CABINET_Y, TABLE_TOP_HEIGHT

logger = logging.getLogger(__name__)

DRAWER_TRAY_Z = 0.387  # drawer floor top (TABLE_TOP_HEIGHT + 0.019) + capsule radius
UTENSIL_XY_JITTER_M = 0.01


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
        spawn_anchor=(-0.05, -0.20, TABLE_TOP_HEIGHT + 0.012),
        spawn_yaw_rad=0.0,
        mass_kg=0.065,
        friction=(0.8, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="rim",
    ),
    "mug": ObjectSpec(
        physics=("cylinder", (0.04, 0.04, 0.0)),
        spawn_anchor=(0.250, -0.065, TABLE_TOP_HEIGHT + 0.04),
        spawn_yaw_rad=0.0,
        mass_kg=0.045,
        friction=(0.9, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="handle",
    ),
    "bottle": ObjectSpec(
        physics=("cylinder", (0.03, 0.07, 0.0)),
        spawn_anchor=(0.09, -0.16, TABLE_TOP_HEIGHT + 0.07),
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
        physics=("capsule", (0.008, 0.055, 0.0)),
        spawn_anchor=(-0.20, CABINET_Y, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.014,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "spoon_2": ObjectSpec(
        physics=("capsule", (0.008, 0.055, 0.0)),
        spawn_anchor=(-0.14, CABINET_Y, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.014,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "fork_1": ObjectSpec(
        physics=("capsule", (0.007, 0.055, 0.0)),
        spawn_anchor=(-0.32, CABINET_Y, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.012,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "fork_2": ObjectSpec(
        physics=("capsule", (0.007, 0.055, 0.0)),
        spawn_anchor=(-0.26, CABINET_Y, DRAWER_TRAY_Z),
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


MIN_PAIRWISE_DISTANCES = {
    ("plate", "mug"): 0.135,
    ("plate", "bottle"): 0.130,
    ("mug", "bottle"): 0.085,
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


def instantiate(spec: mujoco.MjSpec, name: str, pose: tuple[np.ndarray, np.ndarray]) -> None:
    """Add a catalog object body and geometry to the MuJoCo specification."""
    if name not in OBJECT_CATALOG:
        raise SceneObjectError(f"unknown catalog object: {name}")
    obj_spec = OBJECT_CATALOG[name]
    pos, quat = pose
    body = spec.worldbody.add_body(name=name, pos=pos, quat=quat)
    if name != "drawer_top":
        body.add_freejoint()
    geom_type_str, geom_size = obj_spec.physics
    geom_type_map = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    }
    mjt_type = geom_type_map[geom_type_str]
    # Soft contacts (reference-tuned): catalog masses run 4-80 g, and the scene
    # default solref 0.005 is stiff enough to go numerically unstable there.
    solref = np.array([0.012, 1.0], dtype=np.float64)
    if geom_type_str == "capsule":
        # Utensils lie along the drawer's long axis (+Y), handles toward the arms.
        body.add_geom(
            type=mjt_type,
            size=geom_size,
            mass=obj_spec.mass_kg,
            friction=obj_spec.friction,
            quat=np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float64),
            solref=solref,
            condim=4,
        )
    else:
        body.add_geom(
            type=mjt_type,
            size=geom_size,
            mass=obj_spec.mass_kg,
            friction=obj_spec.friction,
            solref=solref,
            condim=4,
        )
    if obj_spec.visual_mesh is not None:
        mesh = spec.add_mesh(file=obj_spec.visual_mesh)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            contype=0,
            conaffinity=0,
            group=1,
        )
