"""Catalog of scene objects and reachability-bounded spawn pose sampler."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import TABLE_TOP_HEIGHT

logger = logging.getLogger(__name__)

DRAWER_TRAY_Z = 0.31
UTENSIL_XY_JITTER_M = 0.03


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
        spawn_anchor=(0.10, 0.05, TABLE_TOP_HEIGHT + 0.012),
        spawn_yaw_rad=0.0,
        mass_kg=0.25,
        friction=(0.8, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="rim",
    ),
    "mug": ObjectSpec(
        physics=("cylinder", (0.04, 0.04, 0.0)),
        spawn_anchor=(-0.10, 0.05, TABLE_TOP_HEIGHT + 0.04),
        spawn_yaw_rad=0.0,
        mass_kg=0.30,
        friction=(0.9, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="handle",
    ),
    "bottle": ObjectSpec(
        physics=("cylinder", (0.035, 0.10, 0.0)),
        spawn_anchor=(0.0, -0.12, TABLE_TOP_HEIGHT + 0.10),
        spawn_yaw_rad=0.0,
        mass_kg=0.60,
        friction=(0.9, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="neck",
    ),
    "spoon_1": ObjectSpec(
        physics=("capsule", (0.008, 0.08, 0.0)),
        spawn_anchor=(0.08, 0.55, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.04,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "spoon_2": ObjectSpec(
        physics=("capsule", (0.008, 0.08, 0.0)),
        spawn_anchor=(0.12, 0.55, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.04,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "fork_1": ObjectSpec(
        physics=("capsule", (0.007, 0.085, 0.0)),
        spawn_anchor=(-0.08, 0.55, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.05,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "fork_2": ObjectSpec(
        physics=("capsule", (0.007, 0.085, 0.0)),
        spawn_anchor=(-0.12, 0.55, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.05,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "drawer_top": ObjectSpec(
        physics=("box", (0.24, 0.07, 0.05)),
        spawn_anchor=(0.0, 0.60, TABLE_TOP_HEIGHT + 0.19),
        spawn_yaw_rad=0.0,
        mass_kg=1.2,
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
                if name.startswith(("spoon", "fork")):
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
        if not has_overlap:
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
    if geom_type_str == "capsule":
        body.add_geom(
            type=mjt_type,
            size=geom_size,
            mass=obj_spec.mass_kg,
            friction=obj_spec.friction,
            quat=np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float64),
        )
    else:
        body.add_geom(
            type=mjt_type,
            size=geom_size,
            mass=obj_spec.mass_kg,
            friction=obj_spec.friction,
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
