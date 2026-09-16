"""Domain randomization engine for mutating MjSpec before compilation."""

from __future__ import annotations

from pathlib import Path
import logging
import mujoco
import numpy as np
import yaml

from dinner_table.config import DinnerTableError
from dinner_table.scene.objects import OBJECT_CATALOG, DrProfile

logger = logging.getLogger(__name__)

TABLE_TEXTURES = [
    "table/table_01.png",
    "table/table_02.png",
    "table/table_03.png",
    "table/table_04.png",
    "table/table_05.png",
]
FLOOR_TEXTURES = [
    "floor/floor_01.png",
    "floor/floor_02.png",
    "floor/floor_03.png",
    "floor/floor_04.png",
]
WALL_TEXTURES = ["wall/wall_01.png", "wall/wall_02.png", "wall/wall_03.png"]
PLACEMAT_TEXTURES = [
    "placemat/placemat_01.png",
    "placemat/placemat_02.png",
    "placemat/placemat_03.png",
]


class RandomizerError(DinnerTableError):
    """Exception raised for domain randomization failures."""


def load_dr_profile(profile_name_or_path: str | Path) -> DrProfile:
    """Load a domain randomization profile from configs/scene/ or a file path."""
    path = Path(profile_name_or_path)
    if not path.is_file():
        candidate = Path("configs/scene") / f"{profile_name_or_path}.yaml"
        if candidate.is_file():
            path = candidate
        else:
            candidate_with_suffix = Path("configs/scene") / profile_name_or_path
            if candidate_with_suffix.is_file():
                path = candidate_with_suffix
            else:
                raise RandomizerError(f"unable to locate dr profile: {profile_name_or_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    object_mass_scale: dict[str, tuple[float, float]] = {}
    for key, value in data.items():
        if key.endswith("_mass"):
            name = key[: -len("_mass")]
            if name not in OBJECT_CATALOG:
                raise RandomizerError(f"unknown object in per-object mass key: {key}")
            object_mass_scale[name] = (float(value[0]), float(value[1]))

    return DrProfile(
        mass_scale=tuple(data.get("mass_scale", (0.5, 2.0))),
        friction_scale=tuple(data.get("friction_scale", (0.4, 1.2))),
        spawn_xy_jitter_m=float(data.get("spawn_xy_jitter_m", 0.06)),
        spawn_yaw_jitter_rad=float(data.get("spawn_yaw_jitter_rad", 0.44)),
        gripper_noise_sigma_rad=float(data.get("gripper_noise_sigma_rad", 0.035)),
        waypoint_jitter_m=float(data.get("waypoint_jitter_m", 0.02)),
        light_intensity=tuple(data.get("light_intensity", (0.5, 1.5))),
        light_color_k=tuple(data.get("light_color_k", (3000, 7000))),
        light_pos_sigma_m=float(data.get("light_pos_sigma_m", 0.08)),
        texture_swap_probability=float(data.get("texture_swap_probability", 0.9)),
        camera_jitter_deg=float(data.get("camera_jitter_deg", 2.0)),
        drawer_friction_scale=tuple(data.get("drawer_friction_scale", (0.5, 2.0))),
        perturb_event_probability=float(data.get("perturb_event_probability", 0.35)),
        holdout_textures=bool(data.get("holdout_textures", False)),
        seed=data.get("seed", None),
        object_mass_scale=object_mass_scale,
    )


def apply_dr(
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
    profile: str | DrProfile = "dr_train",
) -> None:
    """Mutate MjSpec physics, lights, friction, and textures according to the DR profile."""
    if isinstance(profile, str):
        dr = load_dr_profile(profile)
    else:
        dr = profile

    mass_mult = rng.uniform(dr.mass_scale[0], dr.mass_scale[1])
    friction_mult = rng.uniform(dr.friction_scale[0], dr.friction_scale[1])

    for body in spec.worldbody.bodies:
        # Scale free movable object geoms
        has_freejoint = False
        for jnt in body.joints:
            if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
                has_freejoint = True
        if has_freejoint:
            if body.name in dr.object_mass_scale:
                lo, hi = dr.object_mass_scale[body.name]
                mult = rng.uniform(lo, hi)
            else:
                mult = mass_mult
            for geom in body.geoms:
                if not np.isnan(geom.mass):
                    geom.mass = float(geom.mass * mult)
                geom.friction = geom.friction * friction_mult

    # Scale drawer joint friction and damping
    drawer_scale = rng.uniform(dr.drawer_friction_scale[0], dr.drawer_friction_scale[1])
    for body in spec.worldbody.bodies:
        if body.name == "cabinet":
            for child in body.bodies:
                if child.name == "drawer_top":
                    for jnt in child.joints:
                        if jnt.name == "drawer_slide":
                            jnt.damping = float(jnt.damping * drawer_scale)
                            jnt.frictionloss = float(jnt.frictionloss * drawer_scale)

    # Randomize lighting
    light_mult = rng.uniform(dr.light_intensity[0], dr.light_intensity[1])
    for light in spec.worldbody.lights:
        light.diffuse = light.diffuse * light_mult
        light.specular = light.specular * light_mult
        if dr.light_pos_sigma_m > 0.0:
            light.pos = light.pos + rng.normal(0.0, dr.light_pos_sigma_m, size=3)

    if rng.random() < dr.texture_swap_probability:
        if dr.holdout_textures:
            table_pool = TABLE_TEXTURES
            floor_pool = FLOOR_TEXTURES
            wall_pool = WALL_TEXTURES
            placemat_pool = PLACEMAT_TEXTURES
        else:
            table_pool = TABLE_TEXTURES[:-1]
            floor_pool = FLOOR_TEXTURES[:-1]
            wall_pool = WALL_TEXTURES[:-1]
            placemat_pool = PLACEMAT_TEXTURES[:-1]

        for tex in spec.textures:
            if tex.name == "table_tex":
                tex.file = str(rng.choice(table_pool))
            elif tex.name == "floor_tex":
                tex.file = str(rng.choice(floor_pool))
            elif tex.name == "wall_tex":
                tex.file = str(rng.choice(wall_pool))
            elif tex.name == "placemat_tex":
                tex.file = str(rng.choice(placemat_pool))
