"""Water liquid simulation using flex particles or visual proxy cylinder under Decision Gate G1."""

from __future__ import annotations

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError

# Decision Gate G1: flipped to True after particle containment evaluation across 10 seeds
FALLBACK_VISUAL: bool = True

PARTICLE_RADIUS = 0.006
PARTICLE_MASS = 0.001

# Maximum cylinder half-heights (m) for visual proxy encoding
MAX_WATER_HALF_HEIGHT = {
    "bottle": 0.020,
    "mug": 0.035,
}


class WaterError(DinnerTableError):
    """Exception raised for water initialization or measurement errors."""


def attach_water(spec: mujoco.MjSpec, bottle_name: str = "bottle") -> None:
    """Attach flex particles or visual proxy cylinder inside the bottle body."""
    bottle_body = None
    mug_body = None
    for b in spec.worldbody.bodies:
        if b.name == bottle_name:
            bottle_body = b
        else:
            if b.name == "mug":
                mug_body = b

    if bottle_body is None:
        raise WaterError(f"bottle body not found in spec: {bottle_name}")

    if FALLBACK_VISUAL:
        # Visual proxy: translucent blue cylinder in the bottle's lower body
        # (10 cm bottle: wall 0.007-0.048, water half-height 0.02).
        bottle_body.add_geom(
            name="water_bottle_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([0.024, MAX_WATER_HALF_HEIGHT["bottle"], 0.0], dtype=np.float64),
            pos=np.array([0.0, 0.0, 0.027], dtype=np.float64),
            rgba=np.array([0.2, 0.5, 0.85, 0.6], dtype=np.float64),
            contype=0,
            conaffinity=0,
            group=1,
        )
        if mug_body is not None:
            # Visual proxy: initially empty translucent cylinder in mug (half-height 0.0001 m)
            mug_body.add_geom(
                name="water_mug_geom",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=np.array([0.021, 0.0001, 0.0], dtype=np.float64),
                pos=np.array([0.0, 0.0, 0.032], dtype=np.float64),
                rgba=np.array([0.2, 0.5, 0.85, 0.6], dtype=np.float64),
                contype=0,
                conaffinity=0,
                group=1,
            )
    else:
        # Particle system: ~220 particles distributed in lower 70% of bottle volume
        particle_index = 0
        z_levels = np.linspace(-0.08, 0.04, 7)
        rings = [(0.012, 6), (0.020, 10), (0.026, 14)]

        for z in z_levels:
            p_body = spec.worldbody.add_body(
                name=f"water_part_{particle_index}",
                pos=np.array(
                    [bottle_body.pos[0], bottle_body.pos[1], bottle_body.pos[2] + z],
                    dtype=np.float64,
                ),
            )
            p_body.add_freejoint()
            p_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([PARTICLE_RADIUS, 0.0, 0.0], dtype=np.float64),
                mass=PARTICLE_MASS,
                rgba=np.array([0.2, 0.5, 0.85, 0.7], dtype=np.float64),
                friction=np.array([0.01, 0.001, 0.0001], dtype=np.float64),
                group=0,
            )
            particle_index += 1

            for r, count in rings:
                for i in range(count):
                    theta = 2.0 * np.pi * float(i) / float(count)
                    px = bottle_body.pos[0] + r * np.cos(theta)
                    py = bottle_body.pos[1] + r * np.sin(theta)
                    pz = bottle_body.pos[2] + z

                    p_body = spec.worldbody.add_body(
                        name=f"water_part_{particle_index}",
                        pos=np.array([px, py, pz], dtype=np.float64),
                    )
                    p_body.add_freejoint()
                    p_body.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([PARTICLE_RADIUS, 0.0, 0.0], dtype=np.float64),
                        mass=PARTICLE_MASS,
                        rgba=np.array([0.2, 0.5, 0.85, 0.7], dtype=np.float64),
                        friction=np.array([0.01, 0.001, 0.0001], dtype=np.float64),
                        group=0,
                    )
                    particle_index += 1


def set_fill_fraction(model: mujoco.MjModel, container_name: str, fraction: float) -> None:
    """Set the visual liquid height scale for the specified container from 0.0 to 1.0."""
    if container_name not in MAX_WATER_HALF_HEIGHT:
        raise WaterError(f"unknown container for visual water scale: {container_name}")
    geom_name = f"water_{container_name}_geom"
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid == -1:
        raise WaterError(f"water geom not found in model: {geom_name}")
    clamped = float(np.clip(fraction, 0.0, 1.0))
    target_half_h = max(0.0001, MAX_WATER_HALF_HEIGHT[container_name] * clamped)
    model.geom_size[gid][1] = target_half_h


def fill_fraction(model: mujoco.MjModel, data: mujoco.MjData, container_name: str) -> float:
    """Return fraction of water inside the specified container from 0.0 to 1.0."""
    if FALLBACK_VISUAL:
        if container_name not in MAX_WATER_HALF_HEIGHT:
            return 0.0
        geom_name = f"water_{container_name}_geom"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid == -1:
            return 0.0
        current_half_h = float(model.geom_size[gid][1])
        max_half_h = MAX_WATER_HALF_HEIGHT[container_name]
        return float(np.clip(current_half_h / max_half_h, 0.0, 1.0))

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, container_name)
    if cid == -1:
        return 0.0

    c_pos = data.xpos[cid]
    if container_name == "bottle":
        half_extents = np.array([0.045, 0.045, 0.12], dtype=np.float64)
    else:
        if container_name == "mug":
            half_extents = np.array([0.055, 0.055, 0.055], dtype=np.float64)
        else:
            half_extents = np.array([0.050, 0.050, 0.050], dtype=np.float64)

    total_particles = 0
    inside_particles = 0

    for b_idx in range(model.nbody):
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_idx)
        if b_name is not None and b_name.startswith("water_part_"):
            total_particles += 1
            p_pos = data.xpos[b_idx]
            dx = abs(p_pos[0] - c_pos[0])
            dy = abs(p_pos[1] - c_pos[1])
            dz = abs(p_pos[2] - c_pos[2])
            if dx <= half_extents[0] and dy <= half_extents[1] and dz <= half_extents[2]:
                inside_particles += 1

    if total_particles == 0:
        return 0.0
    else:
        return float(inside_particles) / float(total_particles)
