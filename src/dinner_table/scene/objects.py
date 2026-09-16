"""Catalog of scene objects and reachability-bounded spawn pose sampler."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import CABINET_X, CABINET_Y, TABLE_TOP_HEIGHT

logger = logging.getLogger(__name__)

DRAWER_TRAY_Z = 0.389  # cutlery rail tops (TABLE_TOP_HEIGHT + 0.029); utensil origins sit at their base
UTENSIL_XY_JITTER_M = 0.01
UTENSIL_Y_JITTER_M = 0.003  # rails are 6 mm wide in y (+/-3 mm of jitter)
# Cutlery lies in the tray at near-zero yaw (+/-4 deg): a
# strongly yawed box can only be pinched at a corner — a knife-edge grip
# that sags and slips under carry (measured).
UTENSIL_YAW_JITTER_RAD = 0.07
# The bottle's neck grasp sits high in the arm's constrained envelope: the
# verified-feasible region around its anchor is razor-thin, so its spawn
# jitter is minimal (mass/friction DR still fully randomize).
BOTTLE_XY_JITTER_M = 0.015
# Inner faces of the drawer's side walls (scenes/parts/drawer_cabinet.xml:
# walls at local x = +/-0.150, 3 mm thick, on the caddy at CABINET_X). A
# utensil's jittered-and-yawed footprint is clamped to stay inside them with
# a margin: the east column's spread otherwise crosses the wall by a
# fraction of a millimetre and the contact ejects it out of the drawer.
DRAWER_INNER_X = (CABINET_X - 0.147, CABINET_X + 0.147)
DRAWER_WALL_MARGIN_M = 0.003


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
    # arms) in four columns on the cutlery rails, all within arm A's reach once
    # the drawer has slid open. The east-most column stops at x = -0.15: the
    # gripper's body reaches about 31 mm east of its grasp point, so a column
    # any further east drives the jaw assembly into the drawer's east wall
    # and the close stalls on it (measured: 17 mm of penetration, 8/20 picks).
    "spoon_1": ObjectSpec(
        physics=("box", (0.009, 0.055, 0.006)),
        spawn_anchor=(-0.15, CABINET_Y - 0.015, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.014,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "spoon_2": ObjectSpec(
        physics=("box", (0.009, 0.055, 0.006)),
        spawn_anchor=(-0.255, CABINET_Y - 0.015, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.014,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "fork_1": ObjectSpec(
        physics=("box", (0.007, 0.055, 0.006)),
        spawn_anchor=(-0.22, CABINET_Y - 0.015, DRAWER_TRAY_Z),
        spawn_yaw_rad=0.0,
        mass_kg=0.012,
        friction=(0.5, 0.005, 0.0001),
        visual_mesh=None,
        grasp_class="mid_handle",
    ),
    "fork_2": ObjectSpec(
        physics=("box", (0.007, 0.055, 0.006)),
        spawn_anchor=(-0.185, CABINET_Y - 0.015, DRAWER_TRAY_Z),
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


def _clamp_into_drawer(x: float, yaw: float, spec: ObjectSpec) -> float:
    """Clamp a utensil's x (m) so its yawed footprint clears the drawer walls."""
    half_x, half_y = spec.physics[1][0], spec.physics[1][1]
    reach = abs(np.cos(yaw)) * half_x + abs(np.sin(yaw)) * half_y
    low = DRAWER_INNER_X[0] + reach + DRAWER_WALL_MARGIN_M
    high = DRAWER_INNER_X[1] - reach - DRAWER_WALL_MARGIN_M
    return float(np.clip(x, low, high))


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
                    # The cutlery rails are 6 mm wide in y: a large y jitter
                    # lands the handle between them and the utensil rests on
                    # the drawer floor instead (the cutlery rails allow only
                    # +/-3 mm of y jitter; x and yaw still randomize fully).
                    jitter_y_limit = UTENSIL_Y_JITTER_M
                    yaw_limit = UTENSIL_YAW_JITTER_RAD
                elif name == "bottle":
                    # The neck grasp sits high in the arm's constrained
                    # envelope; the verified-feasible region around the anchor
                    # is tight, so the bottle's spawn jitter is halved (mass
                    # and friction DR still fully randomize).
                    jitter_limit = min(BOTTLE_XY_JITTER_M, dr.spawn_xy_jitter_m)
                    jitter_y_limit = jitter_limit
                    yaw_limit = dr.spawn_yaw_jitter_rad
                else:
                    jitter_limit = dr.spawn_xy_jitter_m
                    jitter_y_limit = dr.spawn_xy_jitter_m
                    yaw_limit = dr.spawn_yaw_jitter_rad
                jitter_x = rng.uniform(-jitter_limit, jitter_limit)
                jitter_y = rng.uniform(-jitter_y_limit, jitter_y_limit)
                yaw_jitter = rng.uniform(-yaw_limit, yaw_limit)
                pos = np.array(
                    [
                        spec.spawn_anchor[0] + jitter_x,
                        spec.spawn_anchor[1] + jitter_y,
                        spec.spawn_anchor[2],
                    ],
                    dtype=np.float64,
                )
                yaw = spec.spawn_yaw_rad + yaw_jitter
                if name.startswith(("spoon", "fork")):
                    pos[0] = _clamp_into_drawer(pos[0], yaw, spec)

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


# Composite dimensions: vessels are HOLLOW — base plate plus
# tangential-box ring walls —
# because the grasp strategies pinch rim walls and mug handles that
# solid primitives cannot offer. Body origins sit at the object base; the
# grasp catalog's offsets are measured against these radii.
PLATE_RIM_R = 0.061
MUG_WALL_R = 0.025
BOTTLE_WALL_R = 0.028  # body wall centerline radius
BOTTLE_NECK_Z = 0.073  # neck mid-height above the base (10 cm bottle)
BOTTLE_NECK_WALL = 0.003
BOTTLE_NECK_CENTER_R = 0.011  # neck ring centerline radius
BOTTLE_NECK_R = BOTTLE_NECK_CENTER_R + BOTTLE_NECK_WALL / 2.0  # outer radius
BOTTLE_NECK_H = 0.030
BOTTLE_MOUTH_Z = BOTTLE_NECK_Z + BOTTLE_NECK_H / 2.0  # pour lip above the base
MUG_WALL_THICKNESS = 0.004
MUG_LIP_THICKNESS = 0.0045  # the rim band is the mug's tightest constriction
MUG_INNER_R = MUG_WALL_R - MUG_LIP_THICKNESS / 2.0
MUG_RIM_Z = 0.078  # wall ring top above the base; the pour lip must clear it
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
    if name.startswith("fork"):
        return 1 + 1 + 1 + 4
    if name.startswith("spoon"):
        return 1 + 1 + 1
    return 1


def _box(body, name: str, half_x: float, half_y: float, half_z: float,
         pos, mass: float, friction) -> None:
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.array([half_x, half_y, half_z], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mass=mass,
        friction=friction,
        solref=SOFT_SOLREF,
        condim=4,
    )


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
        # Plate scale (radius 0.066 + 10 mm rim wall): the grasp
        # offset 0.061 is measured against THIS rim, and the smaller disc
        # clears the arm's shoulder structure during rim grasps.
        _cyl(body, name, 0.058, 0.003, 0.003, per, friction)
        _ring(body, name, PLATE_RIM_R, 0.010, 0.020, 0.014, per, friction)
    elif name == "mug":
        # Mug dimensions (the grasp offsets are measured against
        # this exact wall radius and height). Massless
        # geoms + one centered inertial: with
        # per-geom masses the handle capsules offset the COM and the mug
        # pivots toward the handle, creeping ~15 cm/min on the soft contacts
        # (measured) — which trips the placement verify's bystander check.
        _cyl(body, name, MUG_WALL_R, 0.007, 0.007, 0.0, friction)
        _ring(body, name, MUG_WALL_R, MUG_WALL_THICKNESS, 0.064, 0.046, 0.0, friction)
        _ring(body, name, MUG_WALL_R, MUG_LIP_THICKNESS, 0.003, 0.0645, 0.0, friction)
        _capsule(body, name, 0.004, 0.012, [0.035, 0.0, 0.019], 0.0, friction)
        _capsule(body, name, 0.004, 0.012, [0.035, 0.0, 0.055], 0.0, friction)
        _capsule(body, name, 0.004, 0.018, [0.047, 0.0, 0.037], 0.0, friction,
                 quat=np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float64))
        body.explicitinertial = True
        body.mass = mass_kg
        body.ipos = [0.0, 0.0, 0.032]
        # Box inertia over the mug size (.077, .050, .064).
        body.inertia = (mass_kg / 12.0 * np.array(
            [0.050**2 + 0.064**2, 0.077**2 + 0.064**2, 0.077**2 + 0.050**2]
        )).tolist()
    elif name == "bottle":
        # 10 cm hollow bottle: base, body wall, shoulder step, narrow neck.
        # The neck is the grasp feature: its outer diameter must fit inside the
        # gripper's fixed-jaw offset (measured 11.9 mm from the tool point), so
        # it keeps a slim neck rather than a scaled-up
        # one, which the fixed jaw could not descend past.
        _cyl(body, name, BOTTLE_WALL_R, 0.007, 0.007, per, friction)
        _ring(body, name, BOTTLE_WALL_R, 0.004, 0.041, 0.0275, per, friction)
        _ring(body, name, 0.024, 0.008, 0.010, 0.053, per, friction)
        _ring(body, name, BOTTLE_NECK_CENTER_R, BOTTLE_NECK_WALL, BOTTLE_NECK_H,
              BOTTLE_NECK_Z, per, friction)


def _build_utensil(body, name: str, mass_kg: float, friction) -> None:
    """Cutlery as a handle-neck-head composite at real cutlery dimensions.

    A uniform stick gives the jaw tips only +/-6 mm of side face before the
    box pitches off the point contacts mid-carry — regardless of grip force
    (measured at 0.5 and 2.94 N m). The composite's handle is taller (16 mm)
    and massless geoms plus one centered inertial
    keep the COM at the object's middle: a per-geom mass distribution puts
    4/7 of a fork's mass in its tines, and the front-heavy pendulum droops
    its head onto the drawer floor when pinched at the handle (measured:
    29 deg droop). Origins sit at the base, like the vessels.
    """
    is_fork = name.startswith("fork")
    width = 0.017 if is_fork else 0.022
    length, height = 0.110, 0.016
    _box(body, name, 0.0045, 0.034, 0.008, [0.0, -0.019, 0.008], 0.0, friction)
    _box(body, name, 0.003, 0.010, 0.004, [0.0, 0.020, 0.004], 0.0, friction)
    if is_fork:
        _box(body, name, 0.0085, 0.007, 0.004, [0.0, 0.032, 0.004], 0.0, friction)
        for i in range(4):
            _box(body, name, 0.00125, 0.011, 0.0015,
                 [(i - 1.5) * 0.0048, 0.046, 0.004], 0.0, friction)
    else:
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=np.array([0.011, 0.019, 0.003], dtype=np.float64),
            pos=np.array([0.0, 0.038, 0.004], dtype=np.float64),
            mass=0.0,
            friction=friction,
            solref=SOFT_SOLREF,
            condim=4,
        )
    body.explicitinertial = True
    body.mass = mass_kg
    body.ipos = [0.0, 0.0, height / 2.0]
    body.inertia = (mass_kg / 12.0 * np.array(
        [length**2 + height**2, width**2 + height**2, width**2 + length**2]
    )).tolist()


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
    if name.startswith(("fork", "spoon")):
        _build_utensil(body, name, obj_spec.mass_kg, obj_spec.friction)
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
