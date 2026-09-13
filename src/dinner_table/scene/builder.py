"""Physical simulation scene builder assembling MuJoCo dual-arm environment and objects."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import (
    ACTION_DIM,
    ARM_MOUNTS,
    ARM_ORIENTATIONS,
    CAMERA_NAMES,
    DRAWER_TRAVEL,
    HOME_JOINTS,
    JOINT_NAMES,
    OVERHEAD_RESOLUTION,
    PHYSICS_HZ,
    POLICY_IMAGE_SIZE,
    SO101_JOINT_SUFFIXES,
)
from dinner_table.scene.objects import OBJECT_CATALOG, instantiate, sample_spawns
from dinner_table.scene.randomizer import apply_dr, load_dr_profile

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path("assets/meshes/so101/so101_calibration.json")
SCENE_XML_PATH = Path("scenes/dinner_table.xml")

KP_GAINS = {
    "shoulder_pan": 40.0,
    "shoulder_lift": 40.0,
    "elbow_flex": 30.0,
    "wrist_flex": 20.0,
    "wrist_roll": 15.0,
    "gripper": 10.0,
}

FRC_LIMITS = {
    "shoulder_pan": 6.0,
    "shoulder_lift": 6.0,
    "elbow_flex": 4.0,
    "wrist_flex": 3.0,
    "wrist_roll": 2.0,
    "gripper": 2.0,
}


class SceneBuilderError(DinnerTableError):
    """Exception raised for scene construction or physics execution errors."""


class Scene:
    """Complete physical simulation environment composed of table, dual arms, and objects."""

    def __init__(self, seed: int = 42, dr_profile: str = "dr_train") -> None:
        """Initialize the scene with specified seed and domain randomization profile."""
        self.seed = seed
        self.dr_profile_name = dr_profile
        self.ready = False

        if not CALIBRATION_FILE.is_file():
            raise SceneBuilderError(f"calibration file missing: {CALIBRATION_FILE}")
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            self._calibration = json.load(f)

        dr = load_dr_profile(dr_profile)
        self.dr = dr
        rng = np.random.default_rng(seed)

        if not SCENE_XML_PATH.is_file():
            raise SceneBuilderError(f"scene xml file missing: {SCENE_XML_PATH}")

        self.spec = mujoco.MjSpec.from_file(str(SCENE_XML_PATH))

        # Attach dual SO-101 robot arms
        for arm in ("A", "B"):
            self._attach_arm(self.spec, prefix=arm, pos=ARM_MOUNTS[arm], yaw=ARM_ORIENTATIONS[arm])

        # Sample and instantiate objects
        self._spawns = sample_spawns(rng, dr)
        for name, pose in self._spawns.items():
            if name != "drawer_top":
                instantiate(self.spec, name, pose)

        # Attach cameras
        self._attach_cameras(self.spec)

        # Attach drawer position actuator after arm actuators to preserve JOINT_NAMES order
        drawer_act = self.spec.add_actuator()
        drawer_act.name = "drawer_actuator"
        drawer_act.target = "drawer_slide"
        drawer_act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        drawer_act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        drawer_act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        gp = np.zeros(10, dtype=np.float64)
        gp[0] = 200.0
        drawer_act.gainprm = gp
        bp = np.zeros(10, dtype=np.float64)
        bp[1] = -200.0
        bp[2] = -10.0
        drawer_act.biasprm = bp
        drawer_act.ctrllimited = True
        drawer_act.ctrlrange = np.array([0.0, DRAWER_TRAVEL], dtype=np.float64)
        drawer_act.forcelimited = True
        drawer_act.forcerange = np.array([-25.0, 25.0], dtype=np.float64)

        # Apply domain randomization before compiling
        apply_dr(self.spec, rng, dr)

        # Compile model and allocate data
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        self._renderers: dict[str, mujoco.Renderer] = {}
        self._depth_renderer: mujoco.Renderer | None = None

        self.reset()
        self.ready = True

    def _attach_arm(self, spec: mujoco.MjSpec, prefix: str, pos: tuple[float, float, float], yaw: float) -> None:
        """Attach one 6-joint SO-101 kinematic chain and tuned position actuators."""
        half_yaw = yaw * 0.5
        quat = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)

        base = spec.worldbody.add_body(name=f"{prefix}_base", pos=pos, quat=quat)
        base.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.array([0.05, 0.05, 0.01], dtype=np.float64),
            pos=np.array([0.0, 0.0, 0.01], dtype=np.float64),
            mass=0.2,
        )

        parent_body = base
        link_configs = [
            ("shoulder_pan", [0.0, 0.0, 0.058], [0.0, 0.0, 1.0], [0.025, 0.025, 0.0], [0.0, 0.0, 0.02], 0.15),
            ("shoulder_lift", [0.0, 0.0, 0.052], [0.0, 1.0, 0.0], [0.022, 0.050, 0.0], [0.0, 0.0, 0.05], 0.18),
            ("elbow_flex", [0.0, 0.0, 0.115], [0.0, 1.0, 0.0], [0.020, 0.045, 0.0], [0.0, 0.0, 0.045], 0.14),
            ("wrist_flex", [0.0, 0.0, 0.095], [0.0, 1.0, 0.0], [0.018, 0.030, 0.0], [0.0, 0.0, 0.03], 0.10),
            ("wrist_roll", [0.0, 0.0, 0.060], [0.0, 0.0, 1.0], [0.018, 0.025, 0.0], [0.0, 0.0, 0.02], 0.08),
            ("gripper", [0.0, 0.0, 0.045], [0.0, 1.0, 0.0], [0.012, 0.020, 0.0], [0.0, 0.015, 0.025], 0.04),
        ]

        for suffix, rel_pos, axis, geom_size, geom_pos, mass in link_configs:
            body = parent_body.add_body(name=f"{prefix}_{suffix}_link", pos=rel_pos)
            range_deg = self._calibration[suffix]["range_deg"]
            range_rad = np.deg2rad(range_deg)

            kp = KP_GAINS[suffix]
            kv = kp * 0.05
            frc = FRC_LIMITS[suffix]

            body.add_joint(
                name=f"{prefix}.{suffix}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=axis,
                range=range_rad,
                damping=kv,
            )
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=geom_size,
                pos=geom_pos,
                mass=mass,
            )

            act = spec.add_actuator()
            act.name = f"{prefix}.{suffix}"
            act.target = f"{prefix}.{suffix}"
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE

            gp = np.zeros(10, dtype=np.float64)
            gp[0] = kp
            act.gainprm = gp

            bp = np.zeros(10, dtype=np.float64)
            bp[1] = -kp
            bp[2] = -kv
            act.biasprm = bp

            act.ctrllimited = True
            act.ctrlrange = np.array(range_rad, dtype=np.float64)
            act.forcelimited = True
            act.forcerange = np.array([-frc, frc], dtype=np.float64)

            if suffix == "gripper":
                body.add_site(name=f"{prefix}.ee", pos=[0.0, 0.0, 0.055], size=[0.005, 0.0, 0.0])
                body.add_camera(name=f"wrist_{prefix}", pos=[0.0, 0.04, 0.06], quat=[0.92388, 0.38268, 0.0, 0.0], fovy=60.0)

            parent_body = body

    def _attach_cameras(self, spec: mujoco.MjSpec) -> None:
        """Attach overhead and demo cameras to the worldbody."""
        spec.worldbody.add_camera(
            name="overhead",
            pos=[0.0, 0.0, 1.35],
            quat=[1.0, 0.0, 0.0, 0.0],
            fovy=50.0,
        )
        spec.worldbody.add_camera(
            name="demo_cam",
            pos=[0.7, -0.8, 1.0],
            quat=[0.85, 0.35, 0.15, 0.35],
            fovy=55.0,
        )

    def _aperture_to_ctrl(self, arm: str, aperture: float) -> float:
        """Map normalized gripper aperture 0-1 to joint control angle in radians."""
        range_deg = self._calibration["gripper"]["range_deg"]
        min_rad = float(np.deg2rad(range_deg[0]))
        max_rad = float(np.deg2rad(range_deg[1]))
        clamped = float(np.clip(aperture, 0.0, 1.0))
        return min_rad + clamped * (max_rad - min_rad)

    def _ctrl_to_aperture(self, arm: str, qpos_val: float) -> float:
        """Map raw joint angle in radians to normalized gripper aperture 0-1."""
        range_deg = self._calibration["gripper"]["range_deg"]
        min_rad = float(np.deg2rad(range_deg[0]))
        max_rad = float(np.deg2rad(range_deg[1]))
        if max_rad - min_rad <= 1e-9:
            return 0.0
        else:
            return float(np.clip((qpos_val - min_rad) / (max_rad - min_rad), 0.0, 1.0))

    def reset(self) -> None:
        """Reset simulation state deterministically and idempotently for this seed."""
        mujoco.mj_resetData(self.model, self.data)

        # Re-apply catalog object spawn poses
        for name, (pos, quat) in self._spawns.items():
            if name != "drawer_top":
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id != -1:
                    jnt_id = self.model.body_jntadr[body_id]
                    if jnt_id != -1:
                        if self.model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                            qpos_adr = self.model.jnt_qposadr[jnt_id]
                            self.data.qpos[qpos_adr : qpos_adr + 3] = pos
                            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
                            dof_adr = self.model.jnt_dofadr[jnt_id]
                            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

        # Reset drawer slide
        drawer_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        if drawer_jnt != -1:
            self.data.qpos[self.model.jnt_qposadr[drawer_jnt]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[drawer_jnt]] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.settle(0.5)

    def qpos_12(self) -> np.ndarray:
        """Read current joint positions in JOINT_NAMES order with gripper normalized to 0-1."""
        vals = np.zeros(ACTION_DIM, dtype=np.float64)
        for i, jname in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qpos_idx = self.model.jnt_qposadr[jid]
            raw_val = float(self.data.qpos[qpos_idx])
            if jname.endswith(".gripper"):
                arm = jname[0]
                vals[i] = self._ctrl_to_aperture(arm, raw_val)
            else:
                vals[i] = raw_val
        return vals

    def set_targets(self, action: np.ndarray) -> None:
        """Set position actuator targets for both arms in JOINT_NAMES order."""
        if len(action) != ACTION_DIM:
            raise SceneBuilderError(f"action shape must be ({ACTION_DIM},), got {len(action)}")
        targets = np.array(action, dtype=np.float64, copy=True)
        # Convert normalized gripper apertures (indices 5 and 11) to radians
        targets[5] = self._aperture_to_ctrl("A", targets[5])
        targets[11] = self._aperture_to_ctrl("B", targets[11])
        self.data.ctrl[:ACTION_DIM] = targets

    def step(self, duration: float) -> None:
        """Advance physics at PHYSICS_HZ for duration seconds."""
        num_steps = max(1, round(duration * PHYSICS_HZ))
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

    def hold_safe(self) -> None:
        """Command both arms to HOME_JOINTS smoothly over 1.0 s in 25 Hz increments."""
        q_start = self.qpos_12()
        q_home = np.concatenate([HOME_JOINTS["A"], HOME_JOINTS["B"]], dtype=np.float64)
        steps = 25
        dt = 1.0 / float(steps)
        for t in range(1, steps + 1):
            alpha = float(t) / float(steps)
            q_interp = (1.0 - alpha) * q_start + alpha * q_home
            self.set_targets(q_interp)
            self.step(dt)

    def render(self, camera: str) -> np.ndarray:
        """Render uint8 RGB image from specified camera name."""
        if camera not in CAMERA_NAMES:
            raise SceneBuilderError(f"unknown camera name: {camera}")

        if camera not in self._renderers:
            if camera in ("wrist_A", "wrist_B"):
                h, w = POLICY_IMAGE_SIZE
            else:
                h, w = OVERHEAD_RESOLUTION
            self._renderers[camera] = mujoco.Renderer(self.model, h, w)

        renderer = self._renderers[camera]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def render_depth(self) -> np.ndarray:
        """Render float32 depth map in meters from the overhead camera."""
        if self._depth_renderer is None:
            h, w = OVERHEAD_RESOLUTION
            self._depth_renderer = mujoco.Renderer(self.model, h, w)
            self._depth_renderer.enable_depth_rendering()

        self._depth_renderer.update_scene(self.data, camera="overhead")
        return self._depth_renderer.render()

    def camera_intrinsics(self, camera: str) -> np.ndarray:
        """Compute (3, 3) intrinsic matrix K from camera vertical field-of-view."""
        if camera not in CAMERA_NAMES:
            raise SceneBuilderError(f"unknown camera name: {camera}")

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        fovy = float(self.model.cam_fovy[cam_id])

        if camera in ("wrist_A", "wrist_B"):
            h, w = POLICY_IMAGE_SIZE
        else:
            h, w = OVERHEAD_RESOLUTION

        f = float(h) / (2.0 * np.tan(np.deg2rad(fovy) * 0.5))
        return np.array(
            [
                [f, 0.0, float(w) * 0.5],
                [0.0, f, float(h) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return privileged ground truth (pos, quat) for the named object."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise SceneBuilderError(f"unknown object body: {name}")
        pos = np.array(self.data.xpos[body_id], dtype=np.float64)
        quat = np.array(self.data.xquat[body_id], dtype=np.float64)
        return pos, quat

    def spawn_meta(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return initial spawn coordinates and orientations."""
        return self._spawns

    def apply_impulse(self, object_name: str, direction: np.ndarray, magnitude: float) -> None:
        """Apply external force impulse via mj_applyFT to the named object's center of mass."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id == -1:
            raise SceneBuilderError(f"unknown object for impulse: {object_name}")
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            f = (np.array(direction, dtype=np.float64) / norm) * magnitude
        else:
            f = np.zeros(3, dtype=np.float64)
        point = np.array(self.data.xipos[body_id], dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)
        mujoco.mj_applyFT(self.model, self.data, f, torque, point, body_id, self.data.qfrc_applied)

    def fill_fraction(self, container: str) -> float:
        """Return container water fill fraction from 0.0 to 1.0."""
        return 0.0

    def settle(self, seconds: float) -> None:
        """Advance physics with zero control targets for specified seconds."""
        num_steps = max(1, round(seconds * PHYSICS_HZ))
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

    def is_drawer_open(self) -> bool:
        """Return True if drawer joint travel exceeds half of maximum travel."""
        drawer_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        if drawer_jnt == -1:
            return False
        qpos_idx = self.model.jnt_qposadr[drawer_jnt]
        if float(self.data.qpos[qpos_idx]) > (DRAWER_TRAVEL * 0.5):
            return True
        else:
            return False
