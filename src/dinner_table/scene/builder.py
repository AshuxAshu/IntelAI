"""Physical simulation scene builder assembling MuJoCo dual-arm environment and objects."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import (
    ACTION_DIM,
    CAMERA_NAMES,
    CONTROL_HZ,
    DRAWER_TRAVEL,
    HOME_JOINTS,
    JOINT_NAMES,
    PHYSICS_HZ,
)
from dinner_table.scene.arms import attach_arms, expand_includes
from dinner_table.scene.cameras import CameraRig
from dinner_table.scene.objects import instantiate, sample_spawns
from dinner_table.scene.randomizer import apply_dr, load_dr_profile
from dinner_table.scene.water import attach_water
from dinner_table.scene.water import fill_fraction as calc_fill_fraction

CALIBRATION_FILE = Path("assets/meshes/so101/so101_calibration.json")
SCENE_XML_PATH = Path("scenes/dinner_table.xml")


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

        # Expand scene part includes, merge both official SO-101 arms, then hand
        # the flat MJCF to MjSpec for the programmatic passes below. Absolute
        # mesh/texture dirs keep asset resolution independent of the xml origin.
        scene_root = ET.parse(SCENE_XML_PATH).getroot()
        expand_includes(scene_root, SCENE_XML_PATH.parent)
        attach_arms(scene_root)
        compiler = scene_root.find("compiler")
        compiler.set("meshdir", str((SCENE_XML_PATH.parent / "../assets/meshes").resolve()))
        compiler.set("texturedir", str((SCENE_XML_PATH.parent / "../assets/textures").resolve()))
        self.spec = mujoco.MjSpec.from_string(ET.tostring(scene_root, encoding="unicode"))

        # Sample and instantiate objects
        self._spawns = sample_spawns(rng, dr)
        for name, pose in self._spawns.items():
            if name != "drawer_top":
                instantiate(self.spec, name, pose)

        # Attach water in bottle
        attach_water(self.spec, "bottle")

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

        self.camera_rig = CameraRig(self.model)

        self.reset()
        self.ready = True

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

        # Park both arms at HOME (qpos and targets) before any physics: the arm's
        # raw zero pose extends horizontally over the table edge and must be avoided.
        for arm in ("A", "B"):
            home = HOME_JOINTS[arm]
            for i, suffix in enumerate(
                ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
            ):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}.{suffix}")
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr] = home[i]
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}.{suffix}"
                )
                self.data.ctrl[act_id] = home[i]
            grip_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}.gripper")
            grip_ctrl = self._aperture_to_ctrl(arm, home[5])
            self.data.qpos[self.model.jnt_qposadr[grip_jid]] = grip_ctrl
            grip_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}.gripper")
            self.data.ctrl[grip_act] = grip_ctrl

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
        """Command both arms to HOME_JOINTS smoothly over 1.0 s in CONTROL_HZ increments."""
        q_start = self.qpos_12()
        q_home = np.concatenate([HOME_JOINTS["A"], HOME_JOINTS["B"]], dtype=np.float64)
        steps = CONTROL_HZ
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
        return self.camera_rig.render(camera, self.data)

    def render_depth(self) -> np.ndarray:
        """Render float32 depth map in meters from the overhead camera."""
        return self.camera_rig.render_depth(self.data)

    def camera_intrinsics(self, camera: str) -> np.ndarray:
        """Compute 3x3 intrinsic matrix K from camera vertical field-of-view."""
        if camera not in CAMERA_NAMES:
            raise SceneBuilderError(f"unknown camera name: {camera}")
        return self.camera_rig.intrinsics(camera)

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
        return calc_fill_fraction(self.model, self.data, container)

    def settle(self, seconds: float) -> None:
        """Advance physics holding the current control targets so free objects come to rest."""
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
