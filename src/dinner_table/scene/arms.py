"""Official SO-101 arm integration: merge the vendored Menagerie MJCF into the scene.

The arm model is the Apache-2.0 ``robotstudio_so101`` MJCF from MuJoCo Menagerie
(see ``assets/meshes/so101/official/provenance.json``), vendored verbatim. Both
arms are attached side by side on the operator-facing front edge of the table,
facing +Y, with every named element prefixed ``A.``/``B.`` so joint names match
the frozen ``JOINT_NAMES`` contract (``A.shoulder_pan`` ... ``B.gripper``).

The end-effector site ``{arm}.ee`` sits at the tool point between the finger
tips, with its +Z axis along the finger direction (gripper local -Z), so
``solve_ik`` approach vectors and the gripper agree on "approach axis".
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import ARM_MOUNTS, ARM_ORIENTATIONS

ARM_XML_PATH = Path("assets/meshes/so101/official/so101_arm.xml")
MESH_DIR_REL = "so101/official/assets"  # relative to the scene compiler meshdir
GRASP_SITE_POS = (0.003, 0.0, -0.100)  # tool point at the jaw tips' level (gripper frame)
GRASP_SITE_QUAT = (0.0, 0.0, 1.0, 0.0)  # 180 deg about Y: site +Z = gripper -Z (fingers), site +X = -gripper +X
# The site sits at the jaw tips' level: the jaw
# tip spheres straddle the site by +/-2.5 mm. At -0.092 the tips sat
# 5.5-9 mm BELOW every site target, so each cataloged offset had to be 8 mm
# higher — the gripper rode lower in the arm's envelope
# and a lifted fork could not clear the drawer walls (measured).


class ArmIntegrationError(DinnerTableError):
    """Exception raised when the vendored SO-101 MJCF cannot be merged."""


def _load_arm_parts() -> tuple[list[ET.Element], list[ET.Element], ET.Element, list[ET.Element]]:
    """Return (default children, asset children, base body, actuator elements)."""
    if not ARM_XML_PATH.is_file():
        raise ArmIntegrationError(f"arm xml missing: {ARM_XML_PATH}")
    root = ET.parse(ARM_XML_PATH).getroot()

    default_el = root.find("default")
    asset_el = root.find("asset")
    base_body = root.find("./worldbody/body")
    actuators = list(root.find("actuator")) if root.find("actuator") is not None else []
    if default_el is None or asset_el is None or base_body is None or not actuators:
        raise ArmIntegrationError("arm xml is missing default/asset/worldbody/actuator sections")
    return list(default_el), list(asset_el), base_body, actuators


def _prefer_lod(file_name: str) -> str:
    """Return the LOD mesh path when the vendored LOD variant exists."""
    lod = Path(ARM_XML_PATH).parent / "assets" / "lod" / file_name
    return f"{MESH_DIR_REL}/lod/{file_name}" if lod.is_file() else f"{MESH_DIR_REL}/{file_name}"


def expand_includes(root: ET.Element, base_dir: Path) -> None:
    """Recursively inline <include> fragments so the tree is one flat MJCF."""
    while True:
        includes = [(parent, el) for parent in root.iter() for el in parent.findall("include")]
        if not includes:
            return
        for parent, include_el in includes:
            fragment_path = base_dir / include_el.get("file")
            fragment = ET.parse(fragment_path).getroot()
            idx = list(parent).index(include_el)
            parent.remove(include_el)
            for child in list(fragment):
                parent.insert(idx, child)
                idx += 1


def _prefixed(element: ET.Element, prefix: str) -> ET.Element:
    """Deepcopy with every ``name`` attribute prefixed; wrist cam gets the contract name."""
    copy = deepcopy(element)
    for el in copy.iter():
        name = el.get("name")
        if name is None:
            continue
        if el.tag == "camera" and name == "wrist_cam":
            el.set("name", f"wrist_{prefix}")
        else:
            el.set("name", f"{prefix}.{name}")
    return copy


def _mount_quat(yaw: float) -> str:
    half = yaw * 0.5
    return " ".join(f"{v:.8g}" for v in (np.cos(half), 0.0, 0.0, np.sin(half)))


def attach_arms(scene_root: ET.Element) -> None:
    """Merge both prefixed SO-101 arms (defaults, assets, bodies, actuators) in place."""
    defaults, assets, base_body, actuators = _load_arm_parts()

    scene_default = scene_root.find("default")
    scene_asset = scene_root.find("asset")
    scene_worldbody = scene_root.find("worldbody")
    if scene_default is None or scene_asset is None or scene_worldbody is None:
        raise ArmIntegrationError("scene xml must define default/asset/worldbody sections")

    # Defaults and assets are shared by both arms; mesh files prefer LOD variants.
    scene_default.extend(deepcopy(defaults))
    for asset_el in deepcopy(assets):
        if asset_el.tag == "mesh":
            asset_el.set("file", _prefer_lod(asset_el.get("file")))
        scene_asset.append(asset_el)

    scene_actuator = scene_root.find("actuator")
    if scene_actuator is None:
        scene_actuator = ET.SubElement(scene_root, "actuator")

    for arm in ("A", "B"):
        body = _prefixed(base_body, arm)
        mount = ARM_MOUNTS[arm]
        body.set("pos", " ".join(f"{v:.8g}" for v in mount))
        body.set("quat", _mount_quat(ARM_ORIENTATIONS[arm]))

        gripper = next(el for el in body.iter("body") if el.get("name") == f"{arm}.gripper")
        ee_site = ET.SubElement(gripper, "site")
        ee_site.set("name", f"{arm}.ee")
        ee_site.set("pos", " ".join(f"{v:.8g}" for v in GRASP_SITE_POS))
        ee_site.set("quat", " ".join(f"{v:.8g}" for v in GRASP_SITE_QUAT))
        ee_site.set("size", "0.005")

        scene_worldbody.append(body)
        for actuator_el in actuators:
            prefixed = _prefixed(actuator_el, arm)
            prefixed.set("joint", prefixed.get("name"))  # name == joint after prefixing
            scene_actuator.append(prefixed)
