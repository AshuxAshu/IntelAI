"""Render a single front view of the whole dinner-table setup.

The world frame puts the operator at -Y (both arms are front-mounted and face
+Y), so "front" is a free camera at azimuth 90 — on the -Y side looking across
the table.

Because the deliverable is one still with no motion to expose what is missing,
the framing is measured, not guessed:

* The camera is pulled back until nothing in the setup touches the frame edge,
  so nothing can be silently clipped.
* The frame is then cropped to the setup's own silhouette (the world floor and
  wall are excluded, being background planes), so the setup fills the image.
* A segmentation pass reports the pixel count of every body of interest; any
  body below ``--min-pixels`` is listed and the script exits nonzero.

Usage:
  uv run python scripts/demo/render_setup_shot.py
  uv run python scripts/demo/render_setup_shot.py --seed 7 --profile dr_train \
      --out artifacts/demo/setup_front.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import make_scene

# Bodies that must be visible for the shot to count as "the whole setup".
REQUIRED_BODIES = (
    "table", "cabinet", "drawer_top",
    "A.base", "A.upper_arm", "A.lower_arm", "A.gripper",
    "B.base", "B.upper_arm", "B.lower_arm", "B.gripper",
    "plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2",
    "placemat_1_body", "placemat_2_body",
)

# Background planes of the world body: excluded from the framing silhouette so
# they cannot make the setup look like it fills the frame.
BACKGROUND_BODIES = ("world",)

# Free-standing decor. It still renders (it is part of the scene), but it is not
# part of the dinner-table setup, so it is excluded from the silhouette the
# framing is fitted to. Including it would let a picture on the back wall shrink
# the table to a corner of the image.
DECOR_BODIES = ("wall_picture", "plant")


def background_geoms(model: mujoco.MjModel, names: tuple[str, ...] = BACKGROUND_BODIES) -> set[int]:
    """Geom ids belonging to the named background/decor bodies."""
    ids: set[int] = set()
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) == bid:
                ids.add(g)
    return ids


def geom_world_aabb(model: mujoco.MjModel, data: mujoco.MjData,
                    geom_ids) -> tuple[np.ndarray, np.ndarray]:
    """Exact-ish world AABB: mesh vertices where available, else geom size."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in geom_ids:
        gtype = int(model.geom_type[g])
        if gtype == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[g] >= 0:
            mid = int(model.geom_dataid[g])
            adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            local = np.asarray(model.mesh_vert[adr:adr + num], dtype=np.float64)
            rot = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            world = local @ rot.T + np.asarray(data.geom_xpos[g], dtype=np.float64)
            lo = np.minimum(lo, world.min(axis=0))
            hi = np.maximum(hi, world.max(axis=0))
        else:
            centre = np.asarray(data.geom_xpos[g], dtype=np.float64)
            size = np.asarray(model.geom_size[g], dtype=np.float64)
            r = float(np.max(size))
            lo = np.minimum(lo, centre - r)
            hi = np.maximum(hi, centre + r)
    return lo, hi


def render_seg(model: mujoco.MjModel, data: mujoco.MjData,
               cam: mujoco.MjvCamera, w: int, h: int) -> np.ndarray:
    r = mujoco.Renderer(model, h, w)
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=cam)
    seg = r.render()
    r.close()
    return np.asarray(seg[:, :, 0])


def setup_mask(seg: np.ndarray, bg: set[int]) -> np.ndarray:
    """Foreground mask with background planes removed."""
    mask = seg != -1
    for g in bg:
        mask &= seg != g
    return mask


def crop_box(mask: np.ndarray, aspect: float, margin: float) -> tuple[int, int, int, int] | None:
    """Largest centred box of `aspect` ratio enclosing the mask, or None if clipped."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if x0 <= 0 or y0 <= 0 or x1 >= w - 1 or y1 >= h - 1:
        return None                      # touches the edge -> need more distance
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw = (x1 - x0 + 1) * margin
    bh = (y1 - y0 + 1) * margin
    if bw / bh > aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    nx0, ny0 = round(cx - bw / 2), round(cy - bh / 2)
    nx1, ny1 = round(cx + bw / 2), round(cy + bh / 2)
    if nx0 < 0 or ny0 < 0 or nx1 > w or ny1 > h:
        return None
    return nx0, ny0, nx1, ny1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-14.0)
    ap.add_argument("--fovy", type=float, default=45.0)
    ap.add_argument("--margin", type=float, default=1.14,
                    help="padding factor around the setup silhouette")
    ap.add_argument("--distance", type=float, default=1.6,
                    help="starting distance; increased until nothing is clipped")
    ap.add_argument("--min-pixels", type=int, default=150,
                    help="minimum segmentation pixels for a body to count as visible")
    ap.add_argument("--drawer-open", action="store_true",
                    help="slide the drawer open first so the cutlery inside it is "
                         "visible; the scene's initial state has the drawer closed, "
                         "so the shot is annotated accordingly")
    ap.add_argument("--out", type=Path, default=Path("artifacts/demo/setup_front.png"))
    args = ap.parse_args()

    scene = make_scene(args.seed, args.profile)
    model, data = scene.model, scene.data
    model.vis.global_.fovy = float(args.fovy)   # free cameras read the model FOV
    # The offscreen framebuffer is sized at scene compile time and caps the
    # renderer, so widen it for a full-resolution still.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)

    if args.drawer_open:
        # Drive the drawer along its slide by servo, exactly as the open_drawer
        # skill does, so the shot shows the real mechanism rather than a pose.
        from dinner_table.contracts.geometry import DRAWER_TRAVEL
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_actuator")
        steps = round(3.0 * 500)            # 3 s at the 500 Hz physics rate
        for i in range(steps + 1):
            data.ctrl[act] = DRAWER_TRAVEL * (i / steps)
            mujoco.mj_step(model, data)
        scene.settle(0.5)
        print(f"drawer opened (actuator {act} -> {DRAWER_TRAVEL} m)")

    bg = background_geoms(model) | background_geoms(model, DECOR_BODIES)
    setup_geoms = [g for g in range(model.ngeom) if g not in bg]
    lo, hi = geom_world_aabb(model, data, setup_geoms)
    print(f"setup AABB: x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] "
          f"z[{lo[2]:+.3f},{hi[2]:+.3f}]")
    lookat = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0)
    print(f"lookat: {np.round(lookat, 3)}")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth = float(args.azimuth)
    cam.elevation = float(args.elevation)

    aspect = args.width / args.height
    distance = float(args.distance)
    seg = None
    box = None
    for attempt in range(10):
        cam.distance = distance
        seg = render_seg(model, data, cam, args.width, args.height)
        mask = setup_mask(seg, bg)
        box = crop_box(mask, aspect, args.margin)
        if box is not None:
            print(f"framing ok at distance {distance:.2f} m (attempt {attempt + 1})")
            break
        distance *= 1.18
        print(f"  clipped at {distance / 1.18:.2f} m -> backing off to {distance:.2f} m")
    if box is None:
        print("ERROR: could not fit the setup without clipping")
        return 2

    # --- visibility report from the final framing ---
    mask = setup_mask(seg, bg)
    missing: list[str] = []
    report: list[tuple[str, int]] = []
    for name in REQUIRED_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            missing.append(f"{name} (absent from model)")
            continue
        px = int(np.isin(seg, [g for g in setup_geoms
                               if int(model.geom_bodyid[g]) == bid]).sum())
        report.append((name, px))
        if px < args.min_pixels:
            missing.append(f"{name} ({px} px)")

    print("\nvisible bodies (segmentation pixels):")
    for name, px in sorted(report, key=lambda t: t[1]):
        print(f"  {'  ' if px >= args.min_pixels else '!!'} {name:18s} {px:6d}")

    # --- the still ---
    renderer = mujoco.Renderer(model, args.height, args.width)
    renderer.update_scene(data, camera=cam)
    rgb = np.asarray(renderer.render())
    renderer.close()
    image = Image.fromarray(rgb).crop(box)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"\nwrote {args.out} ({image.width}x{image.height}, crop {box})")

    if missing:
        print(f"\nWARNING: {len(missing)} body/bodies not visible:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print("\nall required bodies visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
