"""CLI: step an empty MuJoCo model headlessly and save a PNG. Phase 0 gate."""

from __future__ import annotations

import argparse
import pathlib

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/smoke.png", type=pathlib.Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    spec = mujoco.MjSpec()
    spec.modelname = "smoke"
    body = spec.worldbody.add_body(name="ball", pos=[0, 0, 0.05])
    body.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0])
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, 128, 128)
    renderer.update_scene(data, camera=-1)
    frame = renderer.render()
    if frame.shape != (128, 128, 3):
        raise RuntimeError(f"unexpected frame shape {frame.shape}")
    if float(frame.mean()) < 1.0:
        raise RuntimeError("render is black - rendering pipeline broken")
    import PIL.Image

    PIL.Image.fromarray(frame).save(args.out)
    print(f"smoke ok: wrote {args.out}")


if __name__ == "__main__":
    main()
