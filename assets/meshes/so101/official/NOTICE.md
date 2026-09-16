# Third-party robot assets

The files under `assets/so101` originate from Google DeepMind's MuJoCo Menagerie, `robotstudio_so101`, commit `ac6b2b09983786f3036cab1000221017fa2193b4`.

- Source: [The Robot Studio SO101 Description (MJCF)](https://github.com/google-deepmind/mujoco_menagerie/tree/ac6b2b09983786f3036cab1000221017fa2193b4/robotstudio_so101)
- License: [Apache License 2.0](assets/so101/LICENSE)
- Original source and hashes: [provenance.json](assets/so101/provenance.json)

Original meshes, XML, license, and README are retained. `assets/so101/assets/lod` contains modified, simplified visual meshes generated using trimesh and fast-simplification. Original collision meshes and source model inertias are retained. Runtime composition changes arm names, base poses and visual colors, and adds laboratory/dinner geometry and cameras. These modifications are not an endorsement or certification by the original authors.

The dinner assets in `dinner.py` and `assets/dinner` were created procedurally for this project on September 11, 2026 with Codex assistance. They use the repository's MIT license and contain no third-party dinnerware meshes or image textures. `assets/dinner/manifest.json` identifies the generator and exported files.

Installed software retains its respective upstream licenses. Python dependencies are recorded in the repository's requirements files.
