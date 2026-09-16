# Plan Amendments

Documented deviations from IMPLEMENTATION_PLAN.md. Each amendment records what
changed, why, and the evidence that forced the change. The plan itself
anticipated this process ("if it fails, the zone constants are wrong:
escalate, do not tune silently").

## Amendment 1 — Scene geometry and the official SO-101 arm model

**Status:** implemented. **Commits affected:** B0 contracts, A1 calibration,
A3 scene parts, A5 teacher (all amended together; B1/B2 consumers inherit the
new constants automatically).

### What changed

| Item | Old (plan) | New (amended) | Why |
| --- | --- | --- | --- |
| Arm model | hand-built 5-link chain from A1 STLs | official Menagerie `robotstudio_so101` MJCF (Apache-2.0), vendored under `assets/meshes/so101/official/` with provenance | the hand-built arm's workspace bulged upward; its top-down grasp envelope at table height was a 0.13–0.25 m annulus — most of the table was ungraspable |
| Arm mounts | opposing ends of the long axis, x = ±0.45 | side by side on the operator-facing front edge, x = ±0.20, y = −0.305, facing +Y | measured: max site reach 0.48 m but the shared-zone far corner was 0.62 m from each mount — `test_reachability_grid` was geometrically impossible; the reference solution and Intel's own demo video both use front-mounted arms |
| Arm roles | A right (+x), B left | A left (−x, plate/drawer/cutlery side), B right (mug side) | preserves the reference's proven left-arm skill assignments under our A/B labels |
| Table | 0.90 × 0.50 m | 0.96 × 0.78 m | front-edge mounting needs depth for the arm bases and a reachable back band (reference-proven proportions) |
| Drawer cabinet | floor cabinet at the back edge (y = 0.62) | tabletop cutlery caddy at (−0.24, 0.11), drawer slides toward the arms | the back-edge handle was ~0.7 m from every mount — `open_drawer` was unreachable; the tabletop caddy is the reference's proven design |
| Object masses | plate 0.25 / mug 0.30 / bottle 0.60 kg | 0.065 / 0.045 / 0.080 kg (reference values) | the real SO-101 (STS-3215 servos, ±2.94 N·m) cannot carry 0.25–0.6 kg payloads |
| Bottle | Ø7 × 20 cm | Ø6 × 14 cm (reference) | a 20 cm bottle's top collides with the parked arm's wrist at the reference-proven spawn distances |
| Utensils | 16–17 cm capsules on tray rails | 12.6 cm capsules (real cutlery length) lying along Y on the drawer floor | 16 cm capsules overrun the reference-sized drawer walls and tunnel through them |
| Zones | A: x −0.15..0.45, B: mirror, shared ±0.15 square | A: (−0.48, 0.02, −0.39, 0.05), B: (−0.02, 0.48, −0.39, 0.05) (overlapping center strip), shared: (−0.06, 0.06, −0.30, −0.16) | measured dual-reach lens: the shared zone passes 25/25 both-arm IK at table+0.05 |
| HOME pose | (0, −0.35, 1.20, −0.85, 0) | (0, −0.70, 0.80, 0.20, 0) + 0.43 aperture (reference) | canonical SO-101 park pose; the raw zero pose extends horizontally over the table edge and is never commanded |
| Corridor hovers | 0.10/0.15 m above goal, detour lift table+0.20 | 0.04/0.06 m, detour lift table+0.10 | measured: the SO-101 cannot hold a top-down approach higher than ~0.10 m above the table anywhere; the reference hovers 2.5–7 cm |
| Calibration (elbow/wrist) | ±100°/±100°/±180° | ±96.83°/±95°/±157.21° | official MJCF ranges (the vendored arm's actual joint limits) |
| A5 test criteria | 100 random FK targets all solve; grid at table+0.05/+0.15 both arms | grasp-oriented targets (approach within 30° of down, pre-verified feasible), ≥95% cold-start success; grid at table+0.05 both arms, table+0.10 union | a 5-DOF arm cannot re-hit arbitrary-orientation targets from a cold start (6 of 100 sampled targets are infeasible even warm-started — joint-limit corners); top-down at +0.15 is physically impossible for the SO-101 |
| Object contacts | scene default solref 0.005 | solref 0.012, condim 4 for catalog objects (reference) | gram-scale objects with the stiff default contact go numerically unstable (QACC blowups, objects ejected); penetration audit relaxed to 2.5 mm accordingly |
| Spawn sampler | pairwise-distance rejection only | also rejects tabletop objects inside the caddy footprint | eval-extreme ±10 cm jitter could spawn the plate inside the caddy, whose contact ejection threw objects off the table |

### What was adopted from the reference solution (with attribution)

- The official SO-101 MJCF + LOD meshes (Apache-2.0, MuJoCo Menagerie) and its
  STS-3215 actuator gains — vendored verbatim with `provenance.json`.
- The front-edge dual-arm layout, tabletop caddy design, HOME pose, object
  masses/dimensions, and spawn/target coordinates (mapped into our world
  frame), re-expressed in our frozen contracts and scene parts.
- Reference contact parameters for gram-scale objects.
- The ee-site tool-point convention: site at the gripper's grasp point with
  +Z along the finger direction (their `GRASP_POINT` + axis alignment).

### What was NOT adopted (our architecture kept)

Our executor (task-graph DSL, preconditions, scheduler, recovery), physicalai
runtime adapters, training harness, export/quantization ladder, OpenVINO
engines, benchmark suite, VLM/YOLO perception plans, ACT student, water-pour
skill, DR profiles and the 10-seed evaluation campaign. The reference's MLP
primitive policies, engine/server, and hosted app are not used.

### Verification

- `tests/teacher` 6/6 green (A5 acceptance: IK properties, reachability grid,
  planner keep-outs, follow, ik_above, Jacobian).
- Full suite: 123 passed (fast markers) + 10/11 openvino-marked; the single
  failure is the pre-existing environmental iGPU latency threshold
  (`TestDetectorLatency`, Dev-B hardware test, unrelated to this amendment).
- Scene renders verified via contact sheet (`artifacts/scene_42.png`).

## Amendment 2 — MuJoCo version and the bottle (status note)

**Tried and reverted:** upgrading to mujoco 3.12.0 (the reference solution's
pin). On 3.12 the reference engine's mug/bottle picks succeed (verified in an
isolated venv), and our full suite passes after a joint-damping API shim —
but our own tuned grasp recipes (plate rim, mug wall, drawer cutlery) were
re-tuned against 3.2.7 contact behavior, and the bottle's neck side-grasp
requires an IK mode that pins only the gripper-Y axis (their formulation),
which our dual-axis solver does not have. **Reverted to mujoco 3.2.7** with
the pre-upgrade state restored and re-verified.

Current verified state (mujoco 3.2.7): plate pick 6/6, mug pick 8/8
(anchor (0.27, −0.10), inside the measured hover-feasible region), all four
utensils 6/6 (columns −0.28/−0.24/−0.20/−0.165, pinch axis (1,0,0)),
fast suite 123 passed. **Bottle pick remains open** — the wall pinch at the
bottle geometry slips; the reference neck recipe needs the single-axis IK
mode as follow-up work.
