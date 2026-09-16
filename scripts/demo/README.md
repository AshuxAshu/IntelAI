# Per-seed demonstration videos

Generates the labelled, multi-camera demonstration assets: for every seed, one
video per *run* that completes on this checkout. Each video shows four camera
angles with the run's live OpenVINO telemetry underneath.

## What a video contains

```
+---------------------------+---------------------------+
| demo_cam (third-person)   | overhead                  |
+---------------------------+---------------------------+
| wrist_A (arm A jaw)       | wrist_B (arm B jaw)       |
+---------------------------+---------------------------+
| latency sparkline + p50/p95   | ACT latency by precision  |
| inferences, ips, CPU%, RSS    | run name, step chips,     |
| top detections                | phase, t, outcome         |
+---------------------------+---------------------------+
```

* **`Seed-N`** is burned into the top-right of the scene area.
* Every camera panel is **named** (the run's purpose is deliberately *not*
  drawn over the video; it lives in the manifest).
* The bottom strip is **live OpenVINO telemetry**: the project's real
  `OpenvinoDetector` (YOLO stand-in on the OpenVINO runtime, CPU) is run
  against the live overhead camera frame once every `OV_EVERY` physics ticks,
  and each inference's wall-clock latency is recorded. The sparkline, `p50`,
  `p95`, inference count and throughput are computed from those in-loop
  measurements — they are not copied from a report.
* The ACT bars are a **recorded** reference (this host's committed benchmark
  run), labelled as such in the panel title, so live detector numbers and
  recorded policy numbers are never conflated.

## Commands

```bash
# everything, all 10 seeds (long: ~2-4 h wall)
uv run python scripts/demo/render_demo_videos.py --seeds 0-9

# one run, a few seeds (quick check)
uv run python scripts/demo/render_demo_videos.py --runs mug --seeds 0-2

# without the live OpenVINO sampling
uv run python scripts/demo/render_demo_videos.py --no-ov --seeds 0

# which runs actually complete, without rendering (fast, no video)
uv run python scripts/demo/probe_runs.py --seeds 0-9
```

Output: `artifacts/demo/videos/<run>/seed<NN>.mp4` plus
`artifacts/demo/videos/manifest.json` recording, per (run, seed), every step's
outcome, frame count, sim time, and the OpenVINO statistics observed during
that episode.

## The runs, and why they are split this way

Composition is measured, not assumed (`scripts/demo/probe_runs.py` is the
evidence). Steps that share a scene can conflict — an object's landing site or
transit lane can be blocked by another object or by the open drawer — so
whatever combines is combined, and the rest stays separate.

| run | steps | notes |
| --- | --- | --- |
| `table_setting` | pick+place plate, then pick+place mug | one continuous episode; two arms, landing sites 0.33 m apart |
| `plate` | pick+place plate | the plate alone |
| `mug` | pick+place mug | the mug alone |
| `bottle` | pick bottle | placement unsupported (see below); the pick's end-hold is the demo |
| `drawer` | open, pick+place fork, open, pick spoon | both utensils in one episode |
| `drawer_fork` | open, pick+place fork | the fork alone, drawer closed by the placement |
| `attempt_spoon_place` | open, pick spoon, place spoon | **expected to fail**; rendered so the limitation is on camera |
| `attempt_bottle_place` | pick bottle, place bottle | **expected to fail**; rendered so the limitation is on camera |

The two `attempt_*` runs are flagged `expected_failure: true` in the manifest,
so a red FAILED overlay on those is read as the documented limitation rather
than as a regression. Every other run's failures are genuine per-seed
outcomes and are marked the same way.

### Why some steps are NOT combined

* **Drawer + plate.** The open drawer's front wall advances to `y = -0.099` at
  `x = -0.24` with a half-width of ~0.15 m, so its footprint spans
  `x ≈ -0.39…-0.09` and overlaps `placemat_1` (`x = -0.06`). With the drawer
  open, a plate descend finds the drawer edge instead of the table and fails
  `place/descend/no_support`. This is topology, not a bug in the run.
* **Spoon placement.** The spoon never satisfies the placement verifier (8 mm
  XY / 3 mm Z): every target tried fails `verify/misplaced` or
  `path_blocked` (the plate's sprawl crosses its lane). The `drawer` run
  therefore ends holding the spoon, which still demonstrates the drawer cycle
  and the retrieval.
* **Bottle placement.** The bottle *pick* succeeds on every seed, but *place*
  fails on every target tried (all named settings plus a 6x4 grid over arm A's
  reachable envelope): the neck pinch cannot be re-oriented over the table. The
  bottle run shows the pick and hold.
* **Trailing `close_drawer`.** After a cutlery placement the drawer is already
  shut — `Place`'s cutlery path slides it closed by servo — so an explicit
  `close_drawer` afterwards fails `pregrasp/already_closed`. It is omitted
  rather than rendered as a spurious failure.

## Head comparison

The videos are generated from **this checkout** (`main` @ the commit the sync
landed on, plus the merge-damage fixes described below).

The user-suggested fallback commit `c11df09` was evaluated and is **strictly
less capable**, so no separate asset set was generated from it. Verified on
that tree:

* It has no `teacher/teacher_policy.py` or `teacher/task_graphs.py`, so there
  is no task-graph runner to drive a multi-step run at all.
* Bottle pick — which succeeds on every seed here — fails outright there with
  `pregrasp/ik_unreachable` (3/3 seeds tried).
* Plate place fails there with `verify/misplaced` (3/3 seeds tried); mug place
  does work there.

Note on this: `PLACEMATS` **is** correctly imported at `c11df09` (line 22 of its
`skills.py`). The undefined-name error that broke every `place` step was
introduced *later*, by a merge into `main` — it is merge damage in `main`, not
something inherited from `c11df09`.

Three defects were repaired on this checkout because they were merge damage
that blocked *all* multi-step runs, not tuning:

1. `skills.py` used `PLACEMATS` in `Place._target_xyz` without importing it —
   every `place` step died with `NameError`. The import exists at `c11df09` and
   at `82868f3`; it was lost in the later merge into `main`.
2. `Place._flatten_shift` had been overwritten with a stale `_target_xyz` body
   (referencing an undefined `ctx`), so the descend target collapsed to ~0 and
   every placement failed `hover/ik_unreachable`. Restored to the intended
   `_flatten_rotation`-based descend from `7981e5e`.
3. `_recover` was called with 4 arguments but defined with 3, so any step that
   needed a retry crashed the eval. Pre-existing on both heads, and it made the
   eval CLI unusable.
4. The `make eval` / `make demo` targets pointed at modules that do not exist
   (`eval.harness`, `runtime.demo`, `data.demo_gen`, `release`); the Makefile
   now targets modules that exist.

With those fixes the full test suite is green (137 passed), against 9 failures
before.

## Reproducing the plate's borderline seeds

`plate`/`table_setting` fail on seeds 5 and 9 on some runs and pass on others.
Measurement shows the failure is marginal and z-related: on those seeds the
plate lands within ~1.0 mm in XY and ~3.3 mm high, against a 3.0 mm Z
tolerance — a 0.3 mm miss, not a lost object. It is reported as a failure
rather than hidden; the videos show whichever outcome the episode produced.
