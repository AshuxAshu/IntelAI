PY ?= uv run
SEED ?= 42
SEEDS ?= 0-9

# Targets below run against this tree as it stands: smoke, scene, eval,
# eval-canonical, bench, videos, demo, dataset.
#
# The plan's learned-policy evaluation, runtime-integration (`physicalai run`)
# and release pipelines are not implemented in this tree, so there are
# deliberately no targets for them here. Previously this file pointed
# `eval-policy`, `demo` and `release` at modules that do not exist
# (`dinner_table.eval.harness`, `dinner_table.runtime.demo`,
# `dinner_table.release`), so every one of those commands died on an
# ImportError.
.PHONY: smoke scene eval eval-canonical bench bench-ov videos demo dataset

smoke:
	$(PY) python -m dinner_table.smoke

scene:
	$(PY) python -m dinner_table.scene.contact_sheet --seed $(SEED) --profile dr_train

# Per-graph success and first-attempt rates over randomized seeds, run against
# the privileged teacher oracle (ground-truth grounding, no perception or
# learned policy in the loop). These three graphs are the ones that complete
# today; the report lands in artifacts/eval/teacher_report.json.
eval:
	$(PY) python -m dinner_table.eval.teacher_eval \
		--graphs drawer_cycle,mug_setting,plate_setting --seeds $(SEEDS)

# The flagship dinner-table graph, reported separately because it currently
# stops at the plate placement: the open drawer's front wall overlaps
# placemat_1, so the descend finds the drawer edge instead of the table and
# fails with place/descend/no_support. Kept as its own target so `make eval`
# stays green and this regression is not hidden. Exits nonzero on failure.
eval-canonical:
	$(PY) python -m dinner_table.eval.teacher_eval --graphs dinner_canonical --seeds $(SEEDS)

# Deliverable 3: latency, throughput, device selection and precision per model.
bench:
	$(PY) python -m dinner_table.bench.bench_intel --models all --devices auto

# The optimization matrix in one command: exports the whole precision ladder
# from one checkpoint (training the public stand-in when none is given), times
# every rung on CPU/iGPU/NPU, and checks each rung's actions against the FP32
# reference. Writes bench/ov_matrix/ (markdown + CSV + JSON).
bench-ov:
	$(PY) python scripts/benchmark_openvino.py

# Multi-camera MP4s (demo_cam | overhead | wrist_A | wrist_B) of pick episodes on
# this seed, with the outcome burned in. Failed episodes are rendered too and
# marked FAILED, so the video shows what actually happens.
videos:
	$(PY) python scripts/render_pick_videos.py --profiles dr_train --seeds $(SEED) --force

# The visible demonstration: rendered episodes plus the scene contact sheet.
demo: videos scene

# Teacher demonstrations: rolls the oracle under DR and noise, one EpisodeLog
# JSON per episode under demos/<split>/ plus a run manifest. COUNT=3400 is the
# full dataset-plan budget (days of CPU); pass a small COUNT for a smoke slice.
COUNT ?= 3400
dataset:
	$(PY) python -m dinner_table.data.demo_gen --config dr_train --count $(COUNT) --out demos
