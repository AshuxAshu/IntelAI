PY ?= uv run
SEED ?= 42

.PHONY: smoke scene dataset train eval-policy eval bench run demo release

smoke:
	$(PY) python -m dinner_table.smoke

scene:
	$(PY) python -m dinner_table.scene.contact_sheet --seed $(SEED)

dataset:
	$(PY) python -m dinner_table.data.demo_gen --config configs/scene/dr_train.yaml
	$(PY) python -m dinner_table.data.lerobot_export --root datasets/dinner

train:
	$(PY) physicalai fit --config configs/physicalai/act_dinner.yaml

eval-policy:
	$(PY) python -m dinner_table.eval.harness --mode policy-skills

eval:
	$(PY) python -m dinner_table.eval.harness --mode full --seeds 0-9

bench:
	$(PY) python -m dinner_table.bench.bench_intel --models all --devices auto

run:
	$(PY) physicalai run --config configs/runtime/dinner_demo.yaml

demo:
	$(PY) python -m dinner_table.runtime.demo --seed $(SEED)

release:
	$(PY) python -m dinner_table.release
