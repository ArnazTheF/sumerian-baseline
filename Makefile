SVC ?= $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo gpu || echo cpu)
TASKS ?=
DC  ?= docker compose
CFG ?= configs/mt_byt5.yaml
O   ?=
RUN ?= runs/mt_byt5_small
TEXT ?= 1(disz) udu niga

export TASKS
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

PY = $(DC) run --rm $(SVC) python -m sumbase

.PHONY: help dirs build info smoke prepare train evaluate tm zeroshot memcheck mt translit ocr lite lite-translit lite-ocr lite-all nogpu predict tensorboard shell clean-smoke

help:
	@sed -n '1,12p' Makefile
	@echo "Сейчас выбран образ: $(SVC)"

dirs:
	@mkdir -p data runs hf_cache

build: dirs
	@echo ">>> Собираю образ: $(SVC)"
	$(DC) build $(SVC)

info: dirs
	$(PY) info

smoke: dirs
	$(DC) run --rm $(SVC) bash scripts/smoke_test.sh

prepare: dirs
	$(PY) prepare --config $(CFG) $(O)

train: dirs
	$(PY) train --config $(CFG) $(O)

evaluate: dirs
	$(PY) evaluate --config $(CFG) $(O)

tm: dirs
	$(PY) baseline --config $(CFG) --kind tm $(O)

zeroshot: dirs
	$(PY) baseline --config $(CFG) --kind zeroshot $(O)

memcheck: dirs
	$(PY) memcheck --config $(CFG) --budget-gb $(or $(BUDGET),12) $(O)

# --- лёгкий режим: всё укладывается в 12 ГБ видеопамяти (см. GPU_REQUIREMENTS.md) ---
lite: dirs
	$(PY) memcheck --config configs/lite/mt.yaml --budget-gb 12 $(O)
	$(PY) prepare  --config configs/lite/mt.yaml $(O)
	$(PY) baseline --config configs/lite/mt.yaml --kind tm $(O)
	$(PY) train    --config configs/lite/mt.yaml $(O)

lite-translit: dirs
	$(PY) memcheck --config configs/lite/translit.yaml --budget-gb 12 $(O)
	$(PY) prepare  --config configs/lite/translit.yaml $(O)
	$(PY) baseline --config configs/lite/translit.yaml --kind tm $(O)
	$(PY) train    --config configs/lite/translit.yaml $(O)

lite-ocr: dirs
	$(PY) memcheck --config configs/lite/ocr.yaml --budget-gb 12 $(O)
	$(PY) prepare  --config configs/lite/ocr.yaml $(O)
	$(PY) train    --config configs/lite/ocr.yaml $(O)

lite-all: lite lite-translit lite-ocr

nogpu: dirs
	$(DC) run --rm $(SVC) bash scripts/eval_no_gpu.sh

mt: dirs
	$(PY) prepare --config configs/mt_byt5.yaml $(O)
	$(PY) baseline --config configs/mt_byt5.yaml --kind tm $(O)
	$(PY) train --config configs/mt_byt5.yaml $(O)

translit: dirs
	$(PY) prepare --config configs/translit_byt5.yaml $(O)
	$(PY) baseline --config configs/translit_byt5.yaml --kind tm $(O)
	$(PY) train --config configs/translit_byt5.yaml $(O)

ocr: dirs
	$(PY) prepare --config configs/ocr_swin.yaml $(O)
	$(PY) train --config configs/ocr_swin.yaml $(O)

predict: dirs
	$(PY) predict --run $(RUN) --text "$(TEXT)"

tensorboard: dirs
	$(DC) run --rm -p 6006:6006 $(SVC) tensorboard --logdir runs --host 0.0.0.0 --port 6006

shell: dirs
	$(DC) run --rm $(SVC) bash

clean-smoke:
	rm -rf data/smoke runs/smoke_*
