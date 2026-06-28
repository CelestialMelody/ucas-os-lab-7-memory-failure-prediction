SHELL := /bin/bash

ROOT := $(CURDIR)
UV_CACHE_DIR ?= $(ROOT)/../.uv-cache
MPLCONFIGDIR ?= /tmp
export MEMFAIL_DATA_ROOT
PY := MPLCONFIGDIR=$(MPLCONFIGDIR) UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python
TIME_CMD ?= time
TIME := $(TIME_CMD) env MPLCONFIGDIR=$(MPLCONFIGDIR) UV_CACHE_DIR=$(UV_CACHE_DIR)

TASK1_SMOKE_CONFIG ?= configs/stage1_type_a_m2mfp_smoke.json
TASK1_FULL_A_CONFIG ?= configs/stage1_type_a_full.json
TASK1_FULL_B_CONFIG ?= configs/stage1_type_b_full.json
TASK1_FULL_AB_CONFIG ?= configs/stage1_type_ab_full.json
MAX_FILES ?= 300

.PHONY: help restore-caches compile figures task1-smoke task1-ablation task1-full-a task1-full-b task1-full-ab

help:
	@echo "M2-MFP reproduction commands"
	@echo "  make compile"
	@echo "  make figures"
	@echo "  make restore-caches"
	@echo "  make task1-smoke MAX_FILES=300"
	@echo "  make task1-ablation MAX_FILES=300"
	@echo "  make task1-full-a"
	@echo "  make task1-full-b"
	@echo "  make task1-full-ab"
	@echo "Optional: MEMFAIL_DATA_ROOT=/path/to/data"

restore-caches:
	@find results -name '*.csv.gz' -print -exec gzip -dkf {} \;

compile:
	$(PY) -m compileall src

figures:
	$(PY) ../scripts/make_figures.py --scope task1

task1-smoke:
	$(PY) src/raw_smoke.py --config $(TASK1_SMOKE_CONFIG) --max-files $(MAX_FILES)

task1-ablation:
	$(PY) src/ablation.py --config $(TASK1_SMOKE_CONFIG) --max-files $(MAX_FILES)

task1-full-a:
	$(TIME) uv run python src/ablation.py --config $(TASK1_FULL_A_CONFIG) --rebuild-features --max-files 0

task1-full-b:
	$(TIME) uv run python src/ablation.py --config $(TASK1_FULL_B_CONFIG) --rebuild-features --max-files 0

task1-full-ab:
	$(TIME) uv run python src/ablation.py --config $(TASK1_FULL_AB_CONFIG) --rebuild-features --max-files 0
