SHELL := /bin/bash

ROOT := $(CURDIR)
UV_CACHE_DIR ?= $(ROOT)/../.uv-cache
MPLCONFIGDIR ?= /tmp
export MEMFAIL_DATA_ROOT
PY := MPLCONFIGDIR=$(MPLCONFIGDIR) UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python
TIME_CMD ?= time
TIME := $(TIME_CMD) env MPLCONFIGDIR=$(MPLCONFIGDIR) UV_CACHE_DIR=$(UV_CACHE_DIR)

TASK2_CONFIG ?= configs/stage2_feature.json

.PHONY: help restore-caches compile task2-train task2-submission task2-agent task2-agent-quick task2-best-submission

help:
	@echo "Memory failure prediction agent commands"
	@echo "  make compile"
	@echo "  make restore-caches"
	@echo "  make task2-agent-quick"
	@echo "  make task2-train"
	@echo "  make task2-agent"
	@echo "  make task2-best-submission"
	@echo "Optional: MEMFAIL_DATA_ROOT=/path/to/data"

restore-caches:
	@find results -name '*.csv.gz' -print -exec gzip -dkf {} \;

compile:
	$(PY) -m compileall src

task2-train:
	$(TIME) uv run python src/stage2_feature_pipeline.py --config $(TASK2_CONFIG)

task2-submission:
	$(PY) src/stage2_feature_pipeline.py --config $(TASK2_CONFIG) --process submission

task2-agent:
	$(TIME) uv run python src/agent_search.py --config $(TASK2_CONFIG)

task2-agent-quick:
	$(PY) src/agent_search.py --config $(TASK2_CONFIG) --max-strategies 3

task2-best-submission:
	$(TIME) uv run python src/stage2_feature_pipeline.py --config $(TASK2_CONFIG) --process best_per_type_submission
