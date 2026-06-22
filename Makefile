SHELL := /bin/bash

ROOT := $(CURDIR)
UV_CACHE_DIR ?= $(ROOT)/.uv-cache
MPLCONFIGDIR ?= /tmp
export MEMFAIL_DATA_ROOT

.PHONY: help restore-caches compile task1-smoke task1-full-a task1-full-b task1-full-ab task2-agent-quick task2-agent task2-best-submission

help:
	@echo "OS Lab 7 memory failure prediction"
	@echo "  make restore-caches"
	@echo "  make compile"
	@echo "  make task1-smoke MAX_FILES=300"
	@echo "  make task1-full-a"
	@echo "  make task1-full-b"
	@echo "  make task1-full-ab"
	@echo "  make task2-agent-quick"
	@echo "  make task2-agent"
	@echo "  make task2-best-submission"
	@echo "Optional: MEMFAIL_DATA_ROOT=/path/to/data"

restore-caches:
	@find m2mfp-reproduction memory-failure-prediction-agent -name '*.csv.gz' -print -exec gzip -dkf {} \;

compile:
	$(MAKE) -C m2mfp-reproduction compile UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)
	$(MAKE) -C memory-failure-prediction-agent compile UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)

task1-smoke:
	$(MAKE) -C m2mfp-reproduction task1-smoke UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR) MAX_FILES=$(MAX_FILES)

task1-full-a:
	$(MAKE) -C m2mfp-reproduction task1-full-a UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)

task1-full-b:
	$(MAKE) -C m2mfp-reproduction task1-full-b UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)

task1-full-ab:
	$(MAKE) -C m2mfp-reproduction task1-full-ab UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)

task2-agent-quick:
	$(MAKE) -C memory-failure-prediction-agent task2-agent-quick UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)

task2-agent:
	$(MAKE) -C memory-failure-prediction-agent task2-agent UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)

task2-best-submission:
	$(MAKE) -C memory-failure-prediction-agent task2-best-submission UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)
