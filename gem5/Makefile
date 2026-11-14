# EEL5764 Group 4 -- scratchpad saturation of N x N systolic arrays, gem5.
#
#   make help        list the targets
#   make all         build everything, run all simulations, plot (~2.5 h)
#
# Settings: give them on the command line (make data SIM_JOBS=8) or put
# them in env.local.sh (see env.local.sh.example).
#   GEM5_DIR     gem5 checkout (default third_party/gem5, cloned by make gem5)
#   PYTHON       Python with requirements.txt (.venv if make venv ran,
#                else python3)
#   RESULTS_DIR  where make data / make figures write (results/)
#   CHECK_REF    reference results folder for make check
#   SIM_JOBS     gem5 runs in parallel for make data (12)
#   BUILD_JOBS   compile jobs for make gem5 (nproc - 2)

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.NOTPARALLEL:
.DELETE_ON_ERROR:

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ENV := cd $(ROOT); source scripts/env.sh;
# A setting as scripts/env.sh resolves it, honouring make VAR=... overrides.
SETTINGS := GEM5_DIR GEM5_BIN HOST_BIN WORKLOAD_ROOT BUILD_DIR
envvar = $(shell $(foreach v,$(SETTINGS),$(if $($(v)),$(v)='$($(v))')) \
    bash -c 'source $(ROOT)/scripts/env.sh >/dev/null 2>&1; echo "$$$(1)"')

GEM5_BIN := $(call envvar,GEM5_BIN)
HOST_BIN := $(call envvar,HOST_BIN)
WL_INDEX := $(call envvar,WORKLOAD_ROOT)/out/workloads.json

# Arguments of some targets (make run WL=bert_ffn1 N=32 BANKS=21, ...)
WL ?= bert_attn_qk
N ?= 16
BANKS ?= 8
ARGS ?=
SUITES ?=
PLOTS ?=
PAPER_SOURCE ?= rtlcfg
FORCE ?=
REF ?=

.PHONY: help all deps venv build gem5 host zig workloads prepacked smoke run \
        list data figures paper-figures model check regress clean distclean

help:  ## this list
	@echo "Targets (run in this order the first time, or just: make all):"
	@grep -E '^[a-z][a-z0-9_-]*:.*## ' $(MAKEFILE_LIST) | \
	    sed -E 's/^([a-z0-9_-]+):.*## /  make \1\t/' | expand -t 18
	@echo
	@echo "Examples:"
	@echo "  make run WL=bert_ffn1 N=32 BANKS=21 ARGS='--bank-policy reads_first'"
	@echo "  make data SUITES='main knee' SIM_JOBS=8"
	@echo "  make data FORCE=1          # rerun after changing the model"
	@echo "  make figures PLOTS='fig1 fig17 tables'"
	@echo "  make check REF=path/to/other/results"
	@echo "  make run WL=resnet50_conv3x N=16 BANKS=8 ARGS='--stall-model lockstep'"

all: deps build data figures  ## everything: check, build, simulate (about 2 h), plot

venv:  ## Python packages (requirements.txt) in .venv, used automatically
	cd $(ROOT); python3 -m venv .venv; \
	    .venv/bin/pip install --quiet --upgrade pip; \
	    .venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt; \
	    echo "installed requirements.txt into .venv ($$(.venv/bin/python3 -V))"

deps:  ## check that every tool and package is installed (seconds)
	$(ENV) bash scripts/check_deps.sh

build: $(GEM5_BIN) host workloads  ## gem5 (if not built yet) + host program + workloads

gem5:  ## fetch gem5 v25.1.0.1, apply the patch, build gem5.opt (20-30 min once)
	$(ENV) bash scripts/check_deps.sh --build && bash scripts/build_gem5.sh

# Built when missing or older than the accelerator's sources; make gem5
# always runs scons.
ACCEL_SRC := $(wildcard $(ROOT)/gem5_model/src/accel/*) \
             $(wildcard $(ROOT)/gem5_model/patches/*.patch)
$(GEM5_BIN): $(ACCEL_SRC)
	$(ENV) bash scripts/check_deps.sh --build && bash scripts/build_gem5.sh \
	    && touch "$@"

host: $(HOST_BIN)  ## cross-compile the RISC-V host program

$(HOST_BIN): gem5_model/host/systolic_host.c scripts/build_host.sh
	$(ENV) bash scripts/build_host.sh

zig:  ## optional: RISC-V compiler (zig 0.16.0) in third_party/zig-venv
	$(ENV) "$$PYTHON" -m venv third_party/zig-venv; \
	    third_party/zig-venv/bin/pip install --quiet --disable-pip-version-check ziglang==0.16.0; \
	    echo "zig $$(third_party/zig-venv/bin/python -m ziglang version)" \
	         "in third_party/zig-venv"

workloads: $(WL_INDEX)  ## generate the workload files (seeded, about 40 MB)

$(WL_INDEX): workloads/gen_workloads.py workloads/workloads.yaml
	$(ENV) "$$PYTHON" workloads/gen_workloads.py

# Packed weight tiles for --prepacked-weights (about 150 MB).
PREPACKED := $(dir $(WL_INDEX)).prepacked

prepacked: $(PREPACKED)  ## packed weights for --prepacked-weights

$(PREPACKED): $(WL_INDEX)
	$(ENV) "$$PYTHON" workloads/gen_workloads.py --prepack
	touch "$@"

smoke: $(HOST_BIN) $(WL_INDEX) | $(GEM5_BIN)  ## quick test: 8 small runs, all must verify
	$(ENV) bash scripts/run_all_workloads.sh 8 4 --rtl

run: $(HOST_BIN) $(WL_INDEX) | $(GEM5_BIN)  ## one run: WL=, N=, BANKS=, ARGS=
	$(ENV) bash scripts/run_workload.sh $(WL) $(N) $(BANKS) $(ARGS)

list:  ## the experiment suites and their run counts
	$(ENV) "$$PYTHON" analysis/collect.py --list

data: $(HOST_BIN) $(WL_INDEX) $(PREPACKED) | $(GEM5_BIN)  ## run the experiments -> results/data/ (SUITES=, FORCE=1)
	$(ENV) "$$PYTHON" analysis/collect.py -j "$$SIM_JOBS" \
	    $(if $(FORCE),--force) $(SUITES)

figures:  ## figures, tables, key_results.json -> results/ (PLOTS=)
	$(ENV) "$$PYTHON" analysis/plot_curves.py $(if $(PLOTS),--only $(PLOTS))

paper-figures:  ## the paper's figures, from rtlcfg_* (PAPER_SOURCE=baseline: one port)
	$(ENV) "$$PYTHON" analysis/plot_curves.py --paper $(PAPER_SOURCE)

model:  ## print the analytical model's predictions (Table I)
	$(ENV) "$$PYTHON" analysis/model.py

check:  ## compare results/ with a reference copy (REF=dir)
	$(ENV) ref="$(REF)"; ref="$${ref:-$$CHECK_REF}"; \
	if [ -z "$$ref" ]; then \
	    echo "give a reference results folder: make check REF=dir" \
	         "(or set CHECK_REF in env.local.sh)" >&2; \
	    exit 1; \
	fi; \
	"$$PYTHON" analysis/compare_results.py "$$ref" "$$RESULTS_DIR"

regress: $(HOST_BIN) $(WL_INDEX) | $(GEM5_BIN)  ## default model vs analysis/golden_stats.csv, exactly (1 min)
	$(ENV) bash analysis/regress.sh

clean:  ## delete build/ (raw gem5 runs and scratch output)
	rm -rf $(ROOT)/build

distclean: clean  ## also delete results/, workloads, host, .venv, third_party/
	rm -rf $(ROOT)/results $(ROOT)/workloads/out $(ROOT)/third_party \
	    $(ROOT)/.venv $(HOST_BIN)
