# Paths and settings shared by every script.  Sourced, not run:
#
#   source scripts/env.sh
#
# The Makefile sources it before each step.  Run it yourself only when you
# call a script or gem5 directly.
#
# Machine-specific settings go in env.local.sh at the repository root
# (copy env.local.sh.example; git ignores the copy).  A variable that is
# already set, e.g. on the make command line (make GEM5_DIR=~/gem5 ...),
# wins over both env.local.sh and the defaults below.

SYSTOLIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SYSTOLIC_ROOT

if [ -f "$SYSTOLIC_ROOT/env.local.sh" ]; then
    # shellcheck source=/dev/null
    source "$SYSTOLIC_ROOT/env.local.sh"
fi

# gem5 checkout the accelerator is compiled into, and the release it must be.
export GEM5_DIR="${GEM5_DIR:-$SYSTOLIC_ROOT/third_party/gem5}"
export GEM5_REPO="${GEM5_REPO:-https://github.com/gem5/gem5.git}"
export GEM5_VERSION="${GEM5_VERSION:-v25.1.0.1}"
export GEM5_BIN="${GEM5_BIN:-$GEM5_DIR/build/RISCV_SYSTOLIC/gem5.opt}"

# Python for the workload generator, the analysis and the plots.  It needs
# the packages in requirements.txt: the .venv that `make venv` creates, if
# there is one, else python3.  gem5 itself embeds its own Python.
if [ -z "${PYTHON:-}" ] && [ -x "$SYSTOLIC_ROOT/.venv/bin/python3" ]; then
    PYTHON="$SYSTOLIC_ROOT/.venv/bin/python3"
fi
export PYTHON="${PYTHON:-python3}"

# Generated files.
export WORKLOAD_ROOT="${WORKLOAD_ROOT:-$SYSTOLIC_ROOT/workloads}"
export HOST_BIN="${HOST_BIN:-$SYSTOLIC_ROOT/gem5_model/host/systolic_host}"
export BUILD_DIR="${BUILD_DIR:-$SYSTOLIC_ROOT/build}"
# Where make data / make figures write (not in git).  Point it elsewhere,
# e.g. at a documents folder, to generate the results there directly.
export RESULTS_DIR="${RESULTS_DIR:-$SYSTOLIC_ROOT/results}"
# Reference results for make check (another copy of a results folder).
export CHECK_REF="${CHECK_REF:-}"

# Parallel gem5 runs for `make data` (each run is one process).
_n="$(nproc 2>/dev/null || echo 2)"
export SIM_JOBS="${SIM_JOBS:-$(( _n > 14 ? 12 : (_n > 2 ? _n - 2 : 1) ))}"
unset _n
