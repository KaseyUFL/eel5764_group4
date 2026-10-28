#!/usr/bin/env bash
# Check that everything the build and the experiments need is installed,
# in a few seconds, before anything long starts.
#
#   make deps                        # = bash scripts/check_deps.sh
#   bash scripts/check_deps.sh --build   # only what building gem5 needs
#
# Each line is one of
#   ok      present
#   todo    not built yet; the make target in brackets creates it
#   warn    works, but differs from the setup the results came from
#   MISSING required and absent: the line after it says how to fix it
# Exit status 1 if anything is MISSING.
#
# The gem5 checks run in the same toolchain the build uses (gem5_toolchain
# in env.sh), and the Python check compiles and runs a program against
# libpython the way gem5's configure step does.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ONLY_BUILD=0
[ "${1:-}" = "--build" ] && ONLY_BUILD=1
MISSING=0
ok()      { printf '  %-8s %s\n' ok "$*"; }
todo()    { printf '  %-8s %s\n' todo "$*"; }
warn()    { printf '  %-8s %s\n' warn "$*"; }
missing() { printf '  %-8s %s\n' MISSING "$1"; shift
            for fix in "$@"; do printf '  %-8s   fix: %s\n' "" "$fix"; done
            MISSING=$((MISSING + 1)); }
section() { printf '%s\n' "$*"; }
APT="sudo apt install"

# ------------------------------------------------------------ gem5 build
gem5_needs_build=1
[ -x "$GEM5_BIN" ] && gem5_needs_build=0
(
    gem5_toolchain
    section "Building gem5 (toolchain: $GEM5_TOOLCHAIN)"
    m=0
    CXX="${CXX:-g++}"
    if command -v "$CXX" >/dev/null; then
        v="$("$CXX" -dumpfullversion 2>/dev/null || "$CXX" -dumpversion)"
        major="${v%%.*}"
        if "$CXX" --version | grep -qi clang; then
            ok "$CXX $v (clang)"
        elif [ "$major" -ge 11 ] && [ "$major" -le 14 ]; then
            ok "$CXX $v (gem5 supports GCC 11-14)"
        else
            warn "$CXX $v: gem5 v25.1 supports GCC 11-14 only"
        fi
    else
        missing "C++ compiler ($CXX)" "$APT build-essential"; m=$((m + 1))
    fi
    for tool in scons m4 git; do
        if command -v "$tool" >/dev/null; then
            ok "$tool"
        else
            missing "$tool" "$APT $tool"; m=$((m + 1))
        fi
    done
    if echo '#include <zlib.h>' | "$CXX" -E -x c++ - >/dev/null 2>&1; then
        ok "zlib headers"
    else
        missing "zlib headers" "$APT zlib1g-dev"; m=$((m + 1))
    fi
    # Python for gem5: exactly what scons' "Checking Python version" does.
    if ! command -v python3-config >/dev/null; then
        missing "python3-config (Python development files)" \
            "$APT python3-dev"; m=$((m + 1))
    else
        t="$(mktemp -d)"
        cat > "$t/t.cc" <<'CC'
#include <Python.h>
int main() { Py_Initialize(); int ok = Py_IsInitialized(); Py_Finalize();
             return ok ? 0 : 1; }
CC
        flags="$(python3-config --ldflags --embed 2>/dev/null ||
                 python3-config --ldflags)"
        # shellcheck disable=SC2046,SC2086
        if ! "$CXX" $(python3-config --includes) "$t/t.cc" -o "$t/t" \
                $flags >"$t/log" 2>&1; then
            missing "Python headers/library for gem5 ($(command -v python3-config))" \
                "$APT python3-dev" \
                "or fix gem5_build_env in env.local.sh"; m=$((m + 1))
        elif ! "$t/t" >"$t/log" 2>&1; then
            missing "Python for gem5 compiles but does not run: $(head -1 "$t/log")" \
                "the libpython of $(command -v python3-config) is not on the loader path;" \
                "use the system Python (no gem5_build_env) or set LD_LIBRARY_PATH in gem5_build_env"
            m=$((m + 1))
        else
            ok "Python $(python3-config --includes | grep -o 'python3\.[0-9]*' | head -1 | sed 's/python//') for gem5 (headers and libpython work)"
        fi
        rm -rf "$t"
    fi
    if [ "$gem5_needs_build" = 1 ]; then
        need_gb=7
        dir="$GEM5_DIR"; [ -e "$dir" ] || dir="$(dirname "$GEM5_DIR")"
        while [ ! -e "$dir" ]; do dir="$(dirname "$dir")"; done
        free_gb="$(df -BG --output=avail "$dir" | tail -1 | tr -dc 0-9)"
        if [ "$free_gb" -ge "$need_gb" ]; then
            ok "disk: ${free_gb} GB free (gem5 needs about 6 GB)"
        else
            missing "disk: ${free_gb} GB free, gem5 needs about 6 GB" \
                "free space, or set GEM5_DIR to another disk"; m=$((m + 1))
        fi
        mem_gb="$(free -g | awk '/^Mem:/ {print $2}')"
        jobs="${BUILD_JOBS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"
        if [ "$mem_gb" -lt 8 ]; then
            warn "memory: ${mem_gb} GB for $jobs compile jobs; if the build is killed, use make gem5 BUILD_JOBS=4"
        else
            ok "memory: ${mem_gb} GB, $(nproc) threads"
        fi
    fi
    exit $m
)
MISSING=$((MISSING + $?))

# ----------------------------------------------------------------- gem5
section "gem5"
if [ -d "$GEM5_DIR/.git" ] || [ -f "$GEM5_DIR/SConstruct" ]; then
    have="$(git -C "$GEM5_DIR" describe --tags 2>/dev/null || echo unknown)"
    case "$have" in
        "$GEM5_VERSION"*) ok "checkout $GEM5_DIR ($have)" ;;
        *) warn "checkout $GEM5_DIR is $have; the results used $GEM5_VERSION" ;;
    esac
    for p in "$SYSTOLIC_ROOT"/gem5_model/patches/*.patch; do
        if git -C "$GEM5_DIR" apply --reverse --check "$p" 2>/dev/null; then
            ok "patch applied: $(basename "$p")"
        else
            todo "patch $(basename "$p") not applied yet [make gem5]"
        fi
    done
else
    todo "no checkout at $GEM5_DIR: make gem5 clones $GEM5_VERSION (about 20 MB download)"
fi
if [ "$gem5_needs_build" = 0 ]; then
    newer="$(find "$SYSTOLIC_ROOT/gem5_model/src" -newer "$GEM5_BIN" -type f | head -1)"
    if [ -n "$newer" ]; then
        todo "gem5.opt is older than ${newer#"$SYSTOLIC_ROOT"/} [make gem5]"
    else
        ok "gem5.opt built: $GEM5_BIN"
    fi
else
    todo "gem5.opt not built yet [make gem5, 20-30 min]"
fi

if [ "$ONLY_BUILD" = 1 ]; then
    [ "$MISSING" -gt 0 ] && { echo "$MISSING missing: fix the lines above."; exit 1; }
    exit 0
fi

# ------------------------------------------------------- host program
section "RISC-V host program"
if find_riscv_cc && ! command -v "${RISCV_CC_CMD[0]}" >/dev/null; then
    missing "RISC-V compiler $RISCV_CC_DESC: ${RISCV_CC_CMD[0]} not found" \
        "correct RISCV_CC in env.local.sh, or remove it and run make zig"
elif [ -n "${RISCV_CC_DESC:-}" ]; then
    case "$RISCV_CC_DESC" in
        zig\ 0.16.0*|RISCV_CC=*) ok "compiler: $RISCV_CC_DESC" ;;
        *) warn "compiler: $RISCV_CC_DESC; the results used zig 0.16.0 (make zig)" ;;
    esac
else
    missing "RISC-V Linux C compiler" "make zig   (no sudo)" \
        "or: $APT gcc-riscv64-linux-gnu"
fi
if [ -x "$HOST_BIN" ]; then ok "host program built"
else todo "host program not built yet [make host]"; fi

# --------------------------------------------------------------- Python
section "Python for workloads and analysis ($PYTHON)"
if ! command -v "$PYTHON" >/dev/null; then
    missing "$PYTHON not found" "make venv" "or set PYTHON in env.local.sh"
else
    out="$("$PYTHON" - "$SYSTOLIC_ROOT/requirements.txt" <<'PY'
import importlib, re, sys
mods = {"numpy": "numpy", "pyyaml": "yaml", "pandas": "pandas",
        "matplotlib": "matplotlib", "tabulate": "tabulate"}
for line in open(sys.argv[1]):
    m = re.match(r"\s*([A-Za-z0-9_-]+)==([^\s#]+)", line)
    if not m:
        continue
    pkg, want = m.group(1).lower(), m.group(2)
    try:
        mod = importlib.import_module(mods.get(pkg, pkg))
        have = getattr(mod, "__version__", "?")
        print(("ok" if have == want else "warn") + f" {pkg} {have} {want}")
    except Exception:
        print(f"missing {pkg} - {want}")
PY
)"
    absent=()
    while read -r st pkg have want; do
        case "$st" in
            ok) ok "$pkg $have" ;;
            warn) warn "$pkg $have (results used $want: numbers agree to round-off)" ;;
            missing) absent+=("$pkg") ;;
        esac
    done <<< "$out"
    if [ "${#absent[@]}" -gt 0 ]; then
        missing "Python packages: ${absent[*]}" "make venv" \
            "or: $PYTHON -m pip install -r requirements.txt"
    fi
fi

# ------------------------------------------------------------- outputs
section "Generated inputs"
if [ -f "$WORKLOAD_ROOT/out/workloads.json" ]; then ok "workloads generated"
else todo "workloads not generated yet [make workloads]"; fi
n_runs="$(find "$BUILD_DIR/runs" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)"
if [ "$n_runs" -gt 0 ]; then ok "$n_runs finished or started runs cached in build/runs"
else todo "no simulations run yet [make data, about 1 h]"; fi

echo
if [ "$MISSING" -gt 0 ]; then
    echo "$MISSING missing: fix the lines marked MISSING, then run make deps again."
    exit 1
fi
echo "All dependencies present. 'todo' lines are built by make all."
