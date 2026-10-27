#!/usr/bin/env bash
# Fetch, patch and build gem5 with the SystolicAccel SimObject compiled in.
#
#   bash scripts/build_gem5.sh            # what `make gem5` runs
#   BUILD_JOBS=8 bash scripts/build_gem5.sh   # fewer jobs (less RAM)
#
# 1. If $GEM5_DIR does not exist, clone gem5 $GEM5_VERSION into it
#    (default: third_party/gem5).
# 2. Apply gem5_model/patches/*.patch, unless already applied.
# 3. Build $GEM5_DIR/build/RISCV_SYSTOLIC/gem5.opt with
#    EXTRAS=gem5_model/src (gem5's mechanism for out-of-tree SimObjects).
#
# The build goes to its own directory, build/RISCV_SYSTOLIC, so a stock
# build/RISCV in the same checkout is untouched.  The first build takes
# about 30 min on 16+ cores; after that, editing systolic_accel.cc only
# recompiles that file and relinks (1-2 min).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ ! -e "$GEM5_DIR" ]; then
    echo "cloning gem5 $GEM5_VERSION from $GEM5_REPO into $GEM5_DIR"
    mkdir -p "$(dirname "$GEM5_DIR")"
    git -c advice.detachedHead=false clone --depth 1 \
        --branch "$GEM5_VERSION" "$GEM5_REPO" "$GEM5_DIR"
fi
if [ ! -f "$GEM5_DIR/SConstruct" ]; then
    echo "$GEM5_DIR is not a gem5 checkout" >&2
    exit 1
fi
have="$(git -C "$GEM5_DIR" describe --tags 2>/dev/null || echo unknown)"
case "$have" in
    "$GEM5_VERSION"*) ;;
    *) echo "warning: $GEM5_DIR is at $have; the results in results/" \
            "were produced with gem5 $GEM5_VERSION" >&2 ;;
esac

for p in "$SYSTOLIC_ROOT"/gem5_model/patches/*.patch; do
    if git -C "$GEM5_DIR" apply --reverse --check "$p" 2>/dev/null; then
        echo "patch already applied: $(basename "$p")"
    else
        git -C "$GEM5_DIR" apply "$p"
        echo "applied patch: $(basename "$p")"
    fi
done

JOBS="${BUILD_JOBS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"
if declare -F gem5_build_env >/dev/null; then
    gem5_build_env           # from env.local.sh
fi
cd "$GEM5_DIR"
if [ ! -f build/RISCV_SYSTOLIC/gem5.build/config ]; then
    scons --ignore-style defconfig build/RISCV_SYSTOLIC build_opts/RISCV
fi
scons --ignore-style build/RISCV_SYSTOLIC/gem5.opt \
    EXTRAS="$SYSTOLIC_ROOT/gem5_model/src" -j"$JOBS"
echo "built $GEM5_BIN"
