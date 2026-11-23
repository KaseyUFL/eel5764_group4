#!/usr/bin/env bash
# Build the RISC-V host program (static, runs in gem5 SE mode).
#
#   bash scripts/build_host.sh            # what `make host` runs
#
# Compiler, first match wins:
#   $RISCV_CC                         e.g. RISCV_CC=riscv64-linux-gnu-gcc
#   riscv64-linux-gnu-gcc             (sudo apt install gcc-riscv64-linux-gnu)
#   zig cc from third_party/zig-venv  (make zig; no sudo needed)
#   zig cc from ~/riscv-tools/venv    (pip install ziglang)
#
# The results in results/ were produced with zig 0.16.0 (make zig pins it).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SRC="$SYSTOLIC_ROOT/gem5_model/host/systolic_host.c"
OUT="$HOST_BIN"
# -fno-builtin-memcpy/-memset: keep the word-copy loops in the weight
# packing as loops (see copy_row() in systolic_host.c).
# -s: no symbols or debug info.  Those hold the compiler's and this
# directory's paths, so without -s the binary differs between clones (its
# loaded code and data do not).  Drop -s for symbol names in Exec traces.
FLAGS=(-O2 -static -s -Wall -Wextra -fno-builtin-memcpy -fno-builtin-memset)

if ! find_riscv_cc; then     # env.sh
    echo "No RISC-V compiler found. Either:" >&2
    echo "  make zig     (pip-installs zig into third_party/zig-venv)" >&2
    echo "  sudo apt install gcc-riscv64-linux-gnu" >&2
    exit 1
fi
mkdir -p "$(dirname "$OUT")"
"${RISCV_CC_CMD[@]}" "${FLAGS[@]}" -o "$OUT" "$SRC"
echo "built $OUT ($(file -b "$OUT" | cut -d, -f1-2))"
# The variant with --split-host-stats and --prepacked-weights support.
"${RISCV_CC_CMD[@]}" "${FLAGS[@]}" -DSYSTOLIC_HOST_EXT -o "${OUT}_ext" "$SRC"
echo "built ${OUT}_ext"
