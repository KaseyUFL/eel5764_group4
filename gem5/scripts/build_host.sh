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

zig_cc() {   # zig_cc VENV: the zig C compiler from a venv, if it has one
    [ -x "$1/bin/python" ] && "$1/bin/python" -m ziglang version \
        >/dev/null 2>&1
}
if [ -n "${RISCV_CC:-}" ]; then
    read -r -a CC <<< "$RISCV_CC"
elif command -v riscv64-linux-gnu-gcc >/dev/null; then
    CC=(riscv64-linux-gnu-gcc -march=rv64gc
        -fno-tree-loop-distribute-patterns)
elif zig_cc "$SYSTOLIC_ROOT/third_party/zig-venv"; then
    CC=("$SYSTOLIC_ROOT/third_party/zig-venv/bin/python" -m ziglang cc
        -target riscv64-linux-musl)
elif zig_cc "$HOME/riscv-tools/venv"; then
    CC=("$HOME/riscv-tools/venv/bin/python" -m ziglang cc
        -target riscv64-linux-musl)
else
    echo "No RISC-V compiler found. Either:" >&2
    echo "  make zig     (pip-installs zig into third_party/zig-venv)" >&2
    echo "  sudo apt install gcc-riscv64-linux-gnu" >&2
    exit 1
fi
mkdir -p "$(dirname "$OUT")"
"${CC[@]}" "${FLAGS[@]}" -o "$OUT" "$SRC"
echo "built $OUT ($(file -b "$OUT" | cut -d, -f1-2))"
