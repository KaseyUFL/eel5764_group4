#!/usr/bin/env bash
# Smoke test: every workload once at one (N, BANKS) point, plus the
# ideal-scratchpad baseline for each, then a CSV of the results.
#
#   bash scripts/run_all_workloads.sh [N] [BANKS] [--rtl]
#   make smoke          (= the --rtl form at N=8, BANKS=4: about 10 s)
#
# Defaults: N=16, BANKS=8 (B = 64 bytes/cycle, the proposal's reference).
# With --rtl the reduced *_rtl workloads are used (seconds per run).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

N="${1:-16}"; B="${2:-8}"; SFX=""
[ "${3:-}" = "--rtl" ] && SFX="_rtl"
BASE="$BUILD_DIR/m5out/all_N${N}_B${B}${SFX}"
DIRS=()
for wl in resnet50_conv3x bert_ffn1 bert_attn_qk llama7b_gemv; do
    for mode in banked ideal; do
        out="$BASE/${wl}${SFX}_${mode}"
        opts=()
        [ "$mode" = ideal ] && opts=(--ideal-spm)
        echo "=== ${wl}${SFX}  N=$N  BANKS=$B  ($mode) ==="
        OUT="$out" bash "$SYSTOLIC_ROOT/scripts/run_workload.sh" \
            "${wl}${SFX}" "$N" "$B" ${opts[@]+"${opts[@]}"} 2>&1 \
            | grep -E "SystolicAccel done|peUtilization|verify" || true
        DIRS+=("$out")
    done
done
"$PYTHON" "$SYSTOLIC_ROOT/analysis/parse_stats.py" "${DIRS[@]}" \
    -o "$BASE/results.csv"

# Every run must have finished and verified bit-exactly.
"$PYTHON" - "$BASE/results.csv" "${#DIRS[@]}" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
print(f"\n{'workload':22s} {'scratchpad':10s} {'pe_util':>8s} "
      f"{'model':>6s}  verify")
bad = 0
for r in rows:
    ok = float(r["verify_mismatches"]) == 0
    bad += not ok
    spm = "ideal" if r["ideal_spm"] in ("1", "True") else "banked"
    print(f"{r['workload']:22s} {spm:10s} {float(r['pe_util']):8.3f} "
          f"{float(r['pred_util']):6.3f}  "
          f"{'ok' if ok else r['verify_mismatches'] + ' mismatches'}")
missing = int(sys.argv[2]) - len(rows)
if bad or missing:
    sys.exit(f"FAILED: {bad} run(s) did not verify, {missing} missing")
print(f"all {len(rows)} runs verified")
PY
