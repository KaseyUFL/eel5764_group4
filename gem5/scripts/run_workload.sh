#!/usr/bin/env bash
# Run ONE workload on the SystolicAccel model and print the key stats.
#
#   bash scripts/run_workload.sh WORKLOAD N BANKS [options]
#   make run WL=WORKLOAD N=N BANKS=BANKS [ARGS='options']
#
#   WORKLOAD  resnet50_conv3x | bert_ffn1 | bert_attn_qk | llama7b_gemv
#             (or the reduced *_rtl versions, e.g. bert_attn_qk_rtl)
#   N         array dimension: 8, 16, 24, 32 (any multiple of 8)
#   BANKS     scratchpad banks: any count >= 1 (e.g. 4, 6, 8, 11, 16, 21)
#   options   passed straight to systolic_se.py, e.g.
#             --no-double-buffer  --ideal-spm  --mode sw  --standalone
#             --arbitration lane  --psum-queue 4
#             --stall-model lockstep  --bank-ports 1r1w  --bank-hash xor
#             (all options: $GEM5_BIN gem5_model/configs/systolic_se.py -h)
#
# Environment:
#   OUT=dir        output directory (default build/m5out/WL_N<N>_B<B>)
#   DEBUG=flags    gem5 debug flags, e.g. DEBUG=SystolicAccel or
#                  DEBUG=SystolicAccel,SystolicAccelDma,NoncoherentXBar
#                  (trace goes to OUT/trace.out; use a small workload!)
#
# Example:
#   bash scripts/run_workload.sh bert_attn_qk 16 8
#   DEBUG=SystolicAccel bash scripts/run_workload.sh \
#       bert_attn_qk_rtl 8 4 --max-cycles 300
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ $# -lt 3 ]; then
    sed -n '2,26p' "$0"; exit 1
fi
if [ ! -x "$GEM5_BIN" ]; then
    echo "gem5 binary not found: $GEM5_BIN (run: make gem5)" >&2; exit 1
fi
META="$WORKLOAD_ROOT/out/gem5/$1/meta.json"
if [ ! -f "$META" ]; then
    echo "no workload '$1' at $META (run: make workloads)" >&2; exit 1
fi
WL="$1"; N="$2"; B="$3"; shift 3
OUT="${OUT:-$BUILD_DIR/m5out/${WL}_N${N}_B${B}}"
mkdir -p "$OUT"

DBG=()
if [ -n "${DEBUG:-}" ]; then
    DBG=(--debug-flags="$DEBUG" --debug-file=trace.out)
fi

cat > "$OUT/run.json" <<EOF
{"workload": "$WL", "N": $N, "banks": $B,
 "double_buffer": $([[ " $* " == *" --no-double-buffer "* ]] && echo false || echo true),
 "ideal_spm": $([[ " $* " == *" --ideal-spm "* ]] && echo true || echo false),
 "ideal_dram": $([[ " $* " == *" --ideal-dram "* ]] && echo true || echo false),
 "arbitration": "$([[ " $* " == *" --arbitration lane "* ]] && echo lane || echo bank)",
 "M": $("$PYTHON" -c "import json;print(json.load(open('$META'))['M'])"),
 "K": $("$PYTHON" -c "import json;print(json.load(open('$META'))['K'])"),
 "N_out": $("$PYTHON" -c "import json;print(json.load(open('$META'))['N_out'])"),
 "extra": "$*"}
EOF

"$GEM5_BIN" -d "$OUT" ${DBG[@]+"${DBG[@]}"} \
    "$SYSTOLIC_ROOT/gem5_model/configs/systolic_se.py" \
    --workload "$WL" --workload-dir "$WORKLOAD_ROOT/out/gem5" \
    --host-bin "$HOST_BIN" --array-dim "$N" --banks "$B" "$@"

echo
echo "== Table II metrics for the layer ($OUT/stats.txt) =="
"$PYTHON" "$SYSTOLIC_ROOT/analysis/parse_stats.py" "$OUT" | "$PYTHON" -c '
import csv, sys
r = next(csv.DictReader(sys.stdin))
keys = ["mode", "pe_util", "pred_util", "layer_util", "total_cycles",
        "layer_cycles", "load_cycles", "store_cycles", "spm_stall_cycles",
        "dma_stall_cycles", "spm_bw_total", "bank_imbalance",
        "avg_spm_latency", "layer_time_s", "host_cpi", "host_ipc",
        "dram_bus_util", "dram_energy_J", "edp_dram_Js",
        "verify_mismatches"]
for k in keys:
    print(f"  {k:22s} {r[k]}")
'
