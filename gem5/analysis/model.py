#!/usr/bin/env python3
# EEL5764 Group 4 -- the analytical model of the proposal, in code.
#
# Used by parse_stats.py to put a prediction next to every gem5 result, and
# runnable on its own to print Table I and a predicted utilization grid:
#
#   python3 analysis/model.py
#   python3 analysis/model.py --workload bert_ffn1
#
# Equations (proposal Sec. II.A):
#   D(N) = (5 N M + N^2) / max(M, N)   bytes/cycle        (1)
#   B    = 8 * BANKS                   bytes/cycle        (2)
#   util ~ (M / T) * min(1, B / D)
# One correction to eq. (1), found in the gem5 runs: a weight tile is
# written INTO the scratchpad by the DMA and then read OUT of it by the
# array, so its N^2 bytes cross the banks twice:
#   D_fill(N) = (5 N M + 2 N^2) / max(M, N)
# The extra N^2 / T is negligible for M >> N (1% at M = 784, N = 32) and
# large at low reuse (M = 1: 2N + 5 instead of N + 5).  predict() uses
# D_fill unless the weights do not come through the DMA (ideal_dram), and
# also returns the verbatim eq. (1) prediction as pred_util_eq1.
# plus a DRAM ceiling the proposal states in words for M = 1: every tile's
# N^2 weight bytes must arrive from DRAM within one tile period T, so
#   util <= (M / T) * min(1, B_dram / (N^2 / T)),
# where B_dram is the DRAM bandwidth in bytes per ACCELERATOR cycle:
#   B_dram = DRAM_EFFICIENCY * peak_bytes_per_second / accelerator_clock.

import argparse
import json
import math

WORKLOADS = {  # name: (M, K, N_out); must match workloads/
    "resnet50_conv3x": (784, 1152, 128),
    "bert_ffn1": (128, 768, 3072),
    "bert_attn_qk": (128, 64, 128),
    "llama7b_gemv": (1, 4096, 4096),
}

# DDR4-2400 x64: 2400 MT/s * 8 B = 19.2 GB/s peak.  gem5's single-channel
# DDR4_2400_8x8 sustains 13.5 GB/s under the weight DMA (70% of peak:
# refresh, row misses, bus turnaround).  That one number, measured on
# llama7b_gemv at N = 32 and 2 GHz, is the only calibrated constant in the
# model for DDR4; other array clocks and array sizes are predictions.
DRAM_EFFICIENCY = 0.70
# Sustained / peak bandwidth of the other DRAM models, measured the same
# way (llama7b_gemv, N = 32, 2 GHz, one run each).  DDR5-8400 is held back
# by the memory bus (about 24 GB/s), not by the DRAM itself.
DRAM_EFFICIENCY_BY_TYPE = {
    "DDR3_1600_8x8": 0.78,
    "DDR4_2400_8x8": 0.70,
    "DDR5_6400_4x8": 0.91,
    "DDR5_8400_4x8": 0.71,
}
DDR4_2400_PEAK_BPS = 19.2e9
DEFAULT_CLOCK_HZ = 2e9
DDR4_2400_BYTES_PER_CYCLE = (DRAM_EFFICIENCY * DDR4_2400_PEAK_BPS /
                             DEFAULT_CLOCK_HZ)


def dram_bytes_per_cycle(clock_hz=DEFAULT_CLOCK_HZ,
                         peak_bps=DDR4_2400_PEAK_BPS,
                         mem_type="DDR4_2400_8x8"):
    """Sustained DRAM bandwidth in bytes per accelerator cycle."""
    eff = DRAM_EFFICIENCY_BY_TYPE.get(mem_type, DRAM_EFFICIENCY)
    return eff * peak_bps / clock_hz


def demand_with_fill(N, M):
    """Eq. (1) plus the DMA writing each weight tile into the scratchpad."""
    return (5 * N * M + 2 * N * N) / max(M, N)


def ceil_div(a, b):
    return -(-a // b)


def demand(N, M):
    """Eq. (1), bytes per cycle."""
    return (5 * N * M + N * N) / max(M, N)


def supply(banks):
    """Eq. (2), bytes per cycle."""
    return 8 * banks


def tile_period(N, M, double_buffer=True):
    return max(M, N) if double_buffer else M + N


def schedule_cycles(N, M, K, NOUT, double_buffer=True):
    """Stall-free cycle count of the SystolicAccel schedule (and of the
    RTL testbench schedule): weight prologue, nTiles tile periods, then the
    2N - 1 cycle drain of the last row."""
    tiles = ceil_div(K, N) * ceil_div(NOUT, N)
    T = tile_period(N, M, double_buffer)
    s_last = tiles * T if double_buffer else (tiles - 1) * T + N + 1
    return s_last + M - 1 + 2 * N


def predict(N, banks, M, K, NOUT, double_buffer=True,
            dram_bytes_per_cycle=DDR4_2400_BYTES_PER_CYCLE,
            ideal_spm=False, ideal_dram=False):
    """Returns a dict of predicted quantities for one run."""
    tiles = ceil_div(K, N) * ceil_div(NOUT, N)
    T = tile_period(N, M, double_buffer)
    D = demand(N, M)
    if not double_buffer:
        # Same bytes per tile but spread over M + N cycles.
        D = (5 * N * M + N * N) / T
    # Weights also enter the scratchpad through the DMA: N^2 more per tile.
    D_fill = D if ideal_dram else D + N * N / T
    B = supply(banks)
    spm_ceiling = 1.0 if ideal_spm else min(1.0, B / D_fill)
    spm_ceiling_eq1 = 1.0 if ideal_spm else min(1.0, B / D)
    dram_need = N * N / T
    dram_ceiling = 1.0 if ideal_dram else \
        min(1.0, dram_bytes_per_cycle / dram_need)
    reuse = M / T
    sched = schedule_cycles(N, M, K, NOUT, double_buffer)
    ideal_util = tiles * M / sched
    return {
        "tiles": tiles,
        "T": T,
        "D": D,
        "D_fill": D_fill,
        "B": B,
        "reuse_ceiling": reuse,
        "spm_ceiling": spm_ceiling,
        "dram_ceiling": dram_ceiling,
        "dram_need": dram_need,
        "dram_supply": dram_bytes_per_cycle,
        "sched_cycles": sched,
        "ideal_util": ideal_util,
        "pred_util": ideal_util * min(spm_ceiling, dram_ceiling),
        "pred_util_eq1": ideal_util * min(spm_ceiling_eq1, dram_ceiling),
        "banks_needed": math.ceil(D / 8),  # bank-matching rule
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=sorted(WORKLOADS), nargs="*")
    ap.add_argument("--N", type=int, nargs="*", default=[8, 16, 24, 32])
    ap.add_argument("--banks", type=int, nargs="*", default=[4, 8, 16, 32])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    print("Table I (large M):  N  PEs  edge D  fill/drain  skew regs")
    for n in args.N:
        print(f"                   {n:2d} {n * n:5d} {5 * n:6d} "
              f"{2 * n - 2:8d} {n * (n - 1) // 2:10d}")
    print()

    out = {}
    for wl in args.workload or sorted(WORKLOADS):
        M, K, NOUT = WORKLOADS[wl]
        print(f"{wl}: M={M} K={K} N_out={NOUT}   predicted PE utilization")
        print("   N  D(N)  BANKS=" + "".join(f"{b:>8d}" for b in args.banks)
              + "   ceil(D/8)")
        for n in args.N:
            row = [predict(n, b, M, K, NOUT) for b in args.banks]
            out.update({f"{wl}/N{n}/B{b}": r for b, r in zip(args.banks, row)})
            print(f"  {n:2d} {row[0]['D']:6.1f}        "
                  + "".join(f"{r['pred_util']:8.3f}" for r in row)
                  + f"   {row[0]['banks_needed']:5d}")
        print()
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
