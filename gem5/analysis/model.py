#!/usr/bin/env python3
#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : model.py                                                      #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : Analytical model of the proposal, in code                     #
#-----------------------------------------------------------------------------#
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
#
# Options of the model that change the demand on the banks:
#   weight_store  bank crossings of a weight tile: "spm" 2 (D_fill),
#                 "fill_bypass" 1 (eq. 1), "separate" 0 (eq. 1 minus N^2/T).
#   bank_ports    "1r1w" gives each bank one read and one write port, so the
#                 busier direction sets the bank count, not the sum:
#                   BANKS = max(ceil(D_read / 8), ceil(D_write / 8))
#   psum_rmw      accumulating across K tiles reads the old partial sum back:
#                 4 N M more read bytes per tile period, on all but a row's
#                 first K tile (rmw_fraction).
#   psum_store    "external" takes the 4 N M partial-sum bytes off the banks.
#   stall model   "lockstep": no queues between banks and array, a step takes
#                 as many cycles as its busiest bank has words.  With
#                 words = ceil(N/8) + N/2 spread as evenly as possible,
#                   util = 1 / ceil(words / BANKS)
#                 and lockstep_fixed_address() gives the value for gem5's
#                 fixed per-stream addresses, which spread them less evenly.

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
# DDR5-8400 on a 64-byte memory bus (--membus-width 64), where the DRAM
# rather than the bus sets the limit.  Measured the same way.
DRAM_EFFICIENCY_WIDE_BUS = {"DDR5_8400_4x8": 0.91}
DDR4_2400_PEAK_BPS = 19.2e9
DEFAULT_CLOCK_HZ = 2e9
DDR4_2400_BYTES_PER_CYCLE = (DRAM_EFFICIENCY * DDR4_2400_PEAK_BPS /
                             DEFAULT_CLOCK_HZ)


def dram_bytes_per_cycle(clock_hz=DEFAULT_CLOCK_HZ,
                         peak_bps=DDR4_2400_PEAK_BPS,
                         mem_type="DDR4_2400_8x8", membus_width=0):
    """Sustained DRAM bandwidth in bytes per accelerator cycle."""
    eff = DRAM_EFFICIENCY_BY_TYPE.get(mem_type, DRAM_EFFICIENCY)
    if membus_width >= 64:
        eff = DRAM_EFFICIENCY_WIDE_BUS.get(mem_type, eff)
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


def lockstep_words(N):
    """Activation + partial-sum words of a steady-state lockstep step."""
    return ceil_div(N, 8) + N // 2


def lockstep_staircase(N, banks, bank_ports="1rw", psum_rmw=False,
                       psum_store="spm"):
    """Lockstep steady-state utilization with the words spread evenly.  With
    1r1w ports reads and writes do not meet, so the busier port sets it."""
    act, lanes = ceil_div(N, 8), N // 2
    reads = act + (lanes if psum_rmw else 0)
    writes = 0 if psum_store == "external" else lanes
    if bank_ports == "1r1w":
        return 1 / max(ceil_div(reads, banks), ceil_div(writes, banks))
    return 1 / ceil_div(reads + writes, banks)


def lockstep_peak_words(N, bank_ports="1rw", psum_rmw=False,
                        psum_store="spm"):
    """Words the busiest port must take in the worst step of back-to-back
    tiles, when the next tile's weights load alongside everything else."""
    act, lanes = ceil_div(N, 8), N // 2
    reads = 2 * act + (lanes if psum_rmw else 0)
    writes = 0 if psum_store == "external" else lanes
    return max(reads, writes) if bank_ports == "1r1w" else reads + writes


def lockstep_fixed_address(N, banks, M=784, K=1152, NOUT=128, slots=None,
                           bank_ports="1rw", psum_rmw=False, psum_store="spm"):
    """Lockstep steady-state utilization for gem5's fixed addresses: the
    average of 1 / (words on the busiest bank) over a tile's rows.  In step
    m the activation words are consecutive and partial-sum lane j writes
    word c0 + (m - N - 2j - 1) * N/2 + j, so lanes sit j * (N - 1) words
    apart.  The weight tile's reads (N of T steps) are left out."""
    nkb, nnb = ceil_div(K, N), ceil_div(NOUT, N)
    if slots is None:
        slots = max(2, 16384 // (N * N))
    align64 = lambda x: (x + 63) // 64 * 64  # noqa: E731
    c0 = align64(align64(nkb * M * N) + slots * N * N) // 8
    total, steps = 0, 0
    for kb in range(min(nkb, 8)):
        for nb in range(min(nnb, 4)):
            for m in range(2 * N, min(M, 2 * N + 2 * banks + 1)):
                load = [0] * banks
                # With 1r1w the partial sums use the other port.
                wload = [0] * banks if bank_ports == "1r1w" else load
                for j in range(ceil_div(N, 8)):
                    load[((kb * M + m) * N // 8 + j) % banks] += 1
                for j in range(N // 2):
                    w = c0 + (nb * M + m - N - 2 * j - 1) * (N // 2) + j
                    if psum_store != "external":
                        wload[w % banks] += 1
                    if psum_rmw:
                        load[w % banks] += 1
                total += max(max(load), max(wload))
                steps += 1
    return steps / total if total else float("nan")


def rmw_fraction(N, K):
    """The share of K tiles that have an old partial sum to read back."""
    nkb = ceil_div(K, N)
    return (nkb - 1) / nkb


def port_demand(N, M, weight_store="spm", ideal_dram=False,
                double_buffer=True, rmw_frac=0.0, psum_store="spm"):
    """(D_read, D_write) in bytes per cycle, for 1r1w ports.  Reads N M
    activations plus the array's weight read, plus any partial sums read
    back; writes 4 N M partial sums plus the DMA fill, per tile period."""
    T = tile_period(N, M, double_buffer)
    arr_w = 0 if weight_store == "separate" else N * N
    dma_w = 0 if ideal_dram or weight_store != "spm" else N * N
    psum = 0 if psum_store == "external" else 4 * N * M
    return ((N * M + arr_w + rmw_frac * 4 * N * M) / T, (psum + dma_w) / T)


def banks_needed_1r1w(N, M, weight_store="spm"):
    """The bank rule with separate read and write ports: the busier port.
    weight_store="spm" counts the DMA fill on the write side."""
    T = max(M, N)
    dma_w = N * N if weight_store == "spm" else 0
    arr_w = 0 if weight_store == "separate" else N * N
    return max(ceil_div(N * M + arr_w, 8 * T),
               ceil_div(4 * N * M + dma_w, 8 * T))


def predict(N, banks, M, K, NOUT, double_buffer=True,
            dram_bytes_per_cycle=DDR4_2400_BYTES_PER_CYCLE,
            ideal_spm=False, ideal_dram=False, bank_ports="1rw",
            weight_store="spm", psum_rmw=False, psum_store="spm"):
    """Returns a dict of predicted quantities for one run."""
    tiles = ceil_div(K, N) * ceil_div(NOUT, N)
    T = tile_period(N, M, double_buffer)
    D = demand(N, M)
    if not double_buffer:
        # Same bytes per tile but spread over M + N cycles.
        D = (5 * N * M + N * N) / T
    # Weights also enter the scratchpad through the DMA: N^2 more per tile.
    D_fill = D if ideal_dram else D + N * N / T
    if weight_store == "fill_bypass":
        D_fill = D
    elif weight_store == "separate":
        D_fill = D - N * N / T
    rmw_frac = rmw_fraction(N, K) if psum_rmw else 0.0
    if psum_rmw:
        D_fill = D_fill + rmw_frac * 4 * N * M / T
    if psum_store == "external":
        D_fill = D_fill - 4 * N * M / T
    B = supply(banks)
    spm_ceiling = 1.0 if ideal_spm else min(1.0, B / D_fill)
    spm_ceiling_eq1 = 1.0 if ideal_spm else min(1.0, B / D)
    if bank_ports == "1r1w":
        D_rw = max(port_demand(N, M, weight_store, ideal_dram,
                               double_buffer, rmw_frac, psum_store))
        spm_ceiling = 1.0 if ideal_spm else min(1.0, B / D_rw)
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
