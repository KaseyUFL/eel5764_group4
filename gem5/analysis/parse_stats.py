#!/usr/bin/env python3
# EEL5764 Group 4 -- turn gem5 run directories into one CSV.
#
#   python3 analysis/parse_stats.py RUN_DIR [RUN_DIR ...] \
#       -o results.csv
#
# Each RUN_DIR is a gem5 -d output directory holding stats.txt and the
# systolic_run.json that systolic_se.py writes.  In host-driven runs the
# RISC-V program dumps statistics around the layer, so stats.txt has three
# dumps (setup / layer / verify) and the layer is dump #1; standalone runs
# have one dump.  Everything below is read from the layer dump.
#
# Proposal Table II metric          column(s)            gem5 statistic
# --------------------------------  -------------------  -----------------
# PE utilization                    pe_util              system.accel.peUtilization
# Stall cycles                      spm_stall_cycles     system.accel.spmStallCycles
#                                   (+ read/write/dma)
# Delivered bandwidth               spm_bw_read/write/   system.spm*.bwRead::total,
#                                   total (bytes/cycle)  bwWrite::total
# Bank balance                      bank_imbalance,      system.spm_xbar.pktCount_*
#                                   bank_cov
# Scratchpad AMAT                   avg_spm_latency      system.accel.avgSpmLatency
# Layer time / speedup              layer_time_s         simSeconds (layer dump)
#                                   (speedup: analysis)
# Host CPI / IPC                    host_cpi, host_ipc   system.cpu.cpi, system.cpu.ipc
# DRAM pressure                     dram_bus_util        system.mem_ctrl.dram.busUtil
# EDP                               dram_energy_J,       system.mem_ctrl.dram.rank*.
#                                   edp_dram_Js          totalEnergy x simSeconds
#                                   (+ array energy from Vivado, later)
# Hardware cost                     -- Vivado (RTL) --

import argparse
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model  # noqa: E402

STAT_RE = re.compile(r"^(\S+)\s+(-?[0-9.eE+-]+|nan|inf)\b")


def read_dumps(path):
    """Returns a list of {name: float}, one per statistics dump."""
    dumps, cur = [], None
    with open(path) as f:
        for line in f:
            if line.startswith("---------- Begin"):
                cur = {}
            elif line.startswith("---------- End"):
                dumps.append(cur)
                cur = None
            elif cur is not None:
                m = STAT_RE.match(line)
                if m:
                    try:
                        cur[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
    return dumps


def run_params(d):
    """run.json (from the scripts), overridden by systolic_run.json (what
    the config actually built)."""
    p = {}
    for name in ("run.json", "systolic_run.json"):
        path = os.path.join(d, name)
        if os.path.exists(path):
            with open(path) as f:
                p.update(json.load(f))
    if not p:
        raise RuntimeError("no systolic_run.json / run.json")
    return p


def repo_path(d):
    """d relative to the repository root when it is inside it, so that
    the CSVs do not depend on where the repository was cloned."""
    a = os.path.abspath(d)
    r = os.path.relpath(a, ROOT)
    return a if r == ".." or r.startswith(".." + os.sep) else r


def summarize(d):
    p = run_params(d)
    dumps = read_dumps(os.path.join(d, "stats.txt"))
    if not dumps:
        raise RuntimeError("stats.txt has no complete dump")
    idx = int(p.get("layer_dump", 0))
    if idx >= len(dumps):
        raise RuntimeError(f"expected layer dump #{idx}, found "
                           f"{len(dumps)} dumps (did the host finish?)")
    s = dumps[idx]
    g = lambda k, dflt=float("nan"): s.get(k, dflt)  # noqa: E731
    a = "system.accel."
    clock = float(p.get("clock_hz", 2e9))
    host = bool(p.get("host_driven", False))
    sw = p.get("mode") == "sw"

    # Scratchpad banks: system.spm (1), spm0..9 or spm00..63.
    banks = sorted({m.group(1) for k in s
                    for m in [re.match(r"(system\.spm\d*)\.bwRead::total$",
                                       k)] if m})
    bw_rd = sum(g(b + ".bwRead::total", 0) for b in banks) / clock
    bw_wr = sum(g(b + ".bwWrite::total", 0) for b in banks) / clock
    # Bank balance from the crossbar's packet counts (requests + responses)
    per_bank = {}
    for k, v in s.items():
        m = re.match(r"system\.spm_xbar\.pktCount_.*::(system\.spm\d*)\.port$",
                     k)
        if m:
            per_bank[m.group(1)] = per_bank.get(m.group(1), 0) + v
    vals = list(per_bank.values())
    if vals and sum(vals):
        mean = sum(vals) / len(vals)
        imbalance = max(vals) / mean
        cov = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5 / mean
    else:
        imbalance = cov = float("nan")
    dram_energy_pj = sum(v for k, v in s.items()
                         if re.match(r"system\.mem_ctrl\.dram\.rank\d+\."
                                     r"totalEnergy$", k))
    # simTicks is exact; simSeconds is printed with only 6 decimals.
    layer_s = g("simTicks") / 1e12

    N, B = int(p["N"]), int(p["banks"])
    M, K, NOUT = int(p["M"]), int(p["K"]), int(p["N_out"])
    db = bool(p.get("double_buffer", True))
    peak = float(p.get("dram_peak_bps", model.DDR4_2400_PEAK_BPS))
    pr = model.predict(
        N, B, M, K, NOUT, db,
        dram_bytes_per_cycle=model.dram_bytes_per_cycle(
            clock, peak, p.get("mem_type", "DDR4_2400_8x8")),
        ideal_spm=bool(p.get("ideal_spm", False)),
        ideal_dram=bool(p.get("ideal_dram", False)))
    row = {
        "workload": p["workload"],
        "mode": p.get("mode", "standalone"),
        "N": N, "banks": B, "B_bytes_per_cycle": 8 * B,
        "bank_mapping": p.get("bank_mapping", "interleave"),
        "double_buffer": int(db),
        "arbitration": p.get("arbitration", "bank"),
        "bank_policy": p.get("bank_policy", "fifo"),
        "psum_queue": p.get("psum_queue", ""),
        "ideal_spm": int(bool(p.get("ideal_spm", False))),
        "ideal_dram": int(bool(p.get("ideal_dram", False))),
        "clock_hz": clock,
        "mem_type": p.get("mem_type", "DDR4_2400_8x8"),
        "dram_peak_GBs": peak / 1e9,
        "lookahead": p.get("lookahead", ""),
        "spm_latency": p.get("spm_latency", ""),
        "dma_outstanding": p.get("dma_outstanding", ""),
        "weight_buffer_bytes": p.get("weight_buffer_bytes", ""),
        "M": M, "K": K, "N_out": NOUT, "tiles": pr["tiles"],
        # --- utilization and cycles (compute phase)
        "pe_util": g(a + "peUtilization"),
        "pred_util": pr["pred_util"],
        "pred_util_eq1": pr["pred_util_eq1"],
        "pred_spm_ceiling": pr["spm_ceiling"],
        "pred_dram_ceiling": pr["dram_ceiling"],
        "ideal_util": g(a + "idealUtilization"),
        "mac_util": g(a + "macUtilization"),
        "total_cycles": g(a + "totalCycles"),
        "schedule_cycles": g(a + "scheduleCycles"),
        "spm_stall_cycles": g(a + "spmStallCycles"),
        "spm_read_stall_cycles": g(a + "spmReadStallCycles"),
        "spm_write_stall_cycles": g(a + "spmWriteStallCycles"),
        "dma_stall_cycles": g(a + "dmaStallCycles"),
        # --- whole layer (host-driven: A load + compute + C store)
        "layer_cycles": g(a + "layerCycles"),
        "load_cycles": g(a + "loadCycles"),
        "store_cycles": g(a + "storeCycles"),
        "layer_util": g(a + "layerUtilization"),
        "layer_time_s": layer_s,
        # --- bandwidth
        "model_D": g(a + "modelDemand"),
        "model_D_fill": pr["D_fill"],
        "measured_D": g(a + "demandBytesPerCycle"),
        "delivered_array_bw": g(a + "arrayBytesPerCycle"),
        "spm_bw_read": bw_rd,
        "spm_bw_write": bw_wr,
        "spm_bw_total": bw_rd + bw_wr,
        "bank_imbalance": imbalance,
        "bank_cov": cov,
        "avg_spm_latency": g(a + "avgSpmLatency"),
        "avg_spm_service_latency": g(a + "avgSpmServiceLatency"),
        "lane_retries": g(a + "laneRetries"),
        # --- DRAM and host
        "dram_bytes": g(a + "dramBytes"),
        "dram_bus_util": g("system.mem_ctrl.dram.busUtil"),
        "dram_read_bw_GBs": g("system.mem_ctrl.avgRdBWSys") / 1e9,
        "dram_energy_J": dram_energy_pj * 1e-12,
        "edp_dram_Js": dram_energy_pj * 1e-12 * layer_s,
        "host_cpi": g("system.cpu.cpi") if host else float("nan"),
        "host_ipc": g("system.cpu.ipc") if host else float("nan"),
        "host_insts": g("system.cpu.commitStats0.numInsts")
        if host else float("nan"),
        "verify_mismatches": 0 if sw else g(a + "verifyMismatches"),
        "run_dir": repo_path(d),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--output", default="-")
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        if not os.path.exists(os.path.join(d, "stats.txt")):
            print(f"skip {d}: no stats.txt", file=sys.stderr)
            continue
        try:
            rows.append(summarize(d))
        except Exception as e:  # keep going on one bad run
            print(f"skip {d}: {e}", file=sys.stderr)
    if not rows:
        sys.exit("no runs parsed")
    rows.sort(key=lambda r: (r["workload"], r["mode"], r["double_buffer"],
                             r["N"], r["banks"]))
    out = sys.stdout if args.output == "-" else open(args.output, "w",
                                                     newline="")
    w = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    if out is not sys.stdout:
        out.close()
        print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
