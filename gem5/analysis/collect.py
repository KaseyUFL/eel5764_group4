#!/usr/bin/env python3
#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : collect.py                                                    #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : Runs the gem5 experiment suites and gathers data              #
#-----------------------------------------------------------------------------#
#
#   make data                                  # every suite (about 2 h)
#   make data SUITES="main knee"               # some suites
#   make list                                  # show suites
#   make data FORCE=1                          # rerun, no cache
#
# or directly (after  source scripts/env.sh ):
#
#   $PYTHON analysis/collect.py [SUITE ...] [-j JOBS] [--force] [--list]
#
# Runs go to build/runs/<tag>/ (gem5 -d directory: stats.txt, config.ini,
# simout.txt, simerr.txt, run.json).  A point shared by two suites (e.g.
# main and banks) is simulated once.  Finished runs are reused, so an
# interrupted collection can simply be restarted; use --force after
# rebuilding gem5 or changing model defaults.
#
# A run that fails does not stop the others; collect reports it at the end.
#
# Output: results/data/<suite>.csv (or $RESULTS_DIR/data), one row per run,
# with every column of
# analysis/parse_stats.py (utilization, stalls, bandwidth, AMAT, DRAM,
# model prediction, verify result, ...).

import argparse
import concurrent.futures as cf
import csv
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import experiments  # noqa: E402
import parse_stats  # noqa: E402

CONFIG = os.path.join(ROOT, "gem5_model", "configs", "systolic_se.py")
WL_DIR = os.path.join(os.environ.get("WORKLOAD_ROOT",
                                     os.path.join(ROOT, "workloads")),
                      "out", "gem5")
GEM5_BIN = os.environ.get("GEM5_BIN", os.path.join(
    os.environ.get("GEM5_DIR", os.path.join(ROOT, "third_party", "gem5")),
    "build", "RISCV_SYSTOLIC", "gem5.opt"))
HOST_BIN = os.environ.get("HOST_BIN", os.path.join(
    ROOT, "gem5_model", "host", "systolic_host"))
DATA = os.path.join(os.environ.get("RESULTS_DIR",
                                   os.path.join(ROOT, "results")), "data")
RUNS = os.path.join(os.environ.get("BUILD_DIR", os.path.join(ROOT, "build")),
                    "runs")


def finished(d):
    so = os.path.join(d, "simout.txt")
    if not (os.path.exists(os.path.join(d, "stats.txt")) and
            os.path.exists(so)):
        return False
    with open(so) as f:
        return "systolic accelerator finished" in f.read()


def shape(p):
    with open(os.path.join(WL_DIR, p["workload"], "meta.json")) as f:
        m = json.load(f)
    return m["M"], m["K"], m["N_out"]


def command(p, d):
    cmd = [GEM5_BIN, "-re", "-d", d, CONFIG,
           "--array-dim", str(p["N"]), "--banks", str(p["banks"]),
           "--arbitration", p["arbitration"]]
    cmd += ["--workload", p["workload"], "--workload-dir", WL_DIR]
    if p["mode"] != "standalone":
        cmd += ["--host-bin", HOST_BIN]
    if p["mode"] == "standalone":
        cmd.append("--standalone")
    elif p["mode"] == "sw":
        cmd += ["--mode", "sw"]
    for k, v in sorted(p["opts"].items()):
        cmd.append(experiments.OPT_FLAGS[k])
        if k not in experiments.OPT_FLAG_ONLY:
            cmd.append(str(v))
    if not p["double_buffer"]:
        cmd.append("--no-double-buffer")
    if p["ideal_spm"]:
        cmd.append("--ideal-spm")
    if p["ideal_dram"]:
        cmd.append("--ideal-dram")
    return cmd


def run(p, force):
    d = os.path.join(RUNS, p["tag"])
    if finished(d) and not force:
        return p, True, 0.0, "cached"
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    M, K, NOUT = shape(p)
    cmd = command(p, d)
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(dict(p, M=M, K=K, N_out=NOUT, cmd=cmd), f, indent=2)
    t0 = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, finished(d), time.time() - t0, ""


def write_csv(suite, points):
    rows = []
    for p in points:
        d = os.path.join(RUNS, p["tag"])
        if finished(d):
            row = parse_stats.summarize(d)
            row["suite"] = suite
            row["tag"] = p["tag"]
            rows.append(row)
    if not rows:
        return 0
    path = os.path.join(DATA, f"{suite}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Collect gem5 data points")
    ap.add_argument("suites", nargs="*", help="default: all suites")
    ap.add_argument("-j", "--jobs", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--force", action="store_true",
                    help="rerun points even if a finished run exists")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    names = args.suites or list(experiments.SUITES)
    for n in names:
        if n not in experiments.SUITES:
            sys.exit(f"unknown suite '{n}'; choose from "
                     f"{', '.join(experiments.SUITES)}")
    suites = {n: experiments.SUITES[n]() for n in names}
    if args.list:
        for n, pts in suites.items():
            print(f"{n:10s} {len(pts):3d} runs")
        return
    if not os.path.exists(GEM5_BIN):
        sys.exit(f"gem5 binary not found: {GEM5_BIN}\nrun: make gem5")
    if not os.path.exists(os.path.join(WL_DIR, WORKLOAD_CHECK)):
        sys.exit(f"workloads missing in {WL_DIR}; run: make workloads")
    if not os.path.exists(HOST_BIN):
        sys.exit(f"RISC-V host program missing: {HOST_BIN}\n"
                 "run: make host")

    # Unique points across the requested suites (shared tags run once).
    uniq = {}
    for pts in suites.values():
        for p in pts:
            uniq.setdefault(p["tag"], p)
    for p in uniq.values():
        if "prepacked_weights" in p["opts"] and not os.path.exists(
                os.path.join(WL_DIR, p["workload"],
                             f"W_packed_N{p['N']}.bin")):
            sys.exit(f"packed weights missing for {p['tag']}; "
                     "run: make prepacked")
    # Longest runs first (software baselines, then the GEMV runs whose
    # host packs 16 MiB of weights), so they overlap with the short ones.
    todo = sorted(uniq.values(), key=lambda p: (
        p["mode"] != "sw", "gemv" not in p["workload"],
        "bert_ffn1" not in p["workload"], p["N"]))
    print(f"{len(todo)} unique runs for suites {', '.join(names)} "
          f"({args.jobs} parallel) -> {RUNS}")
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    fails = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run, p, args.force) for p in todo]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            p, ok, dt, note = fu.result()
            if not ok:
                fails.append(p["tag"])
            print(f"[{i:3d}/{len(todo)}] {p['tag']:44s} "
                  f"{'ok' if ok else 'FAILED':6s} {note or f'{dt:5.1f}s'}",
                  flush=True)
    print(f"simulation time {time.time() - t0:.0f} s")

    for n, pts in suites.items():
        k = write_csv(n, pts)
        print(f"wrote {os.path.relpath(os.path.join(DATA, n + '.csv'))} "
              f"({k} rows)")
    if fails:
        print(f"FAILED runs (see {RUNS}/<tag>/simerr.txt):")
        for t in fails:
            print("  ", t)
        sys.exit(1)


WORKLOAD_CHECK = os.path.join("reuse_M1", "meta.json")

if __name__ == "__main__":
    main()
