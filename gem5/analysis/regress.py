#!/usr/bin/env python3
#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : regress.py                                                    #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : Regression check of the default model                         #
#-----------------------------------------------------------------------------#
#
#   make regress                        # rerun the points, diff against golden
#   bash analysis/regress.sh            # the same, without make
#   $PYTHON analysis/regress.py --update --from-runs build/runs
#                                       # rewrite golden_stats.csv from
#                                       # finished runs (only after a change
#                                       # that is meant to move the numbers)
#
# Every new model feature sits behind an option whose default must reproduce
# the existing results.  gem5 is deterministic, so this reruns a few
# representative points of the default model and compares their statistics
# with analysis/golden_stats.csv EXACTLY, without a tolerance.  Runs go to
# build/regress/<tag>/; each takes seconds except the GEMV point (about a
# minute: the host packs 16 MiB of weights).
#
# Exit status 0 when every point matches.

import argparse
import concurrent.futures as cf
import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import collect  # noqa: E402
from experiments import _pt  # noqa: E402
import parse_stats  # noqa: E402

GOLDEN = os.path.join(HERE, "golden_stats.csv")
OUT = os.path.join(os.environ.get("BUILD_DIR", os.path.join(ROOT, "build")),
                   "regress")

RESNET = "resnet50_conv3x"
POINTS = [
    # Main grid: both ends of N and BANKS.
    _pt(RESNET, 8, 4),
    _pt(RESNET, 8, 32),
    _pt(RESNET, 32, 4),
    _pt(RESNET, 32, 32),
    # Low reuse, DRAM-bound.
    _pt("llama7b_gemv", 16, 8),
    # The rule's bank count (non-power of two, modulo mapping).
    _pt("bert_ffn1", 32, 21),
    # Infinite-bandwidth baseline.
    _pt(RESNET, 16, 32, ideal_spm=True),
    # BANKS = N - 1: partial-sum lanes alias.
    _pt(RESNET, 24, 23),
    # The other existing modes: lane arbitration, no CPU, reads-first
    # queues, and a reduced RTL-validation layer.
    _pt(RESNET, 16, 8, arb="lane"),
    _pt(RESNET, 16, 8, mode="standalone"),
    _pt(RESNET, 32, 21, bank_policy="reads_first"),
    _pt("bert_attn_qk_rtl", 8, 4),
]

# Statistics of the accelerator (all but the two gem5 globals at the end).
STATS = ["peUtilization", "idealUtilization", "layerUtilization",
         "totalCycles", "scheduleCycles", "activeCycles", "stallCycles",
         "spmStallCycles", "spmReadStallCycles", "spmWriteStallCycles",
         "dmaStallCycles", "drainCycles", "loadCycles", "storeCycles",
         "layerCycles", "laneRetries", "spmReads", "verifyMismatches"]
GLOBALS = ["simTicks", "simSeconds"]


def row_from(run_dir, tag):
    p = parse_stats.run_params(run_dir)
    dumps = parse_stats.read_dumps(os.path.join(run_dir, "stats.txt"))
    s = dumps[int(p.get("layer_dump", 0))]
    row = {"tag": tag}
    for k in STATS:
        row[k] = repr(s.get("system.accel." + k, float("nan")))
    for k in GLOBALS:
        row[k] = repr(s.get(k, float("nan")))
    return row


def run_point(p):
    d = os.path.join(OUT, p["tag"])
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    M, K, NOUT = collect.shape(p)
    cmd = collect.command(p, d)
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(dict(p, M=M, K=K, N_out=NOUT, cmd=cmd), f, indent=2)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not collect.finished(d):
        return p["tag"], None
    return p["tag"], row_from(d, p["tag"])


def write_golden(rows):
    with open(GOLDEN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag"] + STATS + GLOBALS,
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Regression of the default "
                                 "model against golden_stats.csv")
    ap.add_argument("-j", "--jobs", type=int, default=len(POINTS))
    ap.add_argument("--update", action="store_true",
                    help="rewrite golden_stats.csv instead of comparing")
    ap.add_argument("--from-runs", metavar="DIR",
                    help="with --update: read finished runs from DIR/<tag>/ "
                         "instead of running gem5")
    args = ap.parse_args()

    if args.update and args.from_runs:
        write_golden([row_from(os.path.join(args.from_runs, p["tag"]),
                               p["tag"]) for p in POINTS])
        print(f"wrote {GOLDEN} from {args.from_runs}")
        return

    print(f"running {len(POINTS)} regression points -> {OUT}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        results = dict(ex.map(run_point, POINTS))

    failed = [t for t, r in results.items() if r is None]
    if args.update:
        if failed:
            sys.exit(f"runs failed: {', '.join(failed)}")
        write_golden([results[p["tag"]] for p in POINTS])
        print(f"wrote {GOLDEN}")
        return

    with open(GOLDEN, newline="") as f:
        golden = {r["tag"]: r for r in csv.DictReader(f)}
    bad = 0
    for p in POINTS:
        tag = p["tag"]
        new = results[tag]
        if new is None:
            print(f"FAILED   {tag} (see {OUT}/{tag}/simerr.txt)")
            bad += 1
            continue
        if tag not in golden:
            print(f"NEW      {tag} is not in golden_stats.csv")
            bad += 1
            continue
        diffs = [f"{k}: {golden[tag][k]} -> {new[k]}"
                 for k in STATS + GLOBALS if golden[tag][k] != new[k]]
        print(f"{'same' if not diffs else 'DIFFERS':8s} {tag}")
        for d in diffs:
            print(f"           {d}")
        bad += bool(diffs)
    if bad:
        print(f"\n{bad} of {len(POINTS)} points differ from golden_stats.csv")
        sys.exit(1)
    print(f"\nall {len(POINTS)} points match golden_stats.csv exactly")


if __name__ == "__main__":
    main()
