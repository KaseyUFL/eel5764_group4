#!/usr/bin/env python3
#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : trace_diff.py                                                 #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : Finds where two per-cycle bank traces differ                  #
#-----------------------------------------------------------------------------#
#
#   $PYTHON analysis/trace_diff.py GEM5.csv RTL.csv
#   $PYTHON analysis/trace_diff.py GEM5.csv RTL.csv --offset 3 --context 2
#
# A total-cycle difference says nothing about where it came from; this says
# which cycle and which bank.  Both files have one row per compute-phase
# cycle in the format of systolic_se.py --bank-trace, which the RTL testbench
# can $fwrite too:
#
#   cycle,sched_step,stalled,stall_cause,bank0_op,bank0_addr,bank1_op,...
#
#   cycle        compute-phase cycle, from 0
#   sched_step   the schedule step the array is on
#   stalled      1 if the array did not advance this cycle
#   stall_cause  none | read | write | dma | drain
#   bankN_op     R | W | idle (RW with --bank-ports 1r1w)
#   bankN_addr   word address inside the bank (read/write for RW)
#
# Compared: stalled and every bank's op and address.  sched_step and
# stall_cause are shown at a divergence but only compared with --strict,
# since an RTL has no reason to name its stalls the way gem5 does.
#
# --offset K drops the first K rows of the second file (or, if negative, of
# the first), for a fixed pipeline difference at the start.
#
# Exit status: 0 when the traces match, 1 when they diverge.

import argparse
import csv
import sys


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: no rows")
    return rows


def compared_columns(a, b, strict):
    cols = ["stalled"] + [c for c in a[0] if c.startswith("bank")]
    if strict:
        cols += ["sched_step", "stall_cause"]
    missing = [c for c in cols if c not in b[0]]
    if missing:
        sys.exit("the second trace lacks columns: " + ", ".join(missing[:8]) +
                 " (different BANKS?)")
    return cols


def main():
    ap = argparse.ArgumentParser(
        description="First cycle where two bank traces disagree")
    ap.add_argument("first", help="usually the gem5 trace")
    ap.add_argument("second", help="usually the RTL trace")
    ap.add_argument("--offset", type=int, default=0,
                    help="rows to drop from the start of the second trace "
                         "(negative: of the first)")
    ap.add_argument("--context", type=int, default=2,
                    help="rows to show around the first divergence")
    ap.add_argument("--strict", action="store_true",
                    help="also compare sched_step and stall_cause")
    args = ap.parse_args()

    a, b = read(args.first), read(args.second)
    if args.offset > 0:
        b = b[args.offset:]
    elif args.offset < 0:
        a = a[-args.offset:]
    cols = compared_columns(a, b, args.strict)

    first = None
    diverging = 0
    per_col = {}
    n = min(len(a), len(b))
    for i in range(n):
        bad = [c for c in cols if a[i][c] != b[i][c]]
        if bad:
            diverging += 1
            if first is None:
                first = i
            for c in bad:
                per_col[c] = per_col.get(c, 0) + 1

    stalls = [sum(r["stalled"] == "1" for r in t) for t in (a, b)]
    print(f"rows compared   {n} ({args.first}: {len(a)}, "
          f"{args.second}: {len(b)})")
    print(f"stalled cycles  {stalls[0]} vs {stalls[1]}")
    if first is None:
        if len(a) != len(b):
            print(f"no divergence in the common rows, but one trace is "
                  f"longer by {abs(len(a) - len(b))} rows")
            sys.exit(1)
        print("the traces match")
        return

    print(f"diverging rows  {diverging}")
    print(f"first at row    {first} (cycle {a[first]['cycle']} of the first "
          f"trace, {b[first]['cycle']} of the second)")
    worst = sorted(per_col.items(), key=lambda kv: -kv[1])[:6]
    print("most different  " + ", ".join(f"{c} ({k})" for c, k in worst))
    show = [c for c in ("cycle", "sched_step", "stalled", "stall_cause")]
    bad = [c for c in cols if a[first][c] != b[first][c]
           and c.startswith("bank")]
    show += bad[:8]
    print()
    print("  ".join(f"{c:>12s}" for c in ["trace"] + show))
    for i in range(max(0, first - args.context),
                   min(n, first + args.context + 1)):
        mark = ">>" if i == first else "  "
        for name, t in (("first", a), ("second", b)):
            print(mark + "  ".join(f"{str(t[i].get(c, '')):>12s}"
                                   for c in [name] + show))
    sys.exit(1)


if __name__ == "__main__":
    main()
