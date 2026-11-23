#!/usr/bin/env python3
#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : compare_results.py                                            #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : Compares two result sets, number by number                    #
#-----------------------------------------------------------------------------#
#
#   make check REF=path/to/results   # results/ against a reference copy
#   $PYTHON analysis/compare_results.py REF [NEW] [--rel 1e-9] [--tol 0]
#
# REF and NEW are results directories, each with data/ and tables/.  NEW
# defaults to results/ (or $RESULTS_DIR).  Compared: every
# data/<suite>.csv row by row (matched on the run tag), every tables/*.csv
# cell by cell, and every value in data/key_results.json.  The run_dir
# column is ignored: it only says where the raw gem5 output was.  Figures
# are not compared; they follow from the data, and their pixels depend on
# the matplotlib version.
#
# A column or a key_results.json value that only NEW has is an addition: it
# is listed but does not count as a difference, so results stay comparable
# with a reference made before a statistic was added.
#
# Simulation output (data/<suite>.csv) is deterministic and must match
# exactly.  Derived numbers (tables, key_results.json) may differ in the
# last bits between numpy/pandas versions, which sum in different orders;
# differences below --rel (relative, default 1e-9) are reported as
# round-off and accepted.  Exit status 0 when nothing differs by more.

import argparse
import csv
import glob
import json
import math
import os
import sys

ROUNDOFF = [0]   # values that differed only by round-off
ADDITIONS = set()   # columns and values only the new results have
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IGNORE = {"run_dir"}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def same(a, b, tol, rel=0.0):
    """Two cells are equal if both are numbers within tol, or within rel
    of each other (NaN == NaN), or both are the same string."""
    x, y = num(a), num(b)
    if x is not None and y is not None:
        if math.isnan(x) or math.isnan(y):
            return math.isnan(x) and math.isnan(y)
        if x == y or abs(x - y) <= tol:
            return True
        if abs(x - y) <= rel * max(abs(x), abs(y)):
            ROUNDOFF[0] += 1
            return True
        return False
    return a == b


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compare_rows(ref, new, key, tol, rel):
    """Returns a list of difference strings."""
    out = []
    cols = [c for c in ref[0] if c not in IGNORE] if ref else []
    extra = sorted(set(new[0]) - set(ref[0]) - IGNORE) if new and ref else []
    ADDITIONS.update(extra)
    if key:
        r = {row[key]: row for row in ref}
        n = {row[key]: row for row in new}
        for k in sorted(set(r) - set(n)):
            out.append(f"missing row {k}")
        for k in sorted(set(n) - set(r)):
            out.append(f"extra row {k}")
        pairs = [(k, r[k], n[k]) for k in sorted(set(r) & set(n))]
    else:
        if len(ref) != len(new):
            out.append(f"{len(ref)} rows -> {len(new)} rows")
        pairs = list(zip(range(len(ref)), ref, new))
    worst = {}
    for k, a, b in pairs:
        for c in cols:
            if c not in b:
                worst.setdefault(c, ["missing column", 0, k])
                continue
            if not same(a[c], b[c], tol, rel):
                x, y = num(a[c]), num(b[c])
                d = abs(x - y) if x is not None and y is not None else None
                w = worst.setdefault(c, [d, 0, k])
                w[1] += 1
                if d is not None and (w[0] is None or d > w[0]):
                    w[0], w[2] = d, k
    for c, (d, count, k) in sorted(worst.items()):
        size = f"max |diff| {d:.6g}" if isinstance(d, float) else str(d)
        out.append(f"{c}: {count} row(s) differ, {size} (row {k})")
    return out


def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], x


def compare_json(ref_path, new_path, tol, rel):
    with open(ref_path) as f:
        r = dict(flatten(json.load(f)))
    with open(new_path) as f:
        n = dict(flatten(json.load(f)))
    out = [f"missing {k}" for k in sorted(set(r) - set(n))]
    ADDITIONS.update(sorted(set(n) - set(r)))
    for k in sorted(set(r) & set(n)):
        a, b = r[k], n[k]
        if a is None or b is None or isinstance(a, (str, bool)):
            if a != b:
                out.append(f"{k}: {a!r} -> {b!r}")
        elif not same(a, b, tol, rel):
            out.append(f"{k}: {a!r} -> {b!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref", help="reference results directory")
    ap.add_argument("new", nargs="?", default=os.environ.get(
        "RESULTS_DIR", os.path.join(ROOT, "results")))
    ap.add_argument("--tol", type=float, default=0.0,
                    help="largest absolute difference treated as equal")
    ap.add_argument("--rel", type=float, default=1e-9,
                    help="largest relative difference treated as round-off "
                         "in derived files (tables, key_results.json)")
    args = ap.parse_args()

    files = []
    for path in sorted(glob.glob(os.path.join(args.ref, "data", "*.csv"))):
        files.append((os.path.relpath(path, args.ref), "tag"))
    for path in sorted(glob.glob(os.path.join(args.ref, "tables", "*.csv"))):
        files.append((os.path.relpath(path, args.ref), None))
    if not files:
        sys.exit(f"no data/*.csv or tables/*.csv under {args.ref}")
    files.append((os.path.join("data", "key_results.json"), "json"))

    bad = 0
    for path, key in files:
        rp, np_ = os.path.join(args.ref, path), os.path.join(args.new, path)
        if not os.path.exists(rp):
            continue
        if not os.path.exists(np_):
            print(f"MISSING  {path}")
            bad += 1
            continue
        if key == "json":
            diffs = compare_json(rp, np_, args.tol, args.rel)
        else:
            # simulation output must match exactly; derived tables may
            # carry floating-point round-off
            rel = 0.0 if key == "tag" else args.rel
            diffs = compare_rows(read_csv(rp), read_csv(np_), key, args.tol,
                                 rel)
        rows = "" if key == "json" else f" ({len(read_csv(np_))} rows)"
        print(f"{'same' if not diffs else 'DIFFERS':8s} {path}{rows}")
        for d in diffs[:12]:
            print(f"           {d}")
        if len(diffs) > 12:
            print(f"           ... {len(diffs) - 12} more")
        bad += bool(diffs)
    if bad:
        print(f"\n{bad} of {len(files)} files differ from {args.ref}")
        sys.exit(1)
    if ADDITIONS:
        print(f"\n{len(ADDITIONS)} columns/values are new (not compared): "
              + ", ".join(sorted(ADDITIONS)[:20])
              + (", ..." if len(ADDITIONS) > 20 else ""))
    how = "exactly" if args.tol == 0 else f"within {args.tol:g}"
    if ROUNDOFF[0]:
        how = (f"(simulation data {how}; {ROUNDOFF[0]} derived values "
               f"differ only by floating-point round-off, < {args.rel:g} "
               "relative)")
    print(f"\nall {len(files)} files match {args.ref} {how} "
          "(run_dir ignored)")

if __name__ == "__main__":
    main()
