#!/usr/bin/env python3
#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : gen_workloads.py                                              #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : Workload generator (single source of truth)                   #
#-----------------------------------------------------------------------------#
#
# Reads workloads.yaml and, for every workload, writes INT8 operands and the
# INT32 golden result of  C[M x N_out] = A[M x K] . W[K x N_out]  in the
# formats each simulator needs:
#
#   out/gem5/<name>/          full-size layer, raw little-endian binaries
#   out/gem5/<name>_rtl/      reduced-size layer (same M, K/N_out capped)
#   out/rtl/<name>_rtl/       the reduced layer as $readmemh hex files
#   out/workloads.json        index of everything above
#
# Usage:
#   python3 gen_workloads.py                 # all workloads
#   python3 gen_workloads.py --only bert_attn_qk llama7b_gemv
#   python3 gen_workloads.py --no-full       # only the reduced variants
#   python3 gen_workloads.py --only reuse    # only the synthetic reuse GEMMs
#
# Needs numpy and PyYAML.  The seed in workloads.yaml makes every file
# bit-for-bit reproducible.

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def int8(rng, shape, low=-128, high=127):
    return rng.integers(low, high + 1, size=shape, dtype=np.int16).astype(
        np.int8
    )


def im2col(x, R, S, stride, pad):
    """x is [H, W, C] (HWC).  Returns A[(Ho*Wo) x (R*S*C)] whose column
    order is (r, s, c), matching a filter reshaped from [R, S, C, C_out]."""
    H, W, C = x.shape
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    Ho = (H + 2 * pad - R) // stride + 1
    Wo = (W + 2 * pad - S) // stride + 1
    cols = np.empty((Ho, Wo, R, S, C), dtype=x.dtype)
    for r in range(R):
        for s in range(S):
            cols[:, :, r, s, :] = xp[
                r : r + stride * Ho : stride, s : s + stride * Wo : stride, :
            ]
    return cols.reshape(Ho * Wo, R * S * C)


def direct_conv(x, f, stride, pad):
    """Reference convolution (no im2col) used to check the lowering."""
    H, W, C = x.shape
    R, S, _, K = f.shape
    xp = np.pad(x.astype(np.int32), ((pad, pad), (pad, pad), (0, 0)))
    Ho = (H + 2 * pad - R) // stride + 1
    Wo = (W + 2 * pad - S) // stride + 1
    out = np.zeros((Ho, Wo, K), dtype=np.int32)
    f32 = f.astype(np.int32)
    for r in range(R):
        for s in range(S):
            win = xp[r : r + stride * Ho : stride, s : s + stride * Wo : stride]
            out += np.tensordot(win, f32[r, s], axes=([2], [0]))
    return out.reshape(Ho * Wo, K)


def make_operands(name, spec, rng):
    """Returns (A, W, extra_meta) with the shapes in spec."""
    M, K, N = spec["M"], spec["K"], spec["N_out"]
    kind = spec.get("kind", "gemm")
    extra = {}
    if kind == "conv_im2col":
        c = spec["conv"]
        # Post-ReLU activations are non-negative; keep them in [0, 127] so
        # they still fit the signed INT8 datapath.
        x = int8(rng, (c["H"], c["W"], c["C_in"]), 0, 127)
        f = int8(rng, (c["R"], c["S"], c["C_in"], c["C_out"]))
        A = im2col(x, c["R"], c["S"], c["stride"], c["pad"])
        W = f.reshape(c["R"] * c["S"] * c["C_in"], c["C_out"])
        ref = direct_conv(x, f, c["stride"], c["pad"])
        ok = np.array_equal(A.astype(np.int32) @ W.astype(np.int32), ref)
        if not ok:
            sys.exit(f"{name}: im2col GEMM does not match direct conv")
        extra["im2col_checked_against_direct_conv"] = True
        extra["im2col_column_order"] = "(r, s, c_in)"
    elif kind == "attention_qk":
        Q = int8(rng, (M, K))
        Kmat = int8(rng, (N, K))
        A, W = Q, np.ascontiguousarray(Kmat.T)
        extra["note"] = "A = Q_h [M x d_k], W = K_h^T [d_k x N_out]"
    elif kind == "gemv":
        A = int8(rng, (M, K))
        W = int8(rng, (K, N))
        extra["note"] = "A = x [1 x K], W = projection [K x N_out]"
    else:
        A = int8(rng, (M, K))
        W = int8(rng, (K, N))
    assert A.shape == (M, K) and W.shape == (K, N), (A.shape, W.shape)
    return np.ascontiguousarray(A), np.ascontiguousarray(W), extra


def golden(A, W):
    return A.astype(np.int32) @ W.astype(np.int32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_gem5(outdir, name, spec, A, W, C, extra):
    os.makedirs(outdir, exist_ok=True)
    files = {}
    for tag, arr, dt in (("A", A, "<i1"), ("W", W, "<i1"), ("C", C, "<i4")):
        fn = f"{tag}.bin" if tag != "C" else "C_golden.bin"
        path = os.path.join(outdir, fn)
        arr.astype(dt).tofile(path)
        files[tag] = {"file": fn, "sha256": sha256(path)}
    meta = {
        "name": name,
        "class": spec["class"],
        "M": int(A.shape[0]),
        "K": int(A.shape[1]),
        "N_out": int(W.shape[1]),
        "A": dict(files["A"], dtype="int8", layout="row-major M x K"),
        "W": dict(files["W"], dtype="int8", layout="row-major K x N_out"),
        "C_golden": dict(
            files["C"], dtype="int32 little-endian", layout="row-major M x N_out"
        ),
        "layer": spec.get("layer", "").strip(),
        "source": spec.get("source", "").strip(),
    }
    meta.update(extra)
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def write_prepacked(outdir, W, n_values):
    """W_packed_N<N>.bin: W as the host program packs it for an N x N array.
    Tile t = nb * nkb + kb, tile row r holds W[kb*N + r][nb*N .. nb*N + N-1],
    zero padded.  systolic_se.py --prepacked-weights loads it into DRAM."""
    K, NOUT = W.shape
    for N in n_values:
        nkb, nnb = -(-K // N), -(-NOUT // N)
        padded = np.zeros((nkb * N, nnb * N), dtype=np.int8)
        padded[:K, :NOUT] = W
        # (nnb, nkb, N, N): the file order is nb outer, kb inner.
        tiles = padded.reshape(nkb, N, nnb, N).transpose(2, 0, 1, 3)
        tiles.tofile(os.path.join(outdir, f"W_packed_N{N}.bin"))


def write_hex(path, arr, width_bytes):
    mask = (1 << (8 * width_bytes)) - 1
    fmt = f"{{:0{2 * width_bytes}x}}\n"
    flat = arr.reshape(-1).astype(np.int64) & mask
    with open(path, "w") as f:
        f.write(
            f"// {arr.shape[0]} x {arr.shape[1]} row-major, one "
            f"{8 * width_bytes}-bit two's-complement value per line\n"
        )
        f.writelines(fmt.format(int(v)) for v in flat)


def write_rtl(outdir, name, A, W, C):
    os.makedirs(outdir, exist_ok=True)
    write_hex(os.path.join(outdir, "A.hex"), A, 1)
    write_hex(os.path.join(outdir, "W.hex"), W, 1)
    write_hex(os.path.join(outdir, "C_golden.hex"), C, 4)
    M, K = A.shape
    N = W.shape[1]
    with open(os.path.join(outdir, "shape.svh"), "w") as f:
        f.write(
            f"// {name}: C[M x N_OUT] = A[M x K] . W[K x N_OUT]\n"
            f"localparam int WL_M     = {M};\n"
            f"localparam int WL_K     = {K};\n"
            f"localparam int WL_N_OUT = {N};\n"
        )
    return {"name": name, "M": M, "K": K, "N_out": N,
            "files": ["A.hex", "W.hex", "C_golden.hex", "shape.svh"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(HERE, "workloads.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", nargs="*", help="workload names to generate")
    ap.add_argument("--no-full", action="store_true",
                    help="skip the full-size gem5 binaries")
    ap.add_argument("--prepack", action="store_true",
                    help="write only W_packed_N<N>.bin next to the existing "
                         "W.bin of each workload (N = 8, 16, 24, 32), for "
                         "systolic_se.py --prepacked-weights")
    args = ap.parse_args()

    if args.prepack:
        for name in sorted(os.listdir(os.path.join(args.out, "gem5"))):
            if args.only and name not in args.only:
                continue
            d = os.path.join(args.out, "gem5", name)
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            W = np.fromfile(os.path.join(d, "W.bin"), dtype=np.int8)
            write_prepacked(d, W.reshape(meta["K"], meta["N_out"]),
                            (8, 16, 24, 32))
            print(f"packed {name}")
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    red = cfg["rtl_reduced"]
    index = {"seed": cfg["seed"], "gem5": {}, "rtl": {}}
    idx_path = os.path.join(args.out, "workloads.json")
    if args.only and os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)

    for i, (name, spec) in enumerate(cfg["workloads"].items()):
        if args.only and name not in args.only:
            continue
        # Per-workload seed so --only regenerates identical data.
        rng = np.random.default_rng([cfg["seed"], i])
        A, W, extra = make_operands(name, spec, rng)
        C = golden(A, W)
        M, K, N = spec["M"], spec["K"], spec["N_out"]
        macs = M * K * N
        print(f"{name:18s} M={M:5d} K={K:5d} N_out={N:5d}  "
              f"{macs / 1e6:9.1f} MMAC  A={A.nbytes / 1024:8.1f} KiB  "
              f"W={W.nbytes / 1024:8.1f} KiB")

        if not args.no_full:
            d = os.path.join(args.out, "gem5", name)
            index["gem5"][name] = write_gem5(d, name, spec, A, W, C, extra)

        kr, nr = min(K, red["max_k"]), min(N, red["max_n_out"])
        Ar = np.ascontiguousarray(A[:, :kr])
        Wr = np.ascontiguousarray(W[:kr, :nr])
        Cr = golden(Ar, Wr)
        rname = f"{name}_rtl"
        rextra = dict(extra, reduced_from=name,
                      reduction=f"A[:, :{kr}], W[:{kr}, :{nr}]")
        d = os.path.join(args.out, "gem5", rname)
        index["gem5"][rname] = write_gem5(d, rname, spec, Ar, Wr, Cr, rextra)
        d = os.path.join(args.out, "rtl", rname)
        index["rtl"][rname] = write_rtl(d, rname, Ar, Wr, Cr)

    # Synthetic reuse sweep (gem5 only)
    rs = cfg.get("reuse_sweep")
    if rs and (not args.only or "reuse" in args.only):
        rng = np.random.default_rng([cfg["seed"], 1000])
        for m in rs["M"]:
            name = f"reuse_M{m}"
            spec = {"class": "synthetic", "M": m, "K": rs["K"],
                    "N_out": rs["N_out"],
                    "layer": f"synthetic GEMM {m} x {rs['K']} x "
                             f"{rs['N_out']} for the reuse sweep",
                    "source": "none (synthetic)"}
            A = int8(rng, (m, rs["K"]))
            W = int8(rng, (rs["K"], rs["N_out"]))
            d = os.path.join(args.out, "gem5", name)
            index["gem5"][name] = write_gem5(d, name, spec, A, W,
                                             golden(A, W), {})
        print(f"reuse sweep: M = {rs['M']} x {rs['K']} x {rs['N_out']}")

    os.makedirs(args.out, exist_ok=True)
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {idx_path}")


if __name__ == "__main__":
    main()
