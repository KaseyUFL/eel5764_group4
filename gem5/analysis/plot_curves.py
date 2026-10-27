#!/usr/bin/env python3
# EEL5764 Group 4 -- saturation curves, measurements and tables.
#
#   make figures                                   # everything
#   make figures PLOTS="fig1 fig5 tables"          # some of it
#   $PYTHON analysis/plot_curves.py --format pdf   # one format only
#
# Reads results/data/<suite>.csv (written by collect.py) and writes
# results/figures/ (PNG + PDF), results/tables/ (Markdown + CSV) and
# results/data/key_results.json (results/ = $RESULTS_DIR if set).  A
# figure whose suite has not been
# collected is skipped with a note.  The files carry no dates or version
# strings, so rerunning on unchanged data rewrites identical files.
#
# Optional inputs, used when present:
#   results/data/rtl_cycles.csv  workload,N,banks,cycles -> fig12 (gem5 vs RTL)
#   results/data/vivado.csv      N,power_W,fmax_MHz,dsp,lut,ff,bram
#                                                     -> fig13 (EDP), table
#   (templates: analysis/templates/*.example)

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import experiments as ex  # noqa: E402
import model  # noqa: E402
import style as st  # noqa: E402
from style import plt  # noqa: E402

RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
DATA = os.path.join(RESULTS, "data")
FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")
FORMATS = ["png", "pdf"]
# No creation dates or library versions in the files (reproducible output).
METADATA = {"png": {"Software": None},
            "pdf": {"Creator": None, "Producer": None, "CreationDate": None},
            "svg": {"Creator": None, "Date": None}}
ONE_COL, TWO_COL = 3.5, 7.16  # IEEE column widths, inches


# ----------------------------------------------------------------- helpers
def load(suite):
    p = os.path.join(DATA, f"{suite}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    bad = df[df["verify_mismatches"] != 0]
    if len(bad):
        print(f"  WARNING: {len(bad)} run(s) in {suite}.csv failed verify")
    return df


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for fmt in FORMATS:
        fig.savefig(os.path.join(FIG, f"{name}.{fmt}"),
                    metadata=METADATA[fmt])
    plt.close(fig)
    print(f"  results/figures/{name}.{{{','.join(FORMATS)}}}")


def write_table(df, name, title, notes=""):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w") as f:
        f.write(f"# {title}\n\n")
        if notes:
            f.write(notes.strip() + "\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4g"))
        f.write("\n")
    print(f"  results/tables/{name}.{{md,csv}}")


def shape_of(df, wl):
    r = df[df.workload == wl].iloc[0]
    return int(r.M), int(r.K), int(r.N_out)


def pred(wl_df_row, db=True):
    r = wl_df_row
    return model.predict(int(r.N), int(r.banks), int(r.M), int(r.K),
                         int(r.N_out), db)["pred_util"]


def crossing(xs, ys, thr=0.9):
    """First x where ys drops below thr (linear interpolation)."""
    xs, ys = list(xs), list(ys)
    if ys[0] < thr:
        return f"< {xs[0]}"
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= thr > y1:
            return f"{x0 + (y0 - thr) / (y0 - y1) * (x1 - x0):.1f}"
    return f"> {xs[-1]}"


def model_n90(M, banks, thr=0.9):
    """Continuous-N answer of eq. (1)-(2): where (M/T)*min(1, B/D) = thr,
    and where D(N) = B (supply exactly met)."""
    B = 8 * banks
    ns = np.arange(1.0, 128.0, 0.01)
    D = (5 * ns * M + ns ** 2) / np.maximum(M, ns)
    u = (M / np.maximum(M, ns)) * np.minimum(1.0, B / D)
    n90 = ns[np.argmax(u < thr)] if (u < thr).any() else float("inf")
    nsat = ns[np.argmax(D > B)] if (D > B).any() else float("inf")
    return n90, nsat


def panel_title(ax, wl):
    ax.set_title(st.WORKLOAD_LABEL.get(wl, wl), loc="left")


# ------------------------------------------------------------- figure 1
def fig1_saturation(main):
    """PE utilization vs N, one line per BANKS, model dashed (2x2)."""
    banks = sorted(main.banks.unique())
    col, mk = st.ramp(banks), st.markers(banks)
    fig, axs = plt.subplots(2, 2, figsize=(TWO_COL, 5.0), sharex=True)
    for ax, wl in zip(axs.flat, ex.WORKLOADS):
        d = main[(main.workload == wl)].sort_values("N")
        for b in banks:
            s = d[d.banks == b]
            ax.plot(s.N, s.pe_util, color=col[b], marker=mk[b],
                    label=f"BANKS={b} (B={8 * b} bytes/cycle)", zorder=3)
            st.model_line(ax, s.N, s.pred_util, col[b])
        if wl == "llama7b_gemv":
            ax.set_yscale("log")
            ax.set_ylim(0.004, 0.2)
        else:
            ax.set_ylim(0, 1.05)
            st.threshold(ax, x=8)
        panel_title(ax, wl)
        ax.set_xticks(ex.N_VALUES)
    for ax in axs[1]:
        ax.set_xlabel("Array dimension N  (N x N PEs)")
    for ax in axs[:, 0]:
        ax.set_ylabel("PE utilization")
    h, l = axs[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=st.INK2, lw=1.0, ls=(0, (3, 2))))
    l.append("analytical model, eq. (1)-(2) + DRAM ceiling")
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Scratchpad saturation: PE utilization vs. array size",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_saturation_vs_N")


# ------------------------------------------------------------- figure 2
def fig2_universal(main, knee=None, idealdram=None):
    """Utilization normalized by its stall-free value vs load D/B.
    Left: weights streamed from DDR4 (main + knee sweeps).  Right: the
    main grid with weights free of DRAM, so only the scratchpad can limit
    utilization."""
    def prep(df):
        d = df[df.workload != "llama7b_gemv"].copy()
        d["load"] = d.model_D / d.B_bytes_per_cycle
        d["bw_frac"] = d.pe_util / d.ideal_util
        return d

    parts = [main] + ([knee] if knee is not None else [])
    real = prep(pd.concat(parts).drop_duplicates(["workload", "N", "banks"]))
    panels = [("Weights from DDR4-2400", real)]
    if idealdram is not None:
        panels.append(("Weights free of DRAM", prep(idealdram)))
    fig, axs = plt.subplots(
        1, len(panels), sharey=True,
        figsize=(TWO_COL if len(panels) > 1 else ONE_COL, 2.9))
    axs = np.atleast_1d(axs)
    x = np.logspace(np.log10(0.1), np.log10(6), 300)
    for ax, (title, d) in zip(axs, panels):
        ax.plot(x, np.minimum(1, 1 / x), color=st.INK2, lw=1.2,
                ls=(0, (3, 2)), label="model: min(1, B/D)", zorder=2)
        # Two groups are limited by something other than bank bandwidth
        # and get their own hollow markers: runs the model expects DRAM
        # to limit, and bank counts that divide N - 1 (lane aliasing).
        alias = aliasing(d)
        dram_lim = (d.pred_dram_ceiling < 0.999) & ~alias
        plain = ~alias & ~dram_lim
        for wl in [w for w in ex.WORKLOADS if w in set(d.workload)]:
            s_ = d[(d.workload == wl) & plain]
            ax.scatter(s_.load, s_.bw_frac, s=18,
                       color=st.WORKLOAD_COLOR[wl],
                       marker=st.WORKLOAD_MARKER[wl], edgecolor=st.SURFACE,
                       linewidth=0.5, label=st.WORKLOAD_SHORT[wl], zorder=3)
        for mask, mk_, lab in ((dram_lim, "o", "DRAM-limited (N = 32, "
                                "M = 128)"),
                               (alias, "x", "aliasing: BANKS divides "
                                "N - 1")):
            s_ = d[mask]
            if len(s_):
                # "x" is an unfilled marker: it takes one colour only.
                kw = (dict(facecolor="none", edgecolor=st.INK2)
                      if mk_ == "o" else dict(color=st.INK2))
                ax.scatter(s_.load, s_.bw_frac, s=20, marker=mk_,
                           linewidth=0.8, label=lab, zorder=4, **kw)
        ax.set_xscale("log")
        ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.axvline(1, color=st.MUTED, lw=0.8, ls=(0, (1, 2)))
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Load  D(N) / B   (demand, eq. 1 / supply, eq. 2)")
        ax.set_title(f"{title} ({len(d)} runs)", loc="left")
    axs[0].set_ylabel("Utilization / stall-free utilization")
    axs[0].legend(loc="lower left", fontsize=6.5)
    fig.suptitle("One saturation curve: utilization vs. demand / supply",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_universal_saturation")


# ------------------------------------------------------------- figure 3
def fig3_bandwidth(main):
    """Delivered array bandwidth vs N against demand D(N) and supply B."""
    banks = sorted(main.banks.unique())
    col, mk = st.ramp(banks), st.markers(banks)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 2.9), sharey=True)
    for ax, wl in zip(axs, ex.COMPUTE_BOUND):
        d = main[main.workload == wl].sort_values("N")
        dem = d[d.banks == banks[0]]
        ax.plot(dem.N, dem.model_D, color=st.INK, lw=1.2, ls=(0, (3, 2)),
                label="demand D(N), eq. (1)", zorder=2)
        for b in banks:
            s = d[d.banks == b]
            ax.axhline(8 * b, color=col[b], lw=0.7, ls=(0, (1, 2)),
                       zorder=1)
            ax.plot(s.N, s.delivered_array_bw, color=col[b], marker=mk[b],
                    label=f"delivered, BANKS={b}", zorder=3)
            st.end_label(ax, 32, 8 * b, f"B={8 * b}", dy=5)
        panel_title(ax, wl)
        ax.set_xticks(ex.N_VALUES)
        ax.set_xlim(6, 36)
        ax.set_xlabel("Array dimension N")
    axs[0].set_ylim(0, 300)
    axs[0].set_ylabel("Scratchpad bandwidth (bytes/cycle)")
    axs[0].legend(loc="upper left")
    fig.suptitle("Delivered bandwidth follows demand until it reaches "
                 "supply B = 8 x BANKS (dotted)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_delivered_bandwidth")


# ------------------------------------------------------------- figure 4
def fig4_banks(banks_df):
    """Utilization vs BANKS for each N: how many banks restore it."""
    ns = sorted(banks_df.N.unique())
    col, mk = st.ramp(ns), st.markers(ns)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 2.9), sharey=True)
    for ax, wl in zip(axs, ex.COMPUTE_BOUND):
        d = banks_df[banks_df.workload == wl].sort_values("banks")
        M = shape_of(d, wl)[0]
        for n in ns:
            s = d[d.N == n]
            ax.plot(s.banks, s.pe_util, color=col[n], marker=mk[n],
                    label=f"N={n}", zorder=3)
            st.model_line(ax, s.banks, s.pred_util, col[n])
            rule = math.ceil(model.demand(n, M) / 8)
            ax.plot([rule, rule], [1.02, 1.07], color=col[n], lw=1.6,
                    solid_capstyle="butt", zorder=3)
        ax.text(1.9, 1.075, "bank-matching rule  ceil(D/8)  per N",
                fontsize=6.5, color=st.MUTED, va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ex.FINE_BANKS)
        ax.set_xticklabels([str(b) for b in ex.FINE_BANKS])
        ax.set_ylim(0, 1.12)
        st.threshold(ax, x=2)
        panel_title(ax, wl)
        ax.set_xlabel("BANKS  (1 bank = 1 BRAM36)")
    axs[0].set_ylabel("PE utilization")
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=st.INK2, lw=1.0, ls=(0, (3, 2))))
    l.append("model")
    axs[0].legend(h, l, loc="lower right")
    fig.suptitle("Bank provisioning: banks needed to restore utilization "
                 "(B = 8 x BANKS bytes/cycle)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_bank_provisioning")


# ------------------------------------------------------------- figure 5
def fig5_reuse(reuse):
    """Utilization vs data reuse M for each N (synthetic GEMMs)."""
    ns = sorted(reuse.N.unique())
    col, mk = st.ramp(ns), st.markers(ns)
    bks = sorted(reuse.banks.unique())
    fig, axs = plt.subplots(1, len(bks), figsize=(TWO_COL, 2.9),
                            sharey=True)
    axs = np.atleast_1d(axs)
    for ax, b in zip(axs, bks):
        d = reuse[reuse.banks == b].sort_values("M")
        for n in ns:
            s = d[d.N == n]
            ax.plot(s.M, s.pe_util, color=col[n], marker=mk[n],
                    label=f"N={n}", zorder=3)
            st.model_line(ax, s.M, s.pred_util, col[n])
        ax.set_xscale("log", base=2)
        ax.set_xticks(ex.REUSE_M)
        ax.set_xticklabels([str(m) for m in ex.REUSE_M])
        ax.set_ylim(0, 1.05)
        st.threshold(ax, x=1)
        ax.set_title(f"BANKS={b}  (B={8 * b} bytes/cycle)", loc="left")
        ax.set_xlabel("Reuse M  (activation rows per weight tile)")
    axs[0].set_ylabel("PE utilization")
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=st.INK2, lw=1.0, ls=(0, (3, 2))))
    l.append("model")
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"Utilization vs. data reuse (GEMM M x {ex.REUSE_K} x "
                 f"{ex.REUSE_K}, weights streamed from DDR4)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_utilization_vs_reuse")


# ------------------------------------------------------------- figure 6
def fig6_gemv(main):
    """M = 1: utilization follows 1/N and DRAM, not BANKS."""
    d = main[main.workload == "llama7b_gemv"].sort_values("N")
    if d.empty:
        return
    banks = sorted(d.banks.unique())
    col, mk = st.ramp(banks), st.markers(banks)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 2.8))
    ax = axs[0]
    for b in banks:
        s = d[d.banks == b]
        ax.plot(s.N, s.pe_util, color=col[b], marker=mk[b], ms=7 - b / 8,
                label=f"BANKS={b}", zorder=3)
    n = np.array(ex.N_VALUES, float)
    ax.plot(n, 1 / n, color=st.INK2, lw=1.0, ls=(0, (1, 2)),
            label="reuse ceiling 1/N")
    s = d[d.banks == banks[0]]
    ax.plot(s.N, s.pred_util, color=st.INK2, lw=1.0, ls=(0, (3, 2)),
            label="model: 1/N x DRAM ceiling")
    ax.set_yscale("log")
    ax.set_xticks(ex.N_VALUES)
    ax.set_xlabel("Array dimension N")
    ax.set_ylabel("PE utilization (log)")
    ax.set_title("Utilization: identical for every BANKS", loc="left")
    ax.legend(loc="lower left")
    ax = axs[1]
    for b in banks:
        s = d[d.banks == b]
        ax.plot(s.N, s.dram_bus_util, color=col[b], marker=mk[b],
                ms=7 - b / 8, label=f"BANKS={b}", zorder=3)
    ax.set_ylim(0, 100)
    ax.set_xticks(ex.N_VALUES)
    ax.set_xlabel("Array dimension N")
    ax.set_ylabel("DDR4 data-bus utilization (%)")
    ax.set_title("DRAM is the bottleneck", loc="left")
    fig.suptitle("LLaMA-7B decode GEMV (M=1): bound by reuse and DRAM, "
                 "not the scratchpad", x=0.01, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_gemv_dram_bound")


# ------------------------------------------------------------- figure 7
def fig7_stalls(main):
    """Where the cycles go: active vs. stall causes (stacked)."""
    comps = [("useful (rows streaming)", "useful"),
             ("idle slots (fill/drain, M < N)", "idle"),
             ("spm read stall", "spm_read_stall_cycles"),
             ("spm write stall", "spm_write_stall_cycles"),
             ("DMA stall", "dma_stall_cycles")]
    wls = ex.WORKLOADS
    banks = sorted(main.banks.unique())
    fig, axs = plt.subplots(1, len(wls), figsize=(TWO_COL, 2.9),
                            sharey=True)
    for ax, wl in zip(axs, wls):
        d = main[main.workload == wl]
        xs, labels, x = [], [], 0
        for n in ex.N_VALUES:
            for b in banks:
                xs.append(x)
                labels.append((n, b))
                x += 1
            x += 0.8
        bottom = np.zeros(len(xs))
        for name, colname in comps:
            vals = []
            for n, b in labels:
                r = d[(d.N == n) & (d.banks == b)]
                if r.empty:
                    vals.append(0)
                    continue
                r = r.iloc[0]
                if colname == "useful":
                    v = r.pe_util * r.total_cycles
                elif colname == "idle":
                    v = (r.total_cycles - r.spm_stall_cycles -
                         r.dma_stall_cycles - r.pe_util * r.total_cycles)
                else:
                    v = r[colname]
                vals.append(v / r.total_cycles)
            vals = np.array(vals)
            ax.bar(xs, vals, bottom=bottom, width=0.9,
                   color=st.COMPONENT_COLOR[name], edgecolor=st.SURFACE,
                   linewidth=0.5, label=name)
            bottom += vals
        centers = [np.mean(xs[i * len(banks):(i + 1) * len(banks)])
                   for i in range(len(ex.N_VALUES))]
        ax.set_xticks(centers)
        ax.set_xticklabels([f"N={n}" for n in ex.N_VALUES], fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
        ax.set_title(st.WORKLOAD_SHORT[wl], loc="left")
    axs[0].set_ylabel("Fraction of cycles")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"Cycle breakdown; bars in each group are BANKS = "
                 f"{', '.join(map(str, banks))} (left to right)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_stall_breakdown")


# ------------------------------------------------------------- figure 8
def fig8_double_buffer(main, nodb):
    b = int(nodb.banks.iloc[0])
    on = main[main.banks == b]
    fig, axs = plt.subplots(1, 4, figsize=(TWO_COL, 2.6))
    w = 0.38
    for ax, wl in zip(axs, ex.WORKLOADS):
        x = np.arange(len(ex.N_VALUES))
        a = on[on.workload == wl].set_index("N").reindex(ex.N_VALUES)
        c = nodb[nodb.workload == wl].set_index("N").reindex(ex.N_VALUES)
        ax.bar(x - w / 2 - 0.01, a.pe_util, w, color="#2a78d6",
               label="double-buffered weights", edgecolor=st.SURFACE,
               linewidth=0.5)
        ax.bar(x + w / 2 + 0.01, c.pe_util, w, color="#eb6834",
               label="single-buffered", edgecolor=st.SURFACE, linewidth=0.5)
        pm = [model.predict(n, b, *shape_of(nodb, wl), False)["pred_util"]
              for n in ex.N_VALUES]
        ax.scatter(x + w / 2 + 0.01, pm, marker="_", s=90, color=st.INK,
                   linewidth=1.4, zorder=4,
                   label="model, single-buffered")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ex.N_VALUES])
        ax.set_xlabel("N")
        ax.grid(axis="x", visible=False)
        ax.set_title(st.WORKLOAD_SHORT[wl], loc="left")
        if wl == "llama7b_gemv":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.05)
    axs[0].set_ylabel(f"PE utilization (BANKS={b})")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Double buffering: hiding the N-cycle weight load "
                 "(GEMV panel on log scale)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig8_double_buffering")


# ------------------------------------------------------------- figure 9
def fig9_arbitration(main, lane):
    j = lane.merge(main, on=["workload", "N", "banks"],
                   suffixes=("_lane", "_bank"))
    banks = sorted(j.banks.unique())
    col, mk = st.ramp(banks), st.markers(banks)
    fig, ax = plt.subplots(figsize=(ONE_COL, 3.0))
    ax.plot([0, 1], [0, 1], color=st.MUTED, lw=0.8, ls=(0, (1, 2)))
    for b in banks:
        s = j[j.banks == b]
        ax.scatter(s.pe_util_bank, s.pe_util_lane, color=col[b],
                   marker=mk[b], s=26, edgecolor=st.SURFACE, linewidth=0.6,
                   label=f"BANKS={b}", zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Utilization, per-bank queues (eq. 2 supply)")
    ax.set_ylabel("Utilization, per-lane ports")
    ax.set_title("Arbitration cost (below the diagonal = lost)",
                 loc="left")
    ax.legend(loc="upper left", title="ResNet-50 + BERT FFN1",
              title_fontsize=7)
    save(fig, "fig9_arbitration_lane_vs_bank")


# ------------------------------------------------------------ figure 10
def fig10_amat(main):
    d = main[main.workload != "llama7b_gemv"].copy()
    d["load"] = d.model_D / d.B_bytes_per_cycle
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.8))
    for wl in [w for w in ex.WORKLOADS if w in set(d.workload)]:
        s = d[d.workload == wl]
        ax.scatter(s.load, s.avg_spm_latency, s=22,
                   color=st.WORKLOAD_COLOR[wl],
                   marker=st.WORKLOAD_MARKER[wl], edgecolor=st.SURFACE,
                   linewidth=0.6, label=st.WORKLOAD_SHORT[wl], zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.axvline(1, color=st.MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.set_xlabel("Load  D(N) / B")
    ax.set_ylabel("Scratchpad AMAT (cycles)")
    ax.set_title("Read latency grows with bank load", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig10_spm_amat_vs_load")


# ------------------------------------------------------------ figure 11
def fig11_parity(main):
    fig, ax = plt.subplots(figsize=(ONE_COL, 3.0))
    x = np.linspace(0, 1.05, 10)
    ax.fill_between(x, x * 0.9, np.minimum(1.1, x * 1.1), color=st.GRID,
                    alpha=0.6, lw=0, label="+/-10%")
    ax.plot(x, x, color=st.MUTED, lw=0.8)
    for wl in ex.WORKLOADS:
        s = main[main.workload == wl]
        ax.scatter(s.pred_util, s.pe_util, s=22, color=st.WORKLOAD_COLOR[wl],
                   marker=st.WORKLOAD_MARKER[wl], edgecolor=st.SURFACE,
                   linewidth=0.6, label=st.WORKLOAD_SHORT[wl], zorder=3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Predicted utilization (model.py)")
    ax.set_ylabel("Measured utilization (gem5)")
    ax.set_title("Model vs. gem5, all 64 sweep points", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig11_model_vs_gem5")


# ------------------------------------------------------------ figure 12
def fig12_rtl(main):
    p = os.path.join(DATA, "rtl_cycles.csv")
    if not os.path.exists(p):
        print("  fig12 skipped (no data/rtl_cycles.csv yet)")
        return
    rtl = pd.read_csv(p)
    runs = []
    for suite in ("rtlval", "main"):
        s = load(suite)
        if s is not None:
            runs.append(s)
    g = pd.concat(runs).drop_duplicates(["workload", "N", "banks"])
    j = rtl.merge(g, on=["workload", "N", "banks"])
    if j.empty:
        print("  fig12 skipped (no matching gem5 runs for rtl_cycles.csv)")
        return
    j["err_pct"] = 100 * (j.total_cycles - j.cycles) / j.cycles
    fig, ax = plt.subplots(figsize=(ONE_COL, 3.0))
    lo, hi = j.cycles.min() * 0.8, j.cycles.max() * 1.25
    x = np.logspace(np.log10(lo), np.log10(hi), 20)
    ax.fill_between(x, x * 0.95, x * 1.05, color=st.GRID, lw=0,
                    label="+/-5% target")
    ax.plot(x, x, color=st.MUTED, lw=0.8)
    for wl in sorted(set(j.workload)):
        s = j[j.workload == wl]
        base = wl.replace("_rtl", "")
        ax.scatter(s.cycles, s.total_cycles, s=22,
                   color=st.WORKLOAD_COLOR.get(base, st.INK2),
                   marker=st.WORKLOAD_MARKER.get(base, "o"),
                   edgecolor=st.SURFACE, linewidth=0.6, label=wl, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RTL cycles (Questa)")
    ax.set_ylabel("gem5 cycles")
    ax.set_title("gem5 vs. RTL cycle counts", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig12_gem5_vs_rtl")
    write_table(j[["workload", "N", "banks", "cycles", "total_cycles",
                   "err_pct"]].rename(columns={"cycles": "rtl_cycles",
                                               "total_cycles": "gem5_cycles"}),
                "rtl_crossvalidation", "gem5 vs. RTL cycle counts",
                "Target: |err| <= 5% (proposal Sec. V, phase 2).")


# ------------------------------------------------------------ figure 13
def fig13_edp(main):
    """Proposal Sec. V: EDP = (DRAM energy + array energy) x layer time.
    Array energy = Vivado power x active time (cycles / Fmax); until
    data/vivado.csv exists only the DRAM part is plotted."""
    j = main.copy()
    p = os.path.join(DATA, "vivado.csv")
    have_viv = os.path.exists(p)
    if have_viv:
        j = j.merge(pd.read_csv(p), on="N")
        j["array_energy_J"] = (j.power_W * j.total_cycles /
                               (j.fmax_MHz * 1e6))
    else:
        j["array_energy_J"] = 0.0
    j["edp_Js"] = (j.array_energy_J + j.dram_energy_J) * j.layer_time_s
    banks = sorted(j.banks.unique())
    col, mk = st.ramp(banks), st.markers(banks)
    fig, axs = plt.subplots(2, 2, figsize=(TWO_COL, 4.8))
    for ax, wl in zip(axs.flat, ex.WORKLOADS):
        d = j[j.workload == wl].sort_values("N")
        for b in banks:
            s = d[d.banks == b]
            ax.plot(s.N, s.edp_Js, color=col[b], marker=mk[b],
                    label=f"BANKS={b}")
        if d.edp_Js.max() > 10 * d.edp_Js.min():
            ax.set_yscale("log")
        else:  # narrow range (e.g. DRAM-bound GEMV): linear from 0
            ax.set_ylim(0, d.edp_Js.max() * 1.15)
        ax.set_xticks(ex.N_VALUES)
        panel_title(ax, wl)
    for ax in axs[1]:
        ax.set_xlabel("Array dimension N")
    for ax in axs[:, 0]:
        ax.set_ylabel("EDP (J x s)")
    axs[0, 0].legend()
    what = ("DRAM + array energy" if have_viv else
            "DRAM energy only (array energy needs data/vivado.csv)")
    fig.suptitle(f"Energy-delay product per layer: {what}", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig13_edp")
    cols = ["workload", "N", "banks", "total_cycles", "layer_time_s",
            "dram_energy_J", "array_energy_J", "edp_Js"]
    if have_viv:
        cols[4:4] = ["fmax_MHz", "power_W"]
    write_table(j[cols].sort_values(["workload", "N", "banks"]), "edp",
                "Energy-delay product", f"Energy terms: {what}.")


RULE_COLOR = {"ceil(D/8)": "#2a78d6", "pow2": "#eb6834",
              "headroom": "#1baf7a"}
RULE_LABEL = {"ceil(D/8)": "ceil(D/8) banks (exact, modulo mapping)",
              "pow2": "rounded up to a power of two",
              "headroom": "ceil(D / (0.8 x 8)): 25% headroom"}


def fig14_bank_rule(br):
    """H3: does BANKS = ceil(D/8) restore utilization, and what does the
    partial-sum write buffer depth change?"""
    br = br.copy()
    br["pq"] = br.psum_queue.fillna(8).astype(int)
    pqs = sorted(br.pq.unique())
    fig, axs = plt.subplots(len(ex.COMPUTE_BOUND), len(pqs),
                            figsize=(TWO_COL, 4.8), sharey=True)
    axs = np.atleast_2d(axs)
    w = 0.26
    for r, wl in enumerate(ex.COMPUTE_BOUND):
        M = ex.M_OF[wl]
        for c, pq in enumerate(pqs):
            ax = axs[r, c]
            d = br[(br.workload == wl) & (br.pq == pq)]
            for i, n in enumerate(ex.N_VALUES):
                for k, (lab, b) in enumerate(ex.bank_rule_counts(n, M)):
                    row = d[(d.N == n) & (d.banks == b)]
                    if row.empty:
                        continue
                    u = float(row.pe_util.iloc[0])
                    x = i + (k - 1) * w
                    ax.bar(x, u, w * 0.92, color=RULE_COLOR[lab],
                           edgecolor=st.SURFACE, linewidth=0.5,
                           label=RULE_LABEL[lab] if i == 0 else None)
                    ax.text(x, u + 0.015, str(b), ha="center", va="bottom",
                            fontsize=6, color=st.INK2)
            ax.set_xticks(range(len(ex.N_VALUES)))
            ax.set_xticklabels([f"N={n}" for n in ex.N_VALUES])
            ax.set_ylim(0, 1.12)
            ax.grid(axis="x", visible=False)
            st.threshold(ax, x=-0.45)
            ax.set_title(f"{st.WORKLOAD_SHORT[wl]}, write buffer {pq} "
                         f"entries/lane", loc="left")
        axs[r, 0].set_ylabel("PE utilization")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Bank-matching rule (numbers above bars = BANKS)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig14_bank_rule")


PHASE_COLOR = {"A load (DMA in)": "#2a78d6", "compute": "#eb6834",
               "C store (DMA out)": "#1baf7a"}


def fig15_layer_phases(main, banks=32):
    """Host-driven layer time split into A load / compute / C store."""
    d0 = main[main.banks == banks]
    fig, axs = plt.subplots(1, 4, figsize=(TWO_COL, 2.8))
    for ax, wl in zip(axs, ex.WORKLOADS):
        d = d0[d0.workload == wl].set_index("N").reindex(ex.N_VALUES)
        x = np.arange(len(ex.N_VALUES))
        bottom = np.zeros(len(x))
        parts = [("A load (DMA in)", d.load_cycles),
                 ("compute", d.total_cycles),
                 ("C store (DMA out)", d.store_cycles)]
        tot = d.layer_cycles.values
        for name, v in parts:
            frac = v.values / tot
            ax.bar(x, frac, 0.7, bottom=bottom, color=PHASE_COLOR[name],
                   edgecolor=st.SURFACE, linewidth=0.5, label=name)
            bottom += frac
        for xi, t in zip(x, d.layer_time_s.values):
            ax.text(xi, 1.02, f"{t * 1e6:.0f}" if t >= 1e-5 else
                    f"{t * 1e6:.1f}", ha="center", va="bottom", fontsize=6,
                    color=st.INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ex.N_VALUES])
        ax.set_xlabel("N")
        ax.set_ylim(0, 1.12)
        ax.grid(axis="x", visible=False)
        ax.set_title(st.WORKLOAD_SHORT[wl], loc="left")
    axs[0].set_ylabel("Fraction of layer time")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"Host-driven layer: where the time goes (BANKS={banks}; "
                 "numbers = layer time in us)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig15_layer_phases")


def speedup_frame(main, sw):
    t = sw.set_index("workload").layer_time_s
    d = main[main.workload.isin(t.index)].copy()
    d["sw_time_s"] = d.workload.map(t)
    d["speedup"] = d.sw_time_s / d.layer_time_s
    return d


def fig16_speedup(main, sw):
    d = speedup_frame(main, sw)
    if d.empty:
        print("  fig16 skipped (no matching sw runs)")
        return
    banks = sorted(d.banks.unique())
    col, mk = st.ramp(banks), st.markers(banks)
    fig, axs = plt.subplots(1, 4, figsize=(TWO_COL, 2.8), sharey=True)
    for ax, wl in zip(axs, ex.WORKLOADS):
        s0 = d[d.workload == wl].sort_values("N")
        if s0.empty:
            ax.text(0.5, 0.5, "no software\nbaseline yet", ha="center",
                    va="center", transform=ax.transAxes, color=st.MUTED)
        for b in banks:
            s = s0[s0.banks == b]
            ax.plot(s.N, s.speedup, color=col[b], marker=mk[b],
                    label=f"BANKS={b}")
        ax.set_yscale("log")
        ax.set_xticks(ex.N_VALUES)
        ax.set_xlabel("N")
        ax.set_title(st.WORKLOAD_SHORT[wl], loc="left")
    axs[0].set_ylabel("Speedup over RISC-V software")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Layer speedup vs. the same GEMM in software on the "
                 "RiscvTimingSimpleCPU host (2 GHz)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig16_speedup")


# ------------------------------------------------------------ figure 17
def fig17_knee(knee):
    """BANKS swept one bank at a time across the knee (H1 and H3)."""
    ns = sorted(knee.N.unique())
    col, mk = st.ramp(ns), st.markers(ns)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 3.0), sharey=True)
    for ax, wl in zip(axs, ex.COMPUTE_BOUND):
        d = knee[knee.workload == wl]
        M = ex.M_OF[wl]
        for n in ns:
            s = d[d.N == n].sort_values("banks")
            ax.plot(s.banks, s.pe_util, color=col[n], marker=mk[n], ms=3.5,
                    lw=1.3, label=f"N={n}", zorder=3)
            st.model_line(ax, s.banks, s.pred_util, col[n])
            rule = math.ceil(model.demand(n, M) / 8)
            ax.plot([rule, rule], [1.03, 1.09], color=col[n], lw=1.8,
                    solid_capstyle="butt", zorder=3)
        ax.text(0.4, 1.10, "ceil(D/8) for each N", fontsize=6.5,
                color=st.MUTED, va="bottom")
        ax.text(32.6, 0.05, "dips at BANKS = N - 1:\nevery partial-sum "
                "lane hits the same bank", fontsize=6.5, color=st.MUTED,
                ha="right", va="bottom")
        ax.set_xlim(0, 33)
        ax.set_xticks(range(0, 33, 4))
        ax.set_ylim(0, 1.17)
        st.threshold(ax, x=0.3)
        panel_title(ax, wl)
        ax.set_xlabel("BANKS  (B = 8 x BANKS bytes/cycle)")
    axs[0].set_ylabel("PE utilization")
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=st.INK2, lw=1.0, ls=(0, (3, 2))))
    l.append("model")
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Across the knee, one bank at a time", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig17_knee_bank_by_bank")


# ------------------------------------------------------------ figure 18
def gemv_frame(main, clock, dram):
    """All M = 1 runs with the DRAM supply per array cycle.

    The sustained bandwidth of each DRAM type is taken from ONE run of
    that type (N = 32, 2 GHz, 8 banks: the most DRAM-bound).  Every other
    run of that type, at any N or array clock, is then a prediction."""
    parts = [main[main.workload == "llama7b_gemv"]]
    for df in (clock, dram):
        if df is not None:
            parts.append(df[df.workload == "llama7b_gemv"])
    g = pd.concat(parts).drop_duplicates(
        ["N", "banks", "clock_hz", "mem_type"]).copy()
    g["calibration"] = (g.N == 32) & (g.clock_hz == 2e9) & (g.banks == 8)
    cal = g[g.calibration].set_index("mem_type").dram_read_bw_GBs * 1e9
    g["sustained_GBs"] = g.mem_type.map(cal) / 1e9
    g["supply"] = g.mem_type.map(cal) / g.clock_hz   # bytes / array cycle
    g["ratio"] = g.supply / g.N                # need = N bytes/cycle (M=1)
    g["cap_frac"] = g.pe_util / g.ideal_util   # fraction of the 1/N cap
    # Scratchpad side at M = 1: N + 5 (eq. 1) plus N more for the DMA
    # writing each tile into the scratchpad.
    g["spm_ratio"] = 8 * g.banks / (2 * g.N + 5)
    g["model_frac"] = np.minimum(1.0, np.minimum(g.ratio, g.spm_ratio))
    return g


def clock_label(hz):
    return f"{hz / 1e9:g} GHz" if hz >= 1e9 else f"{hz / 1e6:g} MHz"


def fig18_gemv_dram(main, clock, dram):
    """H2: M = 1 utilization follows the DRAM supply up to the 1/N cap."""
    g = gemv_frame(main, clock, dram)
    fig, axs = plt.subplots(1, 3, figsize=(TWO_COL, 3.5))

    # (a) utilization vs N for each array clock (DDR4, 8 banks)
    ax = axs[0]
    a = g[(g.banks == 8) & (g.mem_type == "DDR4_2400_8x8")]
    clocks = sorted(a.clock_hz.unique())
    col, mk = st.ramp(clocks), st.markers(clocks)
    for c in clocks:
        s = a[a.clock_hz == c].sort_values("N")
        ax.plot(s.N, s.pe_util, color=col[c], marker=mk[c],
                label=clock_label(c), zorder=3)
    n = np.array(sorted(a.N.unique()), float)
    ax.plot(n, 1 / n, color=st.INK2, lw=1.0, ls=(0, (1, 2)),
            label="reuse cap 1/N")
    ax.set_yscale("log")
    ax.set_xticks(ex.N_VALUES)
    ax.set_xlabel("Array dimension N")
    ax.set_ylabel("PE utilization (log)")
    ax.set_title("(a) array clock, DDR4", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              fontsize=6.5)

    # (b) collapse: fraction of the 1/N cap vs DRAM supply / need
    ax = axs[1]
    b = g[(g.banks >= 8) & (g.spm_ratio * 0.9 >= g.cap_frac)] \
        .sort_values("banks").drop_duplicates(
            ["N", "clock_hz", "mem_type"], keep="last")
    ns = sorted(b.N.unique())
    coln = st.ramp(ns)
    x = np.logspace(np.log10(0.15), np.log10(12), 200)
    ax.plot(x, np.minimum(1, x), color=st.INK2, lw=1.2, ls=(0, (3, 2)),
            label="model: min(1, x)", zorder=2)
    for nn in ns:
        for ddr4, m in ((True, "o"), (False, "s")):
            s = b[(b.N == nn) & ((b.mem_type == "DDR4_2400_8x8") == ddr4)]
            ax.scatter(s.ratio, s.cap_frac, s=20, color=coln[nn], marker=m,
                       edgecolor=st.SURFACE, linewidth=0.5, zorder=3,
                       label=f"N={nn}" if ddr4 else None)
    ax.scatter([], [], s=20, color=st.MUTED, marker="s",
               label="other DRAM types")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["1/4", "1/2", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("DRAM supply / need")
    ax.set_ylabel("Utilization / (1/N cap)")
    ax.set_title("(b) all clocks, DRAM types", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              fontsize=6.5)

    # (c) BANKS only matters below B = D = N + 5 (250 MHz: DRAM not binding)
    ax = axs[2]
    c = g[(g.clock_hz == 250e6) & (g.mem_type == "DDR4_2400_8x8")]
    mkn = st.markers(ns)
    for nn in ns:
        s = c[c.N == nn].sort_values("banks")
        ax.plot(s.banks, s.cap_frac, color=coln[nn], marker=mkn[nn],
                label=f"N={nn}", zorder=3)
        need = (2 * nn + 5) / 8
        ax.plot([need, need], [1.04, 1.09], color=coln[nn], lw=1.8,
                solid_capstyle="butt")
    ax.text(0.9, 1.10, "BANKS = (2N + 5) / 8", fontsize=6.5, color=st.MUTED,
            va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_ylim(0, 1.17)
    ax.set_xlabel("BANKS")
    ax.set_ylabel("Utilization / (1/N cap)")
    ax.set_title("(c) BANKS at 250 MHz", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              fontsize=6.5)
    fig.suptitle("M = 1 (LLaMA GEMV): utilization follows DRAM supply up "
                 "to the 1/N cap, not BANKS", x=0.01, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig18_gemv_dram_supply")


# ------------------------------------------------------------ figure 19
SENS_ORDER = ["", "la4", "la16", "lat2", "lat4", "pq16", "pq32", "wb4KiB",
              "wb64KiB", "dma32", "dma128"]
SENS_LABEL = {"": "default", "la4": "lookahead 4", "la16": "lookahead 16",
              "lat2": "bank latency 2", "lat4": "bank latency 4",
              "pq16": "write buffer 16", "pq32": "write buffer 32",
              "wb4KiB": "weight buffer 4 KiB",
              "wb64KiB": "weight buffer 64 KiB", "dma32": "DMA depth 32",
              "dma128": "DMA depth 128"}


def sens_frame(sens):
    d = sens.copy()
    d["variant"] = [t.split(f"_B{b}", 1)[1].lstrip("_")
                    for t, b in zip(d.tag, d.banks)]
    return d


def fig19_sensitivity(sens):
    """One-at-a-time design-knob changes barely move utilization."""
    d = sens_frame(sens)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 3.6), sharey=True)
    for ax, (n, bank_list) in zip(axs, ex.SENS_POINTS):
        col, mk = st.ramp(bank_list), st.markers(bank_list)
        M = ex.M_OF["resnet50_conv3x"]
        for b in bank_list:
            s = d[(d.N == n) & (d.banks == b)].set_index("variant") \
                .reindex(SENS_ORDER)
            load = model.demand(n, M) / (8 * b)
            ax.plot(range(len(SENS_ORDER)), s.pe_util, color=col[b],
                    marker=mk[b], label=f"BANKS={b}  (D/B = {load:.2f})",
                    zorder=3)
        ax.set_xticks(range(len(SENS_ORDER)))
        ax.set_xticklabels([SENS_LABEL[v] for v in SENS_ORDER], fontsize=6.5,
                           rotation=40, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"ResNet-50, N={n}", loc="left")
        ax.legend(loc="lower left")
    axs[0].set_ylabel("PE utilization")
    fig.suptitle("Sensitivity: one design knob changed at a time",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig19_sensitivity")


# ------------------------------------------------------------ figure 20
def rulem_frame(rulem):
    d = rulem.copy()
    d["frac"] = d.pe_util / d.ideal_util
    d["kind"] = ["rule" if b == ex.rule_m_banks(n, m)[0] else "fewer"
                 for n, m, b in zip(d.N, d.M, d.banks)]
    return d


def fig20_rule_vs_m(rulem):
    """H3 at every reuse M: banks = ceil(D(N, M)/8) vs 25% fewer."""
    d = rulem_frame(rulem)
    ns = sorted(d.N.unique())
    col, mk = st.ramp(ns), st.markers(ns)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 2.9), sharey=True)
    for ax, kind, title in zip(
            axs, ("rule", "fewer"),
            ("BANKS = ceil(D(N, M) / 8)", "25% fewer banks than the rule")):
        k = d[d.kind == kind]
        for n in ns:
            s = k[k.N == n].sort_values("M")
            ax.plot(s.M, s.frac, color=col[n], marker=mk[n],
                    label=f"N={n}", zorder=3)
            if kind == "fewer":
                st.model_line(ax, s.M, s.pred_util / s.ideal_util, col[n])
        ax.set_xscale("log", base=2)
        ax.set_xticks(ex.RULE_M)
        ax.set_xticklabels([str(m) for m in ex.RULE_M])
        ax.set_ylim(0, 1.08)
        st.threshold(ax, x=ex.RULE_M[0],
                     text="90%" if kind == "fewer" else None)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Reuse M  (activation rows per weight tile)")
    axs[0].set_ylabel("Utilization / stall-free utilization")
    h, l = axs[1].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=st.INK2, lw=1.0, ls=(0, (3, 2))))
    l.append("model")
    axs[1].legend(h, l, loc="lower right")
    fig.suptitle("The bank-matching rule at every reuse M (weights free of "
                 "DRAM)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig20_rule_vs_reuse")


# ------------------------------------------------------------ figure 21
def extended_frame(main, banks_df, extend):
    wl = "resnet50_conv3x"
    parts = [main[main.workload == wl], extend]
    if banks_df is not None:
        parts.append(banks_df[banks_df.workload == wl])
    d = pd.concat(parts).drop_duplicates(["N", "banks"])
    return d[d.banks.isin([16, 32, 64])]


def fig21_extended(main, banks_df, extend):
    """Beyond the proposal's range: does 32 banks saturate near N = 51?"""
    d = extended_frame(main, banks_df, extend)
    banks = sorted(d.banks.unique())
    col, mk = st.ramp(banks), st.markers(banks)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axvspan(32, 66, color=st.GRID, alpha=0.45, lw=0, zorder=0)
    ax.text(33, 0.04, "beyond the proposal's 8-32 range", fontsize=6.5,
            color=st.MUTED)
    for b in banks:
        s = d[d.banks == b].sort_values("N")
        ax.plot(s.N, s.pe_util, color=col[b], marker=mk[b],
                label=f"BANKS={b}", zorder=3)
        st.model_line(ax, s.N, s.pred_util, col[b])
        nsat = model_n90(ex.M_OF["resnet50_conv3x"], b)[1]
        if nsat < 66:
            ax.plot([nsat, nsat], [1.03, 1.09], color=col[b], lw=1.8,
                    solid_capstyle="butt")
    ax.text(8, 1.10, "eq. (1): N where D = B", fontsize=6.5, color=st.MUTED,
            va="bottom")
    ax.set_xticks(ex.N_VALUES + ex.N_EXTENDED)
    ax.set_xlim(6, 66)
    ax.set_ylim(0, 1.17)
    st.threshold(ax, x=6.5)
    ax.set_xlabel("Array dimension N")
    ax.set_ylabel("PE utilization")
    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=st.INK2, lw=1.0, ls=(0, (3, 2))))
    l.append("model")
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    ax.set_title("ResNet-50: the model predicts saturation past N = 32",
                 loc="left")
    save(fig, "fig21_extended_N")


# ------------------------------------------------------------ figure 22
POLICY_COLOR = {"oldest first, 8-entry buffer": "#2a78d6",
                "reads first, 8-entry buffer": "#eb6834",
                "reads first, 16-entry buffer": "#1baf7a"}


def policy_frame(policy):
    d = policy.copy()
    d["pq"] = d.psum_queue.fillna(8).astype(int)
    d["la"] = d.lookahead.fillna(8).astype(int)
    return d


def fig22_bank_policy(policy):
    """BANKS = N - 1: the dip is arbitration and buffering, not bandwidth."""
    d = policy_frame(policy)
    d = d[d.la == 8]
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 2.9), sharey=True)
    w = 0.26
    variants = [("oldest first, 8-entry buffer", "fifo", 8),
                ("reads first, 8-entry buffer", "reads_first", 8),
                ("reads first, 16-entry buffer", "reads_first", 16)]
    for ax, wl in zip(axs, ex.COMPUTE_BOUND):
        for i, n in enumerate(ex.N_VALUES):
            for k, (lab, pol, pq) in enumerate(variants):
                r = d[(d.workload == wl) & (d.N == n) & (d.banks == n - 1)
                      & (d.bank_policy == pol) & (d.pq == pq)]
                if r.empty:
                    continue
                ax.bar(i + (k - 1) * w, float(r.pe_util.iloc[0]), w * 0.92,
                       color=POLICY_COLOR[lab], edgecolor=st.SURFACE,
                       linewidth=0.5, label=lab if i == 0 else None)
            r = d[(d.workload == wl) & (d.N == n) & (d.banks == n - 1)]
            if len(r):
                top = float(r.pred_util.iloc[0])
                ax.plot([i - 1.5 * w, i + 1.5 * w], [top, top],
                        color=st.INK, lw=1.2, zorder=4,
                        label="model (no aliasing)" if i == 0 else None)
        ax.set_xticks(range(len(ex.N_VALUES)))
        ax.set_xticklabels([f"N={n}\n{n - 1} banks" for n in ex.N_VALUES])
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
        panel_title(ax, wl)
    axs[0].set_ylabel("PE utilization")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Bank aliasing at BANKS = N - 1 and how to remove it",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig22_bank_policy")


def fit_row(name, df):
    """Goodness of fit of model.py's prediction for one set of runs, and
    of eq. (1) exactly as written in the proposal (no DMA-fill term)."""
    df = df.dropna(subset=["pe_util", "pred_util"])
    ss_tot = float(((df.pe_util - df.pe_util.mean()) ** 2).sum())

    def r2(col):
        e = df.pe_util - df[col]
        return 1 - float((e ** 2).sum()) / ss_tot if ss_tot else float("nan")
    e = df.pe_util - df.pred_util
    e1 = df.pe_util - df.pred_util_eq1
    return {"suite": name, "runs": len(df), "R^2": r2("pred_util"),
            "RMSE": float(np.sqrt((e ** 2).mean())),
            "mean |err|": float(e.abs().mean()),
            "max |err|": float(e.abs().max()),
            "R^2, eq. (1) as written": r2("pred_util_eq1"),
            "mean |err|, eq. (1) as written": float(e1.abs().mean())}


def aliasing(df):
    """True where BANKS divides N - 1 (every psum lane on one bank)."""
    return (df.banks > 1) & ((df.N - 1) % df.banks == 0)


def spm_bram36(nbytes, banks):
    """BRAM36 blocks for a banked scratchpad: each bank is 64 data bits
    wide and a BRAM36 in 72-bit mode is 512 words deep (4 KiB of data)."""
    words_per_bank = math.ceil(nbytes / 8 / banks)
    return banks * math.ceil(words_per_bank / 512)


def spm_bytes(M, K, NOUT, N, weight_buffer=16 * 1024):
    a64 = lambda x: (x + 63) // 64 * 64  # noqa: E731
    nkb, nnb = -(-K // N), -(-NOUT // N)
    slots = max(2, weight_buffer // (N * N))
    return a64(a64(nkb * M * N) + slots * N * N) + nnb * M * N * 4


# ----------------------------------------------------------------- tables
def tables(S):
    """Every table, tables/hypotheses.md and data/key_results.json.
    S maps a suite name to its DataFrame (or None if not collected)."""
    main, banks_df, ideal = S["main"], S["banks"], S["ideal"]
    bankrule, sw, standalone = S["bankrule"], S["sw"], S["standalone"]
    knee, idealdram, extend = S["knee"], S["idealdram"], S["extend"]
    sens, clock, dram = S["sens"], S["clock"], S["dram"]
    rulem, lane, nodb, reuse = S["rulem"], S["lane"], S["nodb"], S["reuse"]
    key = {}     # headline numbers -> data/key_results.json
    # Table I of the proposal (predicted scaling, large M)
    t1 = pd.DataFrame([{
        "N": n, "PEs": n * n, "edge demand 5N (bytes/cycle)": 5 * n,
        "peak demand 6N at M=N": 6 * n,
        "banks needed ceil(5N/8)": math.ceil(5 * n / 8),
        "fill/drain 2N-2": 2 * n - 2, "skew regs N(N-1)/2": n * (n - 1) // 2,
        "DSP48E2 (of 1728)": f"{n * n} ({100 * n * n / 1728:.0f}%)",
    } for n in ex.N_VALUES])
    write_table(t1, "table1_predicted_scaling",
                "Table I: predicted scaling (INT8 weight-stationary, M >> N)")

    # Research question: at what N does utilization drop below 90%?
    rows = []
    for wl in ex.WORKLOADS:
        d = main[main.workload == wl]
        if d.empty:
            continue
        M = shape_of(d, wl)[0]
        for b in sorted(d.banks.unique()):
            s = d[d.banks == b].sort_values("N")
            n90, nsat = model_n90(M, b)
            rows.append({
                "workload": wl, "M": M, "BANKS": b, "B (bytes/cycle)": 8 * b,
                **{f"util N={n}": u for n, u in zip(s.N, s.pe_util)},
                "N at 90% (gem5, interp.)": crossing(s.N, s.pe_util),
                "N at 90% (eq. 1-2)": "never" if math.isinf(n90)
                else f"{n90:.1f}",
                "N where D=B (eq. 1)": "never" if math.isinf(nsat)
                else f"{nsat:.1f}",
            })
    write_table(pd.DataFrame(rows), "saturation_points",
                "Where does the scratchpad saturate?",
                "gem5 column: first N (linearly interpolated between swept "
                "points) where measured PE utilization falls below 0.90. "
                "Model columns treat N as continuous and use only eq. (1)-(2) "
                "(reuse and scratchpad; no fill/drain, padding or DRAM), so "
                "they answer the research question analytically.  For M=1 "
                "the reuse ceiling 1/N is already below 90% at N >= 2.")

    # Bank-matching rule
    if banks_df is not None:
        rows = []
        for wl in ex.COMPUTE_BOUND:
            d = banks_df[banks_df.workload == wl]
            M = shape_of(d, wl)[0]
            for n in ex.N_VALUES:
                s = d[d.N == n].sort_values("banks")
                D = model.demand(n, M)
                rule = math.ceil(D / 8)
                p2 = ex.next_pow2(rule)
                at = s[s.banks == p2]
                ok = s[s.pe_util >= 0.9]
                rows.append({
                    "workload": wl, "N": n, "D(N)": round(D, 1),
                    "ceil(D/8)": rule, "pow2 banks": p2,
                    "util at pow2 banks": float(at.pe_util.iloc[0])
                    if len(at) else float("nan"),
                    "util at 2x pow2": float(s[s.banks == 2 * p2]
                                             .pe_util.iloc[0])
                    if len(s[s.banks == 2 * p2]) else float("nan"),
                    "min BANKS for >=90% (gem5)":
                        int(ok.banks.min()) if len(ok) else "none <= 64",
                    "BRAM36 (1 per bank)": int(ok.banks.min())
                    if len(ok) else "-",
                })
        write_table(pd.DataFrame(rows), "bank_matching",
                    "Bank-matching rule BANKS = ceil(D/8): does it restore "
                    "utilization?",
                    "Power-of-two bank counts from the banks suite; see "
                    "fig14 / hypotheses.md for the exact (modulo-mapped) "
                    "ceil(D/8) counts.  One bank = one BRAM36-wide word, so "
                    "the bank count is the minimum BRAM36 count; capacity "
                    "can require more (see PROPOSAL_REVIEW.txt).")

    # Ideal-scratchpad baseline vs. analytical count
    if ideal is not None:
        t = ideal[["workload", "N", "schedule_cycles", "total_cycles",
                   "dma_stall_cycles", "ideal_util", "pe_util"]].copy()
        t["gem5 - analytical (cycles)"] = t.total_cycles - t.schedule_cycles
        t["excess from DRAM stalls"] = t.dma_stall_cycles
        t = t.sort_values(["workload", "N"])
        write_table(t, "ideal_baseline",
                    "Infinite-bandwidth scratchpad vs. analytical cycle count",
                    "schedule_cycles = nTiles*T + M - 1 + 2N.  With an ideal "
                    "scratchpad every remaining cycle above it is a DRAM "
                    "(weight-DMA) stall; with --ideal-spm --ideal-dram the "
                    "difference is exactly 0.")

    # Table II metrics for every main run (proposal names in the notes)
    t = main.copy()
    t["layer_time_us"] = t.layer_time_s * 1e6
    t["dram_energy_uJ"] = t.dram_energy_J * 1e6
    if sw is not None:
        t = speedup_frame(t, sw)
    else:
        t["speedup"] = float("nan")
    t2 = t[["workload", "N", "banks", "pe_util", "pred_util", "layer_util",
            "spm_stall_cycles", "dma_stall_cycles", "spm_bw_read",
            "spm_bw_write", "spm_bw_total", "bank_imbalance",
            "avg_spm_latency", "layer_time_us", "speedup", "host_cpi",
            "host_ipc", "dram_bus_util", "dram_energy_uJ", "edp_dram_Js",
            "verify_mismatches"]].sort_values(["workload", "N", "banks"])
    write_table(t2, "table2_metrics",
                "Table II metrics for the 64-point sweep (gem5, host-driven)",
                "pe_util = system.accel.peUtilization (compute phase); "
                "layer_util also counts the A-load and C-store DMA phases.  "
                "spm_stall_cycles = system.accel.spmStallCycles.  "
                "spm_bw_* = sum over banks of system.spm*.bwRead/bwWrite::"
                "total, in bytes per accelerator cycle.  bank_imbalance = "
                "max/mean of system.spm_xbar.pktCount per bank (1.0 = "
                "perfect).  avg_spm_latency = system.accel.avgSpmLatency "
                "(cycles).  layer_time_us = simSeconds of the layer dump.  "
                "speedup = software GEMM time on the RISC-V host / layer "
                "time.  host_cpi/ipc = system.cpu.cpi/ipc during the layer "
                "(the driver polls the status register).  dram_bus_util = "
                "system.mem_ctrl.dram.busUtil (%).  dram_energy = sum of "
                "system.mem_ctrl.dram.rank*.totalEnergy; edp_dram_Js = "
                "DRAM energy x layer time (array energy needs Vivado).")
    if sw is not None:
        sp = speedup_frame(main, sw)
        piv = sp.pivot_table(index=["workload", "N"], columns="banks",
                             values="speedup").reset_index()
        piv.columns = [f"speedup BANKS={c}" if isinstance(c, (int, np.integer))
                       else c for c in piv.columns]
        sw_t = sw[["workload", "layer_time_s", "host_cpi", "host_ipc",
                   "host_insts"]].rename(columns={
                       "layer_time_s": "sw_time_s"})
        write_table(piv, "speedup", "Speedup over software on the RISC-V "
                    "host", "Baseline (sw suite): the same GEMM as a C "
                    "loop on the RiscvTimingSimpleCPU at 2 GHz with 32 KiB "
                    "L1s and a 512 KiB L2:\n\n" +
                    sw_t.to_markdown(index=False, floatfmt=".4g"))

    # Standalone (no CPU) vs host-driven: the compute phase must agree
    if standalone is not None:
        j = main.merge(standalone, on=["workload", "N", "banks"],
                       suffixes=("_host", "_standalone"))
        j["diff"] = j.pe_util_host - j.pe_util_standalone
        write_table(j[["workload", "N", "banks", "pe_util_host",
                       "pe_util_standalone", "diff", "load_cycles_host",
                       "store_cycles_host"]].sort_values(
                           ["workload", "N", "banks"]),
                    "consistency_host_vs_standalone",
                    "Host-driven vs standalone runs (compute phase)",
                    "Per workload, max |difference in PE utilization|: " +
                    ", ".join(f"{w} {g.abs().max():.4f}" for w, g in
                              j.groupby("workload")["diff"]) + ".  "
                    "Host-driven runs add the A-load and C-store DMA "
                    "phases around the compute phase and reach DRAM "
                    "through the I/O cache.  The weight DMA already "
                    "prefetches tiles while A loads, so the compute phase "
                    "no longer starts with a cold DRAM access.  That is "
                    "the whole difference for the short bert_attn_qk "
                    "layer (dma_stall_cycles drop to 0).")

    # ---- goodness of fit of the analytical model, suite by suite
    fit_suites = ["main", "knee", "banks", "idealdram", "extend", "reuse",
                  "clock", "dram", "rulem", "sens"]
    have = [n for n in fit_suites if S.get(n) is not None]
    fits = [fit_row(n, S[n]) for n in have]
    allruns = pd.concat([S[n] for n in have]).drop_duplicates("tag")
    fits.append(fit_row("all of the above (unique runs)", allruns))
    clean = allruns[~aliasing(allruns)]
    fits.append(fit_row("all, without the aliasing bank counts", clean))
    if knee is not None:
        fits.append(fit_row("knee, without the aliasing bank counts",
                            knee[~aliasing(knee)]))
    write_table(pd.DataFrame(fits), "fit_quality",
                "Analytical model (model.py) vs gem5: PE utilization",
                "pred_util = stall-free utilization x min(1, B/D, DRAM "
                "ceiling).  The DRAM ceiling uses one calibrated constant "
                "(DDR4 sustains 70% of its peak bandwidth); everything else "
                "is eq. (1)-(2), the schedule, and the DMA-fill term (a "
                "weight tile crosses the scratchpad twice: D = (5NM + 2N^2)"
                " / max(M, N)).  The last two columns use eq. (1) exactly "
                "as the proposal writes it.  'Aliasing bank counts' are "
                "those that divide N - 1 (see knee_bank_by_bank and "
                "bank_policy).")
    key["fit"] = {r["suite"]: {k: r[k] for k in r if k != "suite"}
                  for r in fits}

    # ---- H1/H3: the knee, one bank at a time
    knee_rows = []
    if knee is not None:
        for wl in ex.COMPUTE_BOUND:
            M = ex.M_OF[wl]
            for n in ex.N_VALUES:
                s = knee[(knee.workload == wl) & (knee.N == n)] \
                    .sort_values("banks")
                if s.empty:
                    continue
                D = model.demand(n, M)
                rule = math.ceil(D / 8)

                def at(bk):
                    r = s[s.banks == bk]
                    return float(r.pe_util.iloc[0]) if len(r) else \
                        float("nan")
                ok90 = s[s.pe_util >= 0.9]
                ok99 = s[s.pe_util >= 0.99 * s.ideal_util]
                cross = float("nan")
                if len(ok90) and ok90.banks.min() > s.banks.min():
                    b1 = int(ok90.banks.min())
                    u0, u1 = at(b1 - 1), at(b1)
                    cross = D / (8 * (b1 - 1 + (0.9 - u0) / (u1 - u0)))
                knee_rows.append({
                    "workload": wl, "N": n, "D(N)": round(D, 1),
                    "ceil(D/8)": rule,
                    "util at ceil(D/8)-1": at(rule - 1),
                    "util at ceil(D/8)": at(rule),
                    "util at ceil(D/8)+1": at(rule + 1),
                    "stall-free util": float(s.ideal_util.iloc[0]),
                    "min BANKS for 90%": int(ok90.banks.min())
                    if len(ok90) else "none (DRAM-bound)",
                    "min BANKS for 99% of stall-free":
                        int(ok99.banks.min()) if len(ok99)
                        else "none (DRAM-bound)",
                    "D/B at the 90% crossing": cross,
                    "util at BANKS = N-1 (aliasing)": at(n - 1),
                })
        write_table(pd.DataFrame(knee_rows), "knee_bank_by_bank",
                    "The knee, one bank at a time (knee suite)",
                    "BANKS swept in steps of 1 from 0.4 to 1.5 x D/8.  "
                    "Eq. (1)-(2) put the 90% crossing at D/B = 1.11 (times "
                    "the stall-free utilization) and full utilization at "
                    "BANKS = ceil(D/8).")
        key["knee"] = knee_rows

    # ---- H1: scratchpad alone (weights free of DRAM)
    if idealdram is not None:
        d = idealdram.copy()
        d["load"] = d.model_D / d.B_bytes_per_cycle
        d["frac"] = d.pe_util / d.ideal_util
        under, over = d[d.load <= 1.0], d[d.load > 1.0]
        e = (over.frac - 1 / over.load).abs()
        key["idealdram"] = {
            "runs": len(d),
            "min_frac_when_D_le_B": float(under.frac.min()),
            "mean_abs_err_vs_B_over_D_when_D_gt_B": float(e.mean()),
            "max_abs_err_vs_B_over_D_when_D_gt_B": float(e.max()),
            "bert_ffn1_N32_B32": float(d[(d.workload == "bert_ffn1") &
                                         (d.N == 32) & (d.banks == 32)]
                                       .pe_util.iloc[0]),
        }

    # ---- H1: beyond N = 32
    if extend is not None:
        d = extended_frame(main, banks_df, extend).sort_values(
            ["banks", "N"])
        t = d[["N", "banks", "model_D", "pe_util", "pred_util",
               "ideal_util"]].copy()
        t["D/B"] = t.model_D / (8 * t.banks)
        write_table(t, "extended_N", "ResNet-50 beyond N = 32",
                    "Eq. (1) puts D = B at N = 25.4 (16 banks), 50.6 (32 "
                    "banks) and 100 (64 banks).  N = 40-64 is outside the "
                    "proposal's range and only tests that prediction.")
        key["extended"] = {
            "eq_N_where_D_eq_B": {str(b): float(model_n90(784, b)[1])
                                  for b in (16, 32, 64)},
            "util_32_banks": {str(int(r.N)): float(r.pe_util)
                              for r in d[d.banks == 32].itertuples()},
        }

    # ---- H1: sensitivity to design knobs
    if sens is not None:
        d = sens_frame(sens)
        rows = []
        for (n, b), g in d.groupby(["N", "banks"]):
            base = float(g[g.variant == ""].pe_util.iloc[0])
            dev = (g.pe_util - base).abs()
            worst = g.loc[dev.idxmax()]
            rows.append({"N": n, "BANKS": b,
                         "D/B": model.demand(n, 784) / (8 * b),
                         "default": base, "min": float(g.pe_util.min()),
                         "max": float(g.pe_util.max()),
                         "max |change|": float(dev.max()),
                         "largest change from": SENS_LABEL[worst.variant]})
        write_table(pd.DataFrame(rows), "sensitivity",
                    "Sensitivity of PE utilization to design knobs "
                    "(ResNet-50)",
                    "One knob changed at a time: lookahead 4/16 (default "
                    "8), bank latency 2/4 cycles (1), psum write buffer "
                    "16/32 entries per lane (8), weight buffer 4/64 KiB "
                    "(16), DMA reads in flight 32/128 (64).")
        key["sensitivity"] = rows

    # ---- H2: M = 1 against DRAM supply
    if clock is not None:
        g = gemv_frame(main, clock, dram).sort_values(
            ["mem_type", "clock_hz", "N", "banks"])
        t = g[["N", "banks", "clock_hz", "mem_type", "dram_peak_GBs",
               "sustained_GBs", "calibration", "pe_util", "ideal_util",
               "cap_frac", "ratio", "spm_ratio", "model_frac",
               "dram_bus_util", "dram_read_bw_GBs"]] \
            .rename(columns={"ideal_util": "1/N cap (stall-free)",
                             "cap_frac": "util / cap",
                             "ratio": "DRAM supply / need",
                             "spm_ratio": "SPM supply / need",
                             "model_frac": "model util / cap"})
        write_table(t, "gemv_dram_supply",
                    "M = 1 (LLaMA GEMV): utilization vs DRAM supply",
                    "DRAM supply = sustained bandwidth of the DRAM type / "
                    "array clock (bytes per array cycle).  The sustained "
                    "bandwidth of each type is read from one run "
                    "(calibration = True: N = 32, 2 GHz, 8 banks); all "
                    "other rows are predictions.  DRAM need = N bytes per "
                    "cycle (N^2 weight bytes per N-cycle tile).  SPM "
                    "supply / need = 8 x BANKS / (2N + 5).  Model: "
                    "utilization = (1/N cap) x min(1, DRAM ratio, SPM "
                    "ratio).")
        wide = g[~g.calibration]
        e = (wide.cap_frac - wide.model_frac).abs()
        slow = g[(g.clock_hz == 250e6) & (g.mem_type == "DDR4_2400_8x8")]
        key["dram_sustained_GBs"] = {
            r.mem_type: {"peak": float(r.dram_peak_GBs),
                         "sustained": float(r.sustained_GBs),
                         "fraction": float(r.sustained_GBs /
                                           r.dram_peak_GBs)}
            for r in g[g.calibration].itertuples()}
        key["gemv"] = {
            "runs": len(g), "predicted_runs": len(wide),
            "mean_abs_err_cap_frac": float(e.mean()),
            "max_abs_err_cap_frac": float(e.max()),
            "util_2GHz_DDR4": {str(int(r.N)): float(r.pe_util) for r in
                               g[(g.clock_hz == 2e9) & (g.banks == 8) &
                                 (g.mem_type == "DDR4_2400_8x8")]
                               .itertuples()},
            "util_250MHz_DDR4": {str(int(r.N)): float(r.pe_util) for r in
                                 slow[slow.banks == 8].itertuples()},
            "cap_frac_250MHz_by_banks": {
                f"N{int(r.N)}_B{int(r.banks)}": float(r.cap_frac)
                for r in slow.itertuples()},
            "spread_across_banks_4_to_32_at_2GHz_pct": float(
                100 * g[(g.clock_hz == 2e9) & (g.banks >= 4) &
                        (g.mem_type == "DDR4_2400_8x8")]
                .groupby("N").pe_util.agg(lambda v: v.max() / v.min() - 1)
                .max()),
            "cap_frac_2GHz_1_and_2_banks": {
                f"N{int(r.N)}_B{int(r.banks)}": float(r.cap_frac)
                for r in g[(g.clock_hz == 2e9) & (g.banks <= 2) &
                           (g.mem_type == "DDR4_2400_8x8")].itertuples()},
        }
        # Scratchpad-bound layers at 250 MHz vs 2 GHz
        c = clock[clock.workload != "llama7b_gemv"]
        j = c.merge(main, on=["workload", "N", "banks"],
                    suffixes=("_250MHz", "_2GHz"))
        j["diff"] = j.pe_util_250MHz - j.pe_util_2GHz
        write_table(j[["workload", "N", "banks", "pe_util_2GHz",
                       "pe_util_250MHz", "diff", "dma_stall_cycles_2GHz",
                       "dma_stall_cycles_250MHz"]].sort_values(
                           ["workload", "N", "banks"]),
                    "clock_independence",
                    "Array clock 250 MHz vs 2 GHz (compute phase)",
                    "Scratchpad stalls are counted in array cycles and do "
                    "not depend on the clock.  DRAM delivers 8x more "
                    "bytes per array cycle at 250 MHz, so only the "
                    "DRAM-limited points change.")
        key["clock_independence"] = {
            w: float(v.abs().max()) for w, v in
            j.groupby("workload")["diff"]}
    if dram is not None:
        b = dram[dram.workload == "bert_ffn1"]
        base = main[(main.workload == "bert_ffn1") & (main.N == 32) &
                    (main.banks == 32)]
        t = pd.concat([base, b])[["mem_type", "dram_peak_GBs", "pe_util",
                                  "pred_util", "dma_stall_cycles",
                                  "dram_bus_util"]]
        write_table(t.sort_values("dram_peak_GBs"), "bert_n32_dram_types",
                    "BERT FFN1, N = 32, 32 banks: the DRAM-limited point "
                    "under other DRAM types")
        key["bert_n32_dram"] = {r.mem_type: float(r.pe_util)
                                for r in t.itertuples()}

    # ---- H3: the rule at every reuse M
    if rulem is not None:
        d = rulem_frame(rulem)
        t = d[["M", "N", "banks", "kind", "model_D", "frac", "pe_util",
               "ideal_util"]].sort_values(["kind", "M", "N"]) \
            .rename(columns={"frac": "util / stall-free"})
        write_table(t, "rule_vs_reuse",
                    "Bank-matching rule across reuse M (weights free of "
                    "DRAM)", "kind = rule: BANKS = ceil(D(N, M)/8); "
                    "kind = fewer: floor(0.75 x rule).")
        key["rule_vs_reuse"] = {
            k: {"runs": len(v), "min": float(v.frac.min()),
                "mean": float(v.frac.mean()), "max": float(v.frac.max())}
            for k, v in d.groupby("kind")}

    # ---- bank policy (aliasing at BANKS = N - 1)
    policy = S.get("policy")
    if policy is not None:
        d = policy_frame(policy)
        d["case"] = np.where(d.banks == d.N - 1, "BANKS = N-1 (aliasing)",
                             np.where(d.la == 4, "lookahead 4",
                                      np.where(d.model_D > 8 * d.banks,
                                               "saturated (D > B)",
                                               "rule ceil(D/8)")))
        t = d.pivot_table(index=["workload", "N", "banks", "case", "pq"],
                          columns="bank_policy", values="pe_util") \
            .reset_index().rename(columns={"pq": "write buffer"})
        t["pred_util"] = [float(d[(d.workload == r.workload) & (d.N == r.N)
                                  & (d.banks == r.banks)].pred_util.iloc[0])
                          for r in t.itertuples()]
        write_table(t.sort_values(["case", "workload", "N"]), "bank_policy",
                    "Bank queue policy: oldest first (fifo) vs reads first",
                    "BANKS = N-1 makes every partial-sum lane address the "
                    "same bank in the same cycle (the lanes are 2 cycles "
                    "and N-1 words apart).  With oldest-first queues the "
                    "reads wait behind that burst of N/2 writes.  Serving "
                    "reads first and holding the burst in a 16-entry "
                    "write buffer removes the dip.")
        al = t[t.case == "BANKS = N-1 (aliasing)"]
        key["bank_policy"] = {
            "aliasing_fifo_8": {f"{r.workload}_N{r.N}": float(r.fifo)
                                for r in al[al["write buffer"] == 8]
                                .itertuples()},
            "aliasing_reads_first_8": {
                f"{r.workload}_N{r.N}": float(r.reads_first)
                for r in al[al["write buffer"] == 8].itertuples()},
            "aliasing_reads_first_16": {
                f"{r.workload}_N{r.N}": float(r.reads_first)
                for r in al[al["write buffer"] == 16].itertuples()},
            "lookahead4": {f"{r.workload}_N{r.N}_B{r.banks}":
                           [float(r.fifo), float(r.reads_first)]
                           for r in t[t.case == "lookahead 4"].itertuples()},
            "max_abs_change_elsewhere": float(
                (t[t.case.isin(["rule ceil(D/8)", "saturated (D > B)"])]
                 .eval("reads_first - fifo")).abs().max()),
        }

    # ---- BRAM estimate (not a Vivado measurement)
    rows = []
    for wl, (M, K, NOUT) in model.WORKLOADS.items():
        for n in ex.N_VALUES:
            D = model.demand(n, M)
            rule = math.ceil(D / 8)
            nbytes = spm_bytes(M, K, NOUT, n)
            rows.append({
                "workload": wl, "N": n, "D(N)": round(D, 1),
                "rule banks": rule,
                "scratchpad KiB (whole layer)": round(nbytes / 1024, 1),
                "BRAM36, bandwidth only": rule,
                "BRAM36, whole layer resident": spm_bram36(nbytes, rule),
                "of 312 on ZCU106": f"{100 * spm_bram36(nbytes, rule) / 312:.0f}%",
            })
    write_table(pd.DataFrame(rows), "bram_estimate",
                "BRAM36 estimate for the bank-matching rule (NOT measured)",
                "An estimate from the bank count and the scratchpad size "
                "the gem5 model uses (blocked A + 16 KiB weight buffer + "
                "INT32 C); a BRAM36 in 72-bit mode holds 512 x 64 data bits "
                "= 4 KiB.  The proposal's 'measured BRAM cost' needs "
                "Vivado.  Layers above 100% do not fit on the ZCU106 "
                "without tiling M (see PROPOSAL_REVIEW.txt A3).")

    # ---- Hypothesis check (proposal Sec. V)
    lines = ["# Hypothesis checks (auto-generated by plot_curves.py)\n"]
    cb = main[main.workload.isin(ex.COMPUTE_BOUND)].copy()
    cb["load"] = cb.model_D / cb.B_bytes_per_cycle
    under = cb[cb.load <= 1.0]
    over = cb[cb.load > 1.0]
    err = (over.pe_util / over.ideal_util -
           over.B_bytes_per_cycle / over.model_D).abs()
    low = under[under.pe_util < 0.9]
    key["h1_main"] = {
        "runs_D_le_B": len(under), "runs_D_le_B_above_90pct":
        len(under) - len(low),
        "min_util_D_le_B_excluding_below_90": float(
            under[under.pe_util >= 0.9].pe_util.min()),
        "below_90": [f"{r.workload} N={r.N} BANKS={r.banks}: "
                     f"{r.pe_util:.3f}" for r in low.itertuples()],
        "runs_D_gt_B": len(over), "mean_abs_err_vs_B_over_D":
        float(err.mean()), "max_abs_err_vs_B_over_D": float(err.max()),
    }
    lines.append(
        "## H1: utilization > 90% until delivered bandwidth reaches B, then "
        "~B/D\n\n"
        f"* Main sweep, compute-bound runs with D <= B: {len(under)}; "
        f"{len(under) - len(low)} are above 90% (minimum of those "
        f"{key['h1_main']['min_util_D_le_B_excluding_below_90']:.3f}).  "
        "Below 90%: "
        + (", ".join(f"{r.workload} N={r.N} BANKS={r.banks} "
                     f"({r.pe_util:.3f}; DMA stalls {r.dma_stall_cycles:.0f},"
                     f" SPM stalls {r.spm_stall_cycles:.0f})"
                     for r in low.itertuples()) or "none") + ".\n"
        f"* Main sweep, runs with D > B: {len(over)}; utilization / "
        f"stall-free utilization vs. B/D: mean |error| {err.mean():.3f}, "
        f"max {err.max():.3f}.\n")
    if knee_rows:
        lines.append("* Knee suite (BANKS in steps of 1):\n")
        for r in knee_rows:
            lines.append(
                f"  * {r['workload']} N={r['N']}: D={r['D(N)']}, "
                f"ceil(D/8)={r['ceil(D/8)']}; utilization "
                f"{r['util at ceil(D/8)-1']:.3f} / "
                f"{r['util at ceil(D/8)']:.3f} / "
                f"{r['util at ceil(D/8)+1']:.3f} at rule-1 / rule / rule+1 "
                f"(stall-free {r['stall-free util']:.3f}); min BANKS for "
                f"90%: {r['min BANKS for 90%']}; 90% crossing at D/B = "
                f"{r['D/B at the 90% crossing']:.2f}\n")
    if "idealdram" in key:
        k = key["idealdram"]
        lines.append(
            f"* Weights free of DRAM ({k['runs']} runs, 3 layers): minimum "
            f"utilization / stall-free when D <= B is "
            f"{k['min_frac_when_D_le_B']:.3f}; when D > B it matches B/D "
            f"within {k['mean_abs_err_vs_B_over_D_when_D_gt_B']:.3f} on "
            f"average (max {k['max_abs_err_vs_B_over_D_when_D_gt_B']:.3f}).  "
            f"BERT FFN1 N=32 BANKS=32 becomes "
            f"{k['bert_ffn1_N32_B32']:.3f}.\n")
    if "extended" in key:
        k = key["extended"]
        lines.append(
            "* Beyond the swept range (ResNet-50, 32 banks; eq. (1) puts "
            f"D = B at N = {k['eq_N_where_D_eq_B']['32']:.1f}): "
            + ", ".join(f"N={n}: {u:.3f}" for n, u in
                        sorted(k["util_32_banks"].items(),
                               key=lambda kv: int(kv[0]))) + "\n")
    if "sensitivity" in key:
        worst = max(key["sensitivity"], key=lambda r: r["max |change|"])
        lines.append(
            "* Sensitivity (ResNet-50, 10 knob changes at 6 points): "
            f"largest change in utilization {worst['max |change|']:.3f} "
            f"(N={worst['N']}, BANKS={worst['BANKS']}, "
            f"{worst['largest change from']}).\n")
    if "clock_independence" in key:
        lines.append(
            "* Array clock 250 MHz vs 2 GHz, max |change| in utilization: "
            + ", ".join(f"{w} {v:.3f}" for w, v in
                        key["clock_independence"].items()) + "\n")
    if "fit" in key:
        f = key["fit"]["all of the above (unique runs)"]
        c = key["fit"]["all, without the aliasing bank counts"]
        lines.append(
            f"* Model vs gem5 over {f['runs']} unique runs: R^2 = "
            f"{f['R^2']:.4f}, RMSE = {f['RMSE']:.4f}, max |error| = "
            f"{f['max |err|']:.3f}.  Without the {f['runs'] - c['runs']} "
            f"runs at aliasing bank counts: R^2 = {c['R^2']:.4f}, mean "
            f"|error| = {c['mean |err|']:.4f}, max = {c['max |err|']:.3f}.  "
            "With eq. (1) exactly as written (no DMA-fill term): R^2 = "
            f"{c['R^2, eq. (1) as written']:.4f}, mean |error| = "
            f"{c['mean |err|, eq. (1) as written']:.4f}.\n")
    if "bank_policy" in key:
        k = key["bank_policy"]
        lines.append(
            "* Aliasing at BANKS = N-1 (every psum lane on one bank).  "
            "Oldest-first queues: "
            + ", ".join(f"{t} {v:.3f}" for t, v in
                        k["aliasing_fifo_8"].items())
            + ".  Reads first + 16-entry write buffer: "
            + ", ".join(f"{t} {v:.3f}" for t, v in
                        k["aliasing_reads_first_16"].items())
            + ".  Elsewhere (rule counts, saturated points) the policy "
            f"changes utilization by at most "
            f"{k['max_abs_change_elsewhere']:.3f}; with lookahead 4: "
            + ", ".join(f"{t} {a:.3f} -> {b:.3f}" for t, (a, b) in
                        k["lookahead4"].items()) + ".\n")

    g = main[main.workload == "llama7b_gemv"]
    if len(g):
        spread = g.groupby("N").pe_util.agg(lambda v: v.max() / v.min() - 1)
        lines.append(
            "\n## H2: M=1 follows 1/N and DRAM bandwidth regardless of "
            "BANKS\n\n"
            + "".join(f"* N={n} (2 GHz, DDR4): utilization "
                      f"{g[g.N == n].pe_util.mean():.4f} (1/N = "
                      f"{1 / n:.4f}); spread across BANKS 4-32 "
                      f"{100 * spread[n]:.2f}%; DRAM bus "
                      f"{g[g.N == n].dram_bus_util.mean():.1f}%\n"
                      for n in sorted(g.N.unique())))
    if "gemv" in key:
        k = key["gemv"]
        lines.append(
            f"* {k['runs']} M=1 runs over 4 array clocks, 4 DRAM types and "
            "BANKS 1-32.  One run per DRAM type calibrates its sustained "
            f"bandwidth; for the other {k['predicted_runs']}, utilization "
            "/ (1/N cap) vs min(1, DRAM supply / need, SPM supply / need): "
            f"mean |error| {k['mean_abs_err_cap_frac']:.3f}, max "
            f"{k['max_abs_err_cap_frac']:.3f}.\n"
            "* Sustained DRAM bandwidth (GB/s, fraction of peak): "
            + ", ".join(f"{t} {v['sustained']:.1f} ({v['fraction']:.2f})"
                        for t, v in key["dram_sustained_GBs"].items())
            + "\n"
            "* At 250 MHz (DRAM supply 8x per array cycle) utilization "
            "is " + ", ".join(f"N={n}: {u:.4f}" for n, u in
                              sorted(k["util_250MHz_DDR4"].items(),
                                     key=lambda kv: int(kv[0])))
            + " (1/N = 0.125, 0.0625, 0.0417, 0.0313).\n"
            "* BANKS at 250 MHz, utilization / cap: "
            + ", ".join(f"{t}: {v:.2f}" for t, v in
                        sorted(k["cap_frac_250MHz_by_banks"].items(),
                               key=lambda kv: tuple(
                                   int(x[1:]) for x in kv[0].split("_"))))
            + "\n")
    if "bert_n32_dram" in key:
        lines.append("* BERT FFN1 N=32 BANKS=32 by DRAM type: "
                     + ", ".join(f"{t} {u:.3f}" for t, u in
                                 key["bert_n32_dram"].items()) + "\n")

    if bankrule is not None:
        lines.append("\n## H3: the bank-matching rule restores utilization"
                     "\n\n")
        brr = bankrule.copy()
        brr["pq"] = brr.psum_queue.fillna(8).astype(int)
        h3 = []
        for wl in ex.COMPUTE_BOUND:
            for n in ex.N_VALUES:
                D = model.demand(n, ex.M_OF[wl])
                parts, row = [], {"workload": wl, "N": n, "D": D}
                for lab, b in ex.bank_rule_counts(n, ex.M_OF[wl]):
                    for pq in sorted(brr.pq.unique()):
                        r = brr[(brr.workload == wl) & (brr.N == n) &
                                (brr.banks == b) & (brr.pq == pq)]
                        if len(r):
                            u = float(r.pe_util.iloc[0])
                            parts.append(f"{lab}={b} banks, buffer {pq}: "
                                         f"{u:.3f}")
                            row[f"{lab}_banks"] = b
                            row[f"{lab}_buffer{pq}"] = u
                h3.append(row)
                lines.append(f"* {wl} N={n} (D={D:.1f}): " +
                             "; ".join(parts) + "\n")
        key["h3_bankrule"] = h3
    if "rule_vs_reuse" in key:
        k = key["rule_vs_reuse"]
        lines.append(
            "* Rule at every reuse M (M = 8..512, N = 8..32, weights free "
            f"of DRAM): at BANKS = ceil(D/8), utilization / stall-free is "
            f"{k['rule']['min']:.3f}-{k['rule']['max']:.3f} (mean "
            f"{k['rule']['mean']:.3f}, {k['rule']['runs']} runs); with 25% "
            f"fewer banks it is {k['fewer']['min']:.3f}-"
            f"{k['fewer']['max']:.3f} (mean {k['fewer']['mean']:.3f}).\n")
    with open(os.path.join(TAB, "hypotheses.md"), "w") as f:
        f.writelines(lines)
    print("  results/tables/hypotheses.md")

    # ---- other headline numbers, then data/key_results.json
    key["runs"] = {n: (len(S[n]) if S.get(n) is not None else 0)
                   for n in ex.SUITES}
    every = pd.concat([S[n] for n in ex.SUITES if S.get(n) is not None])
    key["runs"]["unique"] = int(every.tag.nunique())
    key["runs"]["verify_failures"] = int(
        (every.verify_mismatches != 0).sum())
    b32 = main[main.banks == 32]
    key["layer_phases_32_banks"] = {
        f"{r.workload}_N{r.N}": {
            "load_frac": float(r.load_cycles / r.layer_cycles),
            "store_frac": float(r.store_cycles / r.layer_cycles),
            "pe_util": float(r.pe_util), "layer_util": float(r.layer_util),
            "layer_time_us": float(r.layer_time_s * 1e6)}
        for r in b32.itertuples()}
    key["host"] = {"cpi_min": float(main.host_cpi.min()),
                   "cpi_max": float(main.host_cpi.max()),
                   "ipc_min": float(main.host_ipc.min()),
                   "ipc_max": float(main.host_ipc.max())}
    key["dram_bus_util_pct"] = {
        w: [float(v.min()), float(v.max())]
        for w, v in main.groupby("workload").dram_bus_util}
    key["avg_spm_latency_cycles"] = [
        float(main[main.workload != "llama7b_gemv"].avg_spm_latency.min()),
        float(main[main.workload != "llama7b_gemv"].avg_spm_latency.max())]
    key["bank_imbalance_max"] = float(main.bank_imbalance.max())
    if sw is not None:
        sp = speedup_frame(main, sw)
        key["speedup"] = {
            w: {"min": float(v.speedup.min()), "max": float(v.speedup.max())}
            for w, v in sp.groupby("workload")}
        key["sw_baseline_s"] = {r.workload: float(r.layer_time_s)
                                for r in sw.itertuples()}
    if nodb is not None:
        j = nodb.merge(main, on=["workload", "N", "banks"],
                       suffixes=("_nodb", "_db"))
        j["loss"] = j.pe_util_db - j.pe_util_nodb
        key["double_buffer_loss"] = {
            w: [float(v.min()), float(v.max())]
            for w, v in j.groupby("workload").loss}
    if lane is not None:
        j = lane.merge(main, on=["workload", "N", "banks"],
                       suffixes=("_lane", "_bank"))
        j["loss"] = j.pe_util_bank - j.pe_util_lane
        w = j.loc[j.loss.idxmax()]
        key["lane_arbitration_worst"] = {
            "workload": w.workload, "N": int(w.N), "banks": int(w.banks),
            "bank": float(w.pe_util_bank), "lane": float(w.pe_util_lane)}
    if reuse is not None:
        key["reuse_min_M_for_90pct"] = {}
        for bk in sorted(reuse.banks.unique()):
            for n in ex.N_VALUES:
                v = reuse[(reuse.banks == bk) & (reuse.N == n) &
                          (reuse.pe_util >= 0.9)]
                key["reuse_min_M_for_90pct"][f"B{bk}_N{n}"] = \
                    int(v.M.min()) if len(v) else None
    if ideal is not None:
        key["ideal_spm_excess_cycles"] = {
            f"{r.workload}_N{r.N}": int(r.total_cycles - r.schedule_cycles)
            for r in ideal.itertuples()}
    if standalone is not None:
        j = main.merge(standalone, on=["workload", "N", "banks"],
                       suffixes=("_host", "_standalone"))
        key["host_vs_standalone_max_abs_diff"] = {
            w: float(v.abs().max()) for w, v in
            (j.pe_util_host - j.pe_util_standalone).groupby(j.workload)}
    import json
    with open(os.path.join(DATA, "key_results.json"), "w") as f:
        json.dump(key, f, indent=2, default=str)
    print("  results/data/key_results.json")


# ------------------------------------------------------------------- main
FIGS = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8",
        "fig9", "fig10", "fig11", "fig12", "fig13", "fig14", "fig15",
        "fig16", "fig17", "fig18", "fig19", "fig20", "fig21", "fig22",
        "tables"]


def main():
    global FORMATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=FIGS)
    ap.add_argument("--format", nargs="*", default=FORMATS,
                    choices=["png", "pdf", "svg"])
    args = ap.parse_args()
    FORMATS = args.format
    want = set(args.only or FIGS)
    st.setup()

    S = {name: load(name) for name in ex.SUITES}
    main_df = S["main"]
    if main_df is None:
        sys.exit("results/data/main.csv missing: run  make data  first")

    def need(*suites):
        missing = [n for n in suites if S[n] is None]
        if missing:
            print(f"  skipped: needs suite(s) {', '.join(missing)} "
                  f"(make data SUITES=\"{' '.join(missing)}\")")
        return not missing

    print("writing figures and tables:")
    if "fig1" in want:
        fig1_saturation(main_df)
    if "fig2" in want:
        fig2_universal(main_df, S["knee"], S["idealdram"])
    if "fig3" in want:
        fig3_bandwidth(main_df)
    if "fig4" in want and need("banks"):
        fig4_banks(S["banks"])
    if "fig5" in want and need("reuse"):
        fig5_reuse(S["reuse"])
    if "fig6" in want:
        fig6_gemv(main_df)
    if "fig7" in want:
        fig7_stalls(main_df)
    if "fig8" in want and need("nodb"):
        fig8_double_buffer(main_df, S["nodb"])
    if "fig9" in want and need("lane"):
        fig9_arbitration(main_df, S["lane"])
    if "fig10" in want:
        fig10_amat(main_df)
    if "fig11" in want:
        fig11_parity(main_df)
    if "fig12" in want:
        fig12_rtl(main_df)
    if "fig13" in want:
        fig13_edp(main_df)
    if "fig14" in want and need("bankrule"):
        fig14_bank_rule(S["bankrule"])
    if "fig15" in want:
        fig15_layer_phases(main_df)
    if "fig16" in want and need("sw"):
        fig16_speedup(main_df, S["sw"])
    if "fig17" in want and need("knee"):
        fig17_knee(S["knee"])
    if "fig18" in want and need("clock", "dram"):
        fig18_gemv_dram(main_df, S["clock"], S["dram"])
    if "fig19" in want and need("sens"):
        fig19_sensitivity(S["sens"])
    if "fig20" in want and need("rulem"):
        fig20_rule_vs_m(S["rulem"])
    if "fig21" in want and need("extend"):
        fig21_extended(main_df, S["banks"], S["extend"])
    if "fig22" in want and need("policy"):
        fig22_bank_policy(S["policy"])
    if "tables" in want:
        tables(S)


if __name__ == "__main__":
    main()
