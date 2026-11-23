#!/usr/bin/env python3
#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : plot_curves.py                                                #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : Saturation curves, measurements and tables                    #
#-----------------------------------------------------------------------------#
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
#   results/data/vivado.csv      N,banks,fmax_mhz,array_power_w,
#                                bram_energy_pj_per_access -> fig13 (EDP),
#                                table (the older N,power_W,fmax_MHz,... also
#                                works, without the BRAM term)
#   (templates: analysis/templates/*.example)

import argparse
import math
import os
import sys

import warnings

import numpy as np
import pandas as pd

# Merges of the wide per-run CSVs; only about speed, and plotting is fast.
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

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
# Set by --paper rtlcfg: figures read the RTL's configuration, where the load
# is on the busier port and the bank rule is that port's.
PORT_LOAD = False


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
    print(f"  results/{os.path.relpath(FIG, RESULTS)}/{name}."
          f"{{{','.join(FORMATS)}}}")


def write_table(df, name, title, notes=""):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w") as f:
        f.write(f"# {title}\n\n")
        if notes:
            f.write(notes.strip() + "\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4g"))
        f.write("\n")
    print(f"  results/{os.path.relpath(TAB, RESULTS)}/{name}.{{md,csv}}")


def fig_rule_banks(n, M):
    """The bank rule the figures mark."""
    return ex.rtl_rule(n, M) if PORT_LOAD else math.ceil(model.demand(n, M) / 8)


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
    l.append("analytical model, busier port + DRAM ceiling" if PORT_LOAD
             else "analytical model, eq. (1)-(2) + DRAM ceiling")
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
        ax.set_xlabel("Load  max(D_read, D_write) / B" if PORT_LOAD else
                      "Load  D(N) / B   (demand, eq. 1 / supply, eq. 2)")
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
    # The RTL testbench preloads the weights, so the runs without DRAM are
    # the matching ones.  Set RTL_STALL_MODEL=decoupled in the environment to
    # compare with the queued model instead.
    # The RTL's configuration is simple dual port with a global stall.
    # The rtlcfg runs assume the settings in experiments.RTL_OPTS; set
    # RTL_STALL_MODEL=baseline to compare with one shared port.
    how = os.environ.get("RTL_STALL_MODEL")
    first = {"decoupled": "rtlcfg_rtlval", "baseline": "rtlval_idealdram",
             "single_buffer": "rtlcfg_nodb_rtlval_lockstep"}.get(
                 how, "rtlcfg_rtlval_lockstep")
    runs = []
    for suite in (first, "rtlval_idealdram", "rtlval", "main"):
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
    Array energy = Vivado power x active time + accesses x energy per BRAM
    access; until data/vivado.csv has numbers only the DRAM part is plotted.
    Blank cells in vivado.csv count as no number yet."""
    j = main.copy()
    p = os.path.join(DATA, "vivado.csv")
    have_viv = os.path.exists(p)
    j["bram_energy_J"] = 0.0
    if have_viv:
        viv = pd.read_csv(p)
        viv = viv.rename(columns={"power_W": "array_power_w",
                                  "fmax_MHz": "fmax_mhz"})
        on = ["N", "banks"] if "banks" in viv else ["N"]
        viv = viv.dropna(subset=["array_power_w", "fmax_mhz"])
        have_viv = len(viv) > 0
    if have_viv:
        j = j.merge(viv, on=on)
        active_s = j.active_cycles / (j.fmax_mhz * 1e6)
        j["array_energy_J"] = j.array_power_w * active_s
        if "bram_energy_pj_per_access" in j:
            j["bram_energy_J"] = (j.spm_reads_total + j.spm_writes_total) * \
                j.bram_energy_pj_per_access.fillna(0) * 1e-12
    else:
        j["array_energy_J"] = 0.0
    j["edp_Js"] = (j.array_energy_J + j.bram_energy_J + j.dram_energy_J) * \
        j.layer_time_s
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
        cols[4:4] = ["fmax_mhz", "array_power_w"]
        cols.insert(cols.index("edp_Js"), "bram_energy_J")
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
            rule = fig_rule_banks(n, M)
            ax.plot([rule, rule], [1.03, 1.09], color=col[n], lw=1.8,
                    solid_capstyle="butt", zorder=3)
        ax.text(0.4, 1.10, "busier-port rule for each N" if PORT_LOAD else
                "ceil(D/8) for each N", fontsize=6.5,
                color=st.MUTED, va="bottom")
        if not PORT_LOAD:
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
    g["spm_ratio"] = (8 * g.banks / g.model_D_port if PORT_LOAD else
                      8 * g.banks / (2 * g.N + 5))
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
        need = (nn + 1) / 8 if PORT_LOAD else (2 * nn + 5) / 8
        ax.plot([need, need], [1.04, 1.09], color=coln[nn], lw=1.8,
                solid_capstyle="butt")
    ax.text(0.9, 1.10, "BANKS = (N + 1) / 8" if PORT_LOAD else
            "BANKS = (2N + 5) / 8", fontsize=6.5, color=st.MUTED,
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
    if PORT_LOAD:
        # The suite also holds ceil(D/8) and its 25% fewer: keep this rule's.
        rule = [ex.rtl_rule(n, m) for n, m in zip(d.N, d.M)]
        d["kind"] = ["rule" if b == r else
                     "fewer" if b == max(1, math.floor(0.75 * r)) else "other"
                     for b, r in zip(d.banks, rule)]
        return d[d.kind != "other"]
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
            ("BANKS = busier-port rule" if PORT_LOAD else
             "BANKS = ceil(D(N, M) / 8)", "25% fewer banks than the rule")):
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


# ------------------------------------------------------- figures 23, 24
def fig23_lockstep(main, knee, lockstep):
    """The same simulator with and without queues between banks and array:
    steady-state utilization vs load, against both models."""
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 3.0), sharey=True)
    ls = lockstep.copy()
    ls["load"] = ls.model_D / ls.B_bytes_per_cycle
    dec = pd.concat([main, knee]).drop_duplicates(["workload", "N", "banks"])
    dec = dec.copy()
    dec["load"] = dec.model_D / dec.B_bytes_per_cycle
    x = np.logspace(np.log10(0.2), np.log10(5.5), 800)
    for ax, wl in zip(axs, ex.COMPUTE_BOUND):
        d = dec[dec.workload == wl].sort_values("load")
        # BANKS dividing N - 1 is the aliasing dip: a different effect.
        d = d[~aliasing(d)]
        ax.plot(x, np.minimum(1, 1 / x), color=st.INK2, lw=1.1,
                ls=(0, (3, 2)), label="model: min(1, B/D)", zorder=2)
        ax.plot(x, 1 / np.ceil(x), color=st.MUTED, lw=1.1,
                ls=(0, (1, 1.5)), label="evenly spread lockstep:\n"
                "1 / ceil(words / BANKS)", zorder=2)
        ax.scatter(d.load, d.steady_util, s=16,
                   color=st.WORKLOAD_COLOR[wl],
                   marker=st.WORKLOAD_MARKER[wl], edgecolor=st.SURFACE,
                   linewidth=0.4, label="gem5, decoupled", zorder=3)
        s_ = ls[ls.workload == wl]
        ax.scatter(s_.load, s_.steady_util, s=26, marker="X",
                   color=st.INK, edgecolor=st.SURFACE, linewidth=0.4,
                   label="gem5, lockstep (fixed addresses)", zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Load  D(N) / B")
        panel_title(ax, wl)
    axs[0].set_ylabel("Steady-state PE utilization")
    axs[0].legend(loc="lower left", fontsize=6.5)
    fig.suptitle("Decoupled vs. lockstep stall, one simulator", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig23_lockstep_vs_decoupled")


def fig24_bank_ports(knee, ports):
    """Utilization vs BANKS, single-port banks against simple dual port."""
    ns = sorted(knee.N.unique())
    col, mk = st.ramp(ns), st.markers(ns)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 3.0), sharey=True)
    p_ = ports[ports.psum_queue == 8]
    for ax, wl in zip(axs, ex.COMPUTE_BOUND):
        for n in ns:
            a = knee[(knee.workload == wl) & (knee.N == n)].sort_values("banks")
            b = p_[(p_.workload == wl) & (p_.N == n)].sort_values("banks")
            ax.plot(a.banks, a.pe_util, color=col[n], marker=mk[n], ms=3,
                    lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.plot(b.banks, b.pe_util, color=col[n], marker=mk[n], ms=3.5,
                    lw=1.4, label=f"N={n}", zorder=3)
        ax.set_xlim(0, 33)
        ax.set_xticks(range(0, 33, 4))
        ax.set_ylim(0, 1.08)
        st.threshold(ax, x=0.3)
        panel_title(ax, wl)
        ax.set_xlabel("BANKS")
    axs[0].set_ylabel("PE utilization")
    h, l = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=st.INK2, lw=1.4),
          plt.Line2D([], [], color=st.INK2, lw=0.9, ls=(0, (3, 2)))]
    l += ["1r1w (simple dual port)", "1rw (one port per bank)"]
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("A separate write port moves the knee", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig24_bank_ports")


# ------------------------------------------------------- figures 25, 26
def rtl_frame(S):
    """Decoupled runs of the RTL's configuration, compute-bound layers."""
    d = pd.concat([S["rtlcfg_main"], S["rtlcfg_knee"]]) \
        .drop_duplicates("tag")
    return d[d.workload.isin(ex.COMPUTE_BOUND)].copy()


def fig25_rtl_config(S):
    """The hypothesis curve under the RTL's configuration, on the proposal's
    load axis and on the load of the busier port."""
    d = rtl_frame(S)
    d["frac"] = d.pe_util / d.ideal_util
    d["literal"] = d.model_D / d.B_bytes_per_cycle
    d["port"] = d.model_D_port / d.B_bytes_per_cycle
    alias, dram = aliasing(d), d.pred_dram_ceiling < 0.999
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 3.0), sharey=True)
    x = np.logspace(np.log10(0.15), np.log10(6), 300)
    for ax, col, title, lab in (
            (axs[0], "literal", "Load as the proposal writes it",
             "Load  D(N) / B   (eq. 1 / eq. 2)"),
            (axs[1], "port", "Load on the busier port",
             "Load  max(D_read, D_write) / B")):
        ax.plot(x, np.minimum(1, 1 / x), color=st.INK2, lw=1.2,
                ls=(0, (3, 2)), label="model: min(1, 1 / load)", zorder=2)
        for wl in ex.COMPUTE_BOUND:
            s_ = d[(d.workload == wl) & ~alias & ~dram]
            ax.scatter(s_[col], s_.frac, s=18, color=st.WORKLOAD_COLOR[wl],
                       marker=st.WORKLOAD_MARKER[wl], edgecolor=st.SURFACE,
                       linewidth=0.5, label=st.WORKLOAD_SHORT[wl], zorder=3)
        s_ = d[dram & ~alias]
        ax.scatter(s_[col], s_.frac, s=20, marker="o", facecolor="none",
                   edgecolor=st.INK2, linewidth=0.8,
                   label="DRAM-limited", zorder=4)
        s_ = d[alias]
        ax.scatter(s_[col], s_.frac, s=20, marker="x", color=st.INK2,
                   linewidth=0.8, label="aliasing: BANKS divides N - 1",
                   zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.axvline(1, color=st.MUTED, lw=0.8, ls=(0, (1, 2)))
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(lab)
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("Utilization / stall-free utilization")
    axs[0].legend(loc="lower left", fontsize=6.5)
    fig.suptitle("Simple-dual-port banks: the knee follows the busier port",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig25_rtl_config_saturation")


def fig26_rtl_stall(S):
    """ResNet-50 in the RTL's configuration: queues against a global stall."""
    dec = rtl_frame(S)
    dec = dec[dec.workload == "resnet50_conv3x"]
    lock = S["rtlcfg_lockstep"]
    lock = lock[lock.workload == "resnet50_conv3x"]
    ns = ex.N_VALUES
    col, mk = st.ramp(ns), st.markers(ns)
    fig, ax = plt.subplots(figsize=(TWO_COL, 3.0))
    for n in ns:
        a = dec[(dec.N == n) & (dec.banks <= 32)].sort_values("banks")
        b = lock[(lock.N == n) & (lock.banks <= 32)].sort_values("banks")
        ax.plot(a.banks, a.steady_util, color=col[n], lw=0.9,
                ls=(0, (3, 2)), marker=mk[n], ms=2.5, zorder=2)
        ax.plot(b.banks, b.steady_util, color=col[n], lw=1.4, marker=mk[n],
                ms=3.5, label=f"N={n}", zorder=3)
    ax.set_xlim(0, 33)
    ax.set_xticks(range(0, 33, 4))
    ax.set_ylim(0, 1.08)
    st.threshold(ax, x=24)
    ax.set_xlabel("BANKS")
    ax.set_ylabel("Steady-state PE utilization")
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], color=st.INK2, lw=1.4),
          plt.Line2D([], [], color=st.INK2, lw=0.9, ls=(0, (3, 2)))]
    l += ["global stall (lockstep)", "decoupled queues"]
    ax.legend(h, l, loc="lower right", ncol=2, fontsize=6.5)
    ax.set_title("Same rule, but a global stall steps and dips between "
                 "its counts (ResNet-50, 1r1w)", loc="left")
    fig.tight_layout()
    save(fig, "fig26_rtl_config_stall")


# ------------------------------------------------------------- figure 27
PSUM_LABEL = {"A": "A: in the banks, one port per bank",
              "B": "B: in the banks, SDP, written once",
              "B+rmw": "B: SDP, read back and rewritten",
              "C": "C: outside the counted banks"}


def psum_option(df):
    """Which of experiments.PSUM_OPTIONS each run of the psumopts suite is."""
    return np.select(
        [df.psum_store == "external", df.psum_rmw == 1,
         df.bank_ports == "1r1w"], ["C", "B+rmw", "B"], default="A")


def fig27_psum_options(po):
    """What the place partial sums live does to the bank count."""
    d = po[po.workload == "resnet50_conv3x"].copy()
    d["option"] = psum_option(d)
    ns = ex.N_VALUES
    col, mk = st.ramp(ns), st.markers(ns)
    fig, axs = plt.subplots(2, 2, figsize=(TWO_COL, 4.6), sharex=True,
                            sharey=True)
    for ax, opt in zip(axs.flat, ex.PSUM_OPTIONS):
        for n in ns:
            s = d[(d.option == opt) & (d.N == n) & (d.banks <= 28)]
            a = s[s.stall_model == "decoupled"].sort_values("banks")
            b = s[s.stall_model == "lockstep"].sort_values("banks")
            ax.plot(a.banks, a.steady_util, color=col[n], marker=mk[n], ms=3,
                    lw=1.4, label=f"N={n}", zorder=3)
            ax.plot(b.banks, b.steady_util, color=col[n], lw=0.9,
                    ls=(0, (3, 2)), zorder=2)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, 29)
        ax.set_xticks(range(0, 29, 4))
        st.threshold(ax, x=20, text=None)
        ax.set_title(PSUM_LABEL[opt], loc="left")
    for ax in axs[1]:
        ax.set_xlabel("BANKS")
    for ax in axs[:, 0]:
        ax.set_ylabel("Steady-state PE utilization")
    h, l = axs[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=st.INK2, lw=1.4),
          plt.Line2D([], [], color=st.INK2, lw=0.9, ls=(0, (3, 2)))]
    l += ["decoupled queues", "global stall, fixed addresses"]
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Where partial sums live sets the bank count (ResNet-50, no "
                 "DRAM)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig27_psum_options")


# ------------------------------------------------- tables of the extensions
def saturation_from_load(key, knee, main):
    """Table VI's 90% crossing without interpolating between N = 8 and 16.
    The model only accepts N that are multiples of 8, but the knee suite
    measures the same crossing at a fixed N by moving BANKS, as a load D/B.
    Utilization depends on N and BANKS only through D/B, so the measured
    load at 90% converts into the N at which a given B crosses 90%."""
    rows = []
    for wl in ex.COMPUTE_BOUND:
        M = ex.M_OF[wl]
        cross = []
        for n in ex.N_VALUES:
            s = knee[(knee.workload == wl) & (knee.N == n)] \
                .sort_values("banks")
            s = s[~aliasing(s) & (s.pred_dram_ceiling > 0.999)]
            hit = s[s.pe_util >= 0.9]
            if len(hit) and hit.banks.min() > s.banks.min():
                b1 = int(hit.banks.min())
                r1 = s[s.banks == b1].iloc[0]
                before = s[s.banks < b1].iloc[-1]
                frac = (0.9 - before.pe_util) / (r1.pe_util - before.pe_util)
                b_eq = before.banks + frac * (b1 - before.banks)
                cross.append(model.demand(n, M) / (8 * b_eq))
        load = float(np.mean(cross))
        for bk in ex.BANK_VALUES:
            B = 8 * bk
            ns = np.arange(1.0, 128.0, 0.01)
            D = (5 * ns * M + ns ** 2) / np.maximum(M, ns)
            n_load = float(ns[np.argmax(D / B > load)])
            d = main[(main.workload == wl) & (main.banks == bk)] \
                .sort_values("N")
            rows.append({
                "workload": wl, "BANKS": bk, "B (bytes/cycle)": B,
                "N at 90%, interpolated (Table VI)":
                    crossing(d.N, d.pe_util),
                "measured D/B at 90%": round(load, 3),
                "N at 90%, from the measured D/B": round(n_load, 1),
                "N at 90% (eq. 1-2)": round(model_n90(M, bk)[0], 1),
            })
    write_table(pd.DataFrame(rows), "saturation_from_load",
                "Where the scratchpad saturates, from the measured load",
                "Runs at N between multiples of 8 are not possible (see "
                "gem5_model/README.txt 6.9).  Interpolating linearly between "
                "N = 8 and N = 16 places the crossing too low, because "
                "utilization stays near 1 up to D = B and only then falls.  "
                "The knee suite measures the crossing directly at fixed N "
                "by moving BANKS one at a time (aliasing and DRAM-bound "
                "points excluded); this table converts that load to N "
                "through eq. (1).  At M >> N it is derived, not a run at "
                "that N.")
    key["saturation_from_load"] = {
        f"{r['workload']}_B{r['BANKS']}":
        r["N at 90%, from the measured D/B"] for r in rows}


def m_tile_rows(mt):
    """Each tiled run against the whole layer on-chip at the same clock."""
    rows = []
    tiled = mt[mt.spm_depth > 0]
    whole = mt[mt.spm_depth == 0]
    for r in tiled.itertuples():
        b = whole[(whole.workload == r.workload) & (whole.N == r.N) &
                  (whole.banks == r.banks) &
                  (whole.clock_hz == r.clock_hz)].iloc[0]
        rows.append({
            "workload": r.workload, "N": r.N, "BANKS": r.banks,
            "array MHz": r.clock_hz / 1e6,
            "whole layer KiB": round(b.spm_required_bytes / 1024, 1),
            "capacity KiB (512 words)":
                round(r.spm_capacity_bytes / 1024, 1),
            "M tile rows": r.m_tile or r.M, "M tiles": int(r.m_tiles),
            "N_out slice": r.n_tile or r.N_out,
            "slices": int(r.n_slices) if pd.notna(r.n_slices) else 1,
            "util, whole layer": b.pe_util,
            "util, tiled": r.pe_util,
            "model, tiled": r.pred_util,
            "layer time ms, whole layer": b.layer_time_s * 1e3,
            "layer time ms, tiled": r.layer_time_s * 1e3,
            "time ratio": r.layer_time_s / b.layer_time_s,
            "DRAM MiB, whole layer": b.dram_bytes / 2 ** 20,
            "DRAM MiB, tiled": r.dram_bytes / 2 ** 20,
        })
    return rows


def extension_tables(S, key):
    """Tables of the model options that are off by default."""
    main, knee = S["main"], S["knee"]
    rtlval, rtl_ideal = S["rtlval"], S["rtlval_idealdram"]
    rtl_lock = S.get("rtlval_lockstep")
    if knee is not None:
        saturation_from_load(key, knee, main)

    # ---- gem5 side of the gem5-vs-RTL table
    if rtlval is not None and rtl_ideal is not None:
        j = rtlval.merge(rtl_ideal, on=["workload", "N", "banks"],
                         suffixes=("_dram", "_ideal"))
        j["dram_stall_cycles"] = j.total_cycles_dram - j.total_cycles_ideal
        cols = ["workload", "N", "banks", "total_cycles_ideal",
                "pe_util_ideal", "total_cycles_dram", "dram_stall_cycles"]
        if rtl_lock is not None:
            j = j.merge(rtl_lock[["workload", "N", "banks", "total_cycles",
                                  "pe_util"]]
                        .rename(columns={"total_cycles": "lockstep_cycles",
                                         "pe_util": "lockstep_util"}),
                        on=["workload", "N", "banks"])
            cols += ["lockstep_cycles", "lockstep_util"]
        for suite, name in (("rtlcfg_rtlval", "sdp"),
                            ("rtlcfg_rtlval_lockstep", "sdp_lockstep"),
                            ("rtlcfg_nodb_rtlval_lockstep",
                             "sdp_one_array_lockstep")):
            if S.get(suite) is not None:
                j = j.merge(S[suite][["workload", "N", "banks",
                                      "total_cycles", "pe_util"]]
                            .rename(columns={
                                "total_cycles": f"{name}_cycles",
                                "pe_util": f"{name}_util"}),
                            on=["workload", "N", "banks"])
                cols += [f"{name}_cycles", f"{name}_util"]
        write_table(j[cols].sort_values(["workload", "N", "banks"]),
                    "rtl_gem5_columns",
                    "gem5 cycle counts for the gem5-vs-RTL comparison",
                    "The testbench preloads the weights, so "
                    "total_cycles_ideal (no DRAM) is the column to compare "
                    "with RTL cycles; total_cycles_dram includes weight "
                    "refill from DDR4.  lockstep_cycles is the same layer "
                    "with the lockstep stall model, also without DRAM: the "
                    "comparison point if the RTL implements a global stall.  "
                    "sdp_* are the RTL's configuration (simple-dual-port "
                    "banks), and sdp_lockstep_cycles is the column closest "
                    "to rtl/src today, which has no queues; "
                    "sdp_one_array_lockstep_cycles is that with one weight "
                    "register array, if the RTL keeps it.")
        key["rtlval_dram_stall_cycles"] = {
            wl: {"min": float(v.min()), "max": float(v.max())} for wl, v in
            j.groupby("workload").dram_stall_cycles}

    # ---- lockstep
    lock, check = S["lockstep"], S["lockcheck"]
    if lock is not None:
        j = lock.merge(main, on=["workload", "N", "banks"],
                       suffixes=("_lock", "_dec"))
        t = j[["workload", "N", "banks", "pe_util_dec", "steady_util_dec",
               "pe_util_lock", "steady_util_lock",
               "pred_lockstep_staircase_lock",
               "pred_lockstep_fixed_addr_lock"]].rename(columns={
                   "pe_util_dec": "decoupled",
                   "steady_util_dec": "decoupled steady",
                   "pe_util_lock": "lockstep",
                   "steady_util_lock": "lockstep steady",
                   "pred_lockstep_staircase_lock": "staircase",
                   "pred_lockstep_fixed_addr_lock": "address-map model"})
        write_table(t.sort_values(["workload", "N", "banks"]),
                    "lockstep_vs_decoupled",
                    "Decoupled vs. lockstep stall model (main grid)",
                    "steady = steadyStateUtilization, the window from the "
                    "first row's last partial sum to the last row entering "
                    "the array.  staircase = 1 / ceil(words / BANKS), the "
                    "words spread evenly.  address-map model = the "
                    "staircase's busiest bank worked out from gem5's fixed "
                    "per-stream addresses (model.lockstep_fixed_address).")
        key["lockstep_steady_resnet50"] = {
            f"N{r.N}_B{r.banks}": float(r.steady_util_lock)
            for r in j[j.workload == "resnet50_conv3x"].itertuples()}
    if check is not None:
        c = check.copy()
        opt = c[c.arbitration == "lane"]
        rows = []
        for r in opt.itertuples():
            rows.append({
                "kind": "--arbitration lane --lookahead 1 --psum-queue 1",
                "N": r.N, "BANKS": r.banks, "steady util": r.steady_util,
                "staircase": model.lockstep_staircase(r.N, r.banks),
                "address-map model": ""})
        for r in c[c.stall_model == "lockstep"].itertuples():
            rows.append({
                "kind": "--stall-model lockstep --ideal-dram",
                "N": r.N, "BANKS": r.banks, "steady util": r.steady_util,
                "staircase": r.pred_lockstep_staircase,
                "address-map model": r.pred_lockstep_fixed_addr})
        write_table(pd.DataFrame(rows), "lockstep_check",
                    "Lockstep steady state against the staircase (ResNet-50)",
                    "The existing options cannot give lockstep: with a "
                    "lookahead of 1 the read round trip caps every point at "
                    "1/3.  The new stall model matches the staircase where "
                    "gem5's fixed addresses happen to spread the words "
                    "evenly and falls below it elsewhere; the address map "
                    "accounts for the difference.")

    # ---- simple dual port
    ports, bankrule = S["bankports"], S["bankrule"]
    if ports is not None and knee is not None:
        rows = []
        for wl in ex.COMPUTE_BOUND:
            M = ex.M_OF[wl]
            for n in ex.N_VALUES:
                a = knee[(knee.workload == wl) & (knee.N == n)]
                b = ports[(ports.workload == wl) & (ports.N == n) &
                          (ports.psum_queue == 8)]

                def at(d, bk):
                    r = d[d.banks == bk]
                    return float(r.pe_util.iloc[0]) if len(r) else \
                        float("nan")

                def fewest(d, thr):
                    v = d[d.pe_util >= thr * d.ideal_util]
                    return int(v.banks.min()) if len(v) else "none"
                rule1 = math.ceil(model.demand(n, M) / 8)
                half = n // 2
                rows.append({
                    "workload": wl, "N": n,
                    "1rw rule ceil(D/8)": rule1,
                    "util 1rw at rule": at(a, rule1),
                    "1r1w: ceil(4N/8)": half,
                    "util 1r1w at N/2": at(b, half),
                    "util 1rw at N/2": at(a, half),
                    "1r1w rule with DMA fill":
                        model.banks_needed_1r1w(n, M),
                    "fewest 1rw for 99%": fewest(a, 0.99),
                    "fewest 1r1w for 99%": fewest(b, 0.99),
                    "fewest 1rw for 90%": fewest(a, 0.9),
                    "fewest 1r1w for 90%": fewest(b, 0.9),
                })
        write_table(pd.DataFrame(rows), "bank_rule_1r1w",
                    "The bank rule with simple-dual-port banks",
                    "With one read and one write port per bank the write "
                    "stream sets the bank count: BANKS = max(ceil(D_read/8), "
                    "ceil(D_write/8)), N/2 for M >> N, against ceil(D/8) "
                    "with one shared port.  'with DMA fill' also counts the "
                    "weight DMA on the write side, which rounds up one bank "
                    "at M >> N and is what model.banks_needed_1r1w returns.")
        key["bank_rule_1r1w"] = {
            f"{r['workload']}_N{r['N']}": {
                "util_at_half_N": r["util 1r1w at N/2"],
                "fewest_for_99pct": r["fewest 1r1w for 99%"]}
            for r in rows}

    # ---- weight store
    ws = S["weightstore"]
    if ws is not None:
        t = ws[["workload", "M", "N", "banks", "clock_hz", "weight_store",
                "pe_util", "ideal_util", "pred_util", "spm_bw_total",
                "dram_bus_util"]].copy()
        t["util / stall-free"] = t.pe_util / t.ideal_util
        t["array MHz"] = t.clock_hz / 1e6
        t = t.drop(columns=["clock_hz"])
        write_table(t.sort_values(["M", "array MHz", "N", "banks",
                                   "weight_store"]), "weight_store",
                    "Bank crossings of a weight tile",
                    "spm: DMA write + array read, 2 crossings, D_fill "
                    "(2N + 5 at M = 1).  fill_bypass: array read only, 1 "
                    "crossing, eq. (1) as the proposal writes it (N + 5).  "
                    "separate: a staging buffer outside the banks, 0 "
                    "crossings (5 bytes/cycle at M = 1).  250 MHz keeps "
                    "DRAM out of the picture.")
        v = ws[(ws.workload == "reuse_M1") & (ws.N == 8) &
               (ws.clock_hz == 250e6)]
        key["weight_store_M1_N8_util_over_stall_free"] = {
            f"{r.weight_store}_B{r.banks}": float(r.pe_util / r.ideal_util)
            for r in v.itertuples()}

    # ---- XOR bank hash
    bh = S["bankhash"]
    if bh is not None:
        t = bh.pivot_table(index=["workload", "N", "banks"],
                           columns="bank_hash", values="pe_util") \
            .reset_index()
        t["stall-free"] = [float(bh[(bh.workload == r.workload) &
                                    (bh.N == r.N) & (bh.banks == r.banks)]
                                 .ideal_util.iloc[0])
                           for r in t.itertuples()]
        t["case"] = np.where(t.banks == t.N - 1, "BANKS = N-1 (aliasing)",
                             "rule ceil(D/8)")
        write_table(t.sort_values(["case", "workload", "N"]), "bank_hash",
                    "XOR skew hash vs. modulo bank map",
                    "xor: with row r = w div BANKS, bank = (w + (r ^ (r >> "
                    "s))) mod BANKS, s = ceil(log2 BANKS).  One-to-one for "
                    "any BANKS, oldest-first queues, 8-entry write buffer.")
        al = t[t.case == "BANKS = N-1 (aliasing)"]
        key["bank_hash_aliasing"] = {
            f"{r.workload}_N{r.N}": {"modulo": float(r.modulo),
                                     "xor": float(r.xor)}
            for r in al.itertuples()}

    # ---- bank depth and M tiling
    mt = S["mtile"]
    if mt is not None:
        rows = m_tile_rows(mt)
        write_table(pd.DataFrame(rows), "m_tiling",
                    "512-word banks at the rule's bank counts",
                    "--spm-depth 512 --m-tile auto: the largest M tile that "
                    "fits, with at most a quarter of the capacity for weight "
                    "prefetch.  Every weight tile is reloaded from DRAM for "
                    "each M tile, and the reuse M of eq. (1) becomes the "
                    "tile height, so the rule's bank counts, sized for the "
                    "whole layer's M, no longer match the demand.")
        key["m_tiling"] = {
            f"{r['workload']}_N{r['N']}_{r['array MHz']:.0f}MHz": {
                "m_tile": int(r["M tile rows"]),
                "util_whole": float(r["util, whole layer"]),
                "util_tiled": float(r["util, tiled"]),
                "time_ratio": float(r["time ratio"])}
            for r in rows}

    # ---- host-side options
    hx = S["hostext"]
    if hx is not None:
        rows = []
        for r in hx.itertuples():
            b = main[(main.workload == r.workload) & (main.N == r.N) &
                     (main.banks == r.banks)].iloc[0]
            rows.append({
                "workload": r.workload, "N": r.N, "BANKS": r.banks,
                "option": "--prepacked-weights" if "prepack" in r.tag
                else "--split-host-stats",
                "total cycles, default": b.total_cycles,
                "total cycles": r.total_cycles,
                "layer time us, default": b.layer_time_s * 1e6,
                "layer time us": r.layer_time_s * 1e6,
                "host CPI, default (polling)": b.host_cpi,
                "host CPI (setup)" if "split" in r.tag else "host CPI":
                    r.host_cpi,
            })
        write_table(pd.DataFrame(rows), "host_options",
                    "Host-side options against the default",
                    "--split-host-stats: host CPI of the register setup "
                    "instead of the polling loop.  --prepacked-weights: "
                    "the weights arrive in the DRAM image, so the simulation "
                    "skips packing them (the setup stats dump is not the "
                    "measured region).")

    # ---- DDR5-8400 on a wider memory bus
    dw, dram = S["dramwide"], S["dram"]
    if dw is not None and dram is not None:
        rows = []
        for r in dw.itertuples():
            ref = main if r.mem_type == "DDR4_2400_8x8" else dram
            b = ref[(ref.workload == r.workload) & (ref.N == r.N) &
                    (ref.banks == r.banks)]
            b = b[b.mem_type == r.mem_type].iloc[0]
            rows.append({
                "workload": r.workload, "N": r.N, "DRAM": r.mem_type,
                "util, 16-byte bus": b.pe_util, "util, 64-byte bus": r.pe_util,
                "DRAM read GB/s, 16-byte bus": b.dram_read_bw_GBs,
                "DRAM read GB/s, 64-byte bus": r.dram_read_bw_GBs,
                "DRAM peak GB/s": r.dram_peak_GBs,
                "sustained / peak, 64-byte bus":
                    r.dram_read_bw_GBs / r.dram_peak_GBs,
            })
        write_table(pd.DataFrame(rows), "dram_wide_bus",
                    "DDR5-8400 on a wider memory bus",
                    "gem5's 16-byte SystemXBar passes one 64-byte packet "
                    "every 5 of its 2 GHz cycles (25.6 GB/s), below what "
                    "DDR5-8400 can supply.  --membus-width 64 removes that "
                    "limit.  DDR4-2400 is included to show the wider bus "
                    "changes little where the DRAM was already the limit.")


def option_tables(S, key, L):
    """The RTL team's open choices: where partial sums live (R4), reading
    them back (R1), one weight register array (R2), and tiling (R3)."""
    # ---- where partial sums live
    po = S.get("psumopts")
    if po is not None:
        d = po.copy()
        d["option"] = psum_option(d)
        d["frac"] = d.pe_util / d.ideal_util
        rows, grid = [], []
        for wl in ex.COMPUTE_BOUND:
            M, K = ex.M_OF[wl], {"resnet50_conv3x": 1152, "bert_ffn1": 768}[wl]
            for n in ex.N_VALUES:
                T = max(M, n)
                for opt, o in ex.PSUM_OPTIONS.items():
                    s_ = d[(d.workload == wl) & (d.N == n) &
                           (d.option == opt)]
                    dec = s_[s_.stall_model == "decoupled"] \
                        .sort_values("banks")
                    lock = s_[s_.stall_model == "lockstep"] \
                        .sort_values("banks")
                    ports = o.get("bank_ports", "1rw")
                    rmw = o.get("psum_rmw", False)
                    store = o.get("psum_store", "spm")
                    if ports == "1r1w":
                        avg = max(model.port_demand(
                            n, M, "fill_bypass", True, True,
                            model.rmw_fraction(n, K) if rmw else 0.0, store))
                    else:
                        avg = (5 * n * M + n * n) / T

                    def fewest(f, thr):
                        v = f[f.frac >= thr]
                        return int(v.banks.min()) if len(v) else "none swept"
                    one_tile = max(
                        model.ceil_div(n, 8) + (n // 2 if rmw else 0),
                        0 if store == "external" else n // 2)
                    if ports == "1rw":
                        one_tile = model.lockstep_words(n)
                    if store == "external":
                        one_tile = 2 * model.ceil_div(n, 8)
                    peak = model.lockstep_peak_words(n, ports, rmw, store)

                    def at(f, b):
                        v = f[f.banks == b]
                        return float(v.frac.iloc[0]) if len(v) else float("nan")
                    rows.append({
                        "workload": wl, "N": n, "option": opt,
                        "average demand, banks": math.ceil(avg / 8),
                        "gem5 decoupled, fewest for 99%": fewest(dec, 0.99),
                        "bank simulator (one tile)": one_tile,
                        "peak of back-to-back tiles": peak,
                        "gem5 global stall, fewest for 99%":
                            fewest(lock, 0.99),
                        "global stall, util at the simulator's count":
                            at(lock, one_tile),
                        "global stall, util at peak count": at(lock, peak),
                    })
                    for b in ex.BANK_VALUES:
                        grid.append({"workload": wl, "N": n, "BANKS": b,
                                     "option": opt,
                                     "decoupled": at(dec, b),
                                     "global stall": at(lock, b)})
        t = pd.DataFrame(rows)
        write_table(t, "psum_options",
                    "Where partial sums live, side by side (no DRAM)",
                    "A: accumulator banks in the scratchpad, one port per "
                    "bank, written once.  B: the same with SDP banks.  B+rmw: "
                    "SDP, read back through the read port and rewritten.  C: "
                    "outside the counted banks.  'Bank simulator' is the "
                    "evenly spread count for one tile as bank-sim reports it, "
                    "from a formula here (the page was not run).  'Peak' adds "
                    "the next tile's weights loading alongside.  gem5's "
                    "global stall uses fixed per-stream addresses.  Values "
                    "are utilization / stall-free.")
        write_table(pd.DataFrame(grid), "psum_options_grid",
                    "The options at the main grid's bank counts (no DRAM)",
                    "Utilization / stall-free.")
        L += ["", "## Where partial sums live (no DRAM)", ""]
        for r in rows:
            if r["workload"] == "resnet50_conv3x":
                L.append(f"* N={r['N']} option {r['option']}: gem5 decoupled "
                         f"needs {r['gem5 decoupled, fewest for 99%']} banks "
                         f"for 99%; global stall with fixed addresses, "
                         f"{r['gem5 global stall, fewest for 99%']}.")
        key["psum_options"] = {
            f"{r['workload']}_N{r['N']}_{r['option']}": {
                "decoupled_99": r["gem5 decoupled, fewest for 99%"],
                "lockstep_99": r["gem5 global stall, fewest for 99%"]}
            for r in rows}

    # ---- partial sums read back
    rm_, rk_ = S.get("rtlcfg_rmw_main"), S.get("rtlcfg_rmw_knee")
    if rm_ is not None and rk_ is not None:
        lock = S.get("rtlcfg_rmw_lockstep")
        base = pd.concat([S["rtlcfg_knee"], S["rtlcfg_main"]]) \
            .drop_duplicates("tag")
        rows = []
        for wl in ex.COMPUTE_BOUND:
            for n in ex.N_VALUES:
                r5 = math.ceil(5 * n / 8)
                k = rk_[(rk_.workload == wl) & (rk_.N == n)]
                b_ = base[(base.workload == wl) & (base.N == n)]
                lk = lock[(lock.workload == wl) & (lock.N == n)]

                def at(f, b):
                    v = f[f.banks == b]
                    return float(v.pe_util.iloc[0]) if len(v) else float("nan")
                v99 = k[k.pe_util >= 0.99 * k.ideal_util]
                rows.append({
                    "workload": wl, "N": n,
                    "N/2 banks, written once": at(b_, n // 2),
                    "N/2 banks, read back": at(k, n // 2),
                    "ceil(5N/8) banks": r5,
                    "ceil(5N/8), read back": at(k, r5),
                    "fewest for 99%, read back":
                        int(v99.banks.min()) if len(v99) else "none (DRAM)",
                    "global stall, ceil(5N/8)": at(lk, r5),
                    "global stall, ceil(3N/4)":
                        at(lk, math.ceil(3 * n / 4)),
                })
        t = pd.DataFrame(rows)
        write_table(t, "rtl_config_rmw",
                    "Partial sums read back through the read port",
                    "--psum-rmw on the RTL's configuration: accumulating "
                    "across K tiles reads the old partial sum before writing "
                    "the new one, except on a row's first K tile.  The read "
                    "port carries N activation bytes plus 4N old partial-sum "
                    "bytes.  Global stall with fixed per-stream addresses.")
        h = pd.concat([rm_, rk_]).drop_duplicates("tag")
        h = h[h.workload.isin(ex.COMPUTE_BOUND) &
              (h.pred_dram_ceiling >= 0.999) & ~aliasing(h)]
        load = h.model_D_port / h.B_bytes_per_cycle
        frac = h.pe_util / h.ideal_util
        err = (frac - 1 / load)[load > 1].abs()
        L += ["", "## Partial sums read back (--psum-rmw)", "",
              f"* {len(h)} runs: below D_read = B, minimum utilization / "
              f"stall-free {frac[load <= 1].min():.3f}; above it, |frac - "
              f"B/D_read| mean {err.mean():.3f}, max {err.max():.3f}."]
        for r in rows:
            L.append(f"* {r['workload']} N={r['N']}: N/2 banks "
                     f"{r['N/2 banks, written once']:.3f} -> "
                     f"{r['N/2 banks, read back']:.3f}; "
                     f"{r['ceil(5N/8) banks']} banks "
                     f"{r['ceil(5N/8), read back']:.3f}; global stall there "
                     f"{r['global stall, ceil(5N/8)']:.3f}.")
        key["rtl_config_rmw"] = {
            f"{r['workload']}_N{r['N']}": {
                "half_N": r["N/2 banks, read back"],
                "five_eighths": r["ceil(5N/8), read back"],
                "lockstep_five_eighths": r["global stall, ceil(5N/8)"]}
            for r in rows}
        fits = [fit_row(n, S[n]) for n in ("rtlcfg_rmw_main",
                                           "rtlcfg_rmw_knee")]
        key["rtl_config_rmw_fit"] = {
            r["suite"]: {k: r[k] for k in r if k != "suite"} for r in fits}

    # ---- one weight register array
    nm, nk = S.get("rtlcfg_nodb_main"), S.get("rtlcfg_nodb_knee")
    if nm is not None and nk is not None:
        h = pd.concat([nm, nk]).drop_duplicates("tag")
        h = h[h.workload.isin(ex.COMPUTE_BOUND)].copy()
        h["cap"] = h.M / (h.M + h.N)
        s_ = h[(h.pred_dram_ceiling >= 0.999) & ~aliasing(h)]
        load = s_.model_D_port / s_.B_bytes_per_cycle
        frac = s_.pe_util / s_.ideal_util
        err = (frac - 1 / load)[load > 1].abs()
        rows = []
        nl = S.get("rtlcfg_nodb_lockstep")
        for wl in ex.COMPUTE_BOUND:
            M = ex.M_OF[wl]
            for n in ex.N_VALUES:
                b = ex.rtl_rule(n, M)
                d_ = h[(h.workload == wl) & (h.N == n) & (h.banks == b)]
                l_ = nl[(nl.workload == wl) & (nl.N == n) & (nl.banks == b)]
                rows.append({
                    "workload": wl, "N": n, "BANKS": b,
                    "M / (M + N)": M / (M + n),
                    "util, decoupled": float(d_.pe_util.iloc[0]),
                    "util, global stall": float(l_.pe_util.iloc[0]),
                })
        write_table(pd.DataFrame(rows), "rtl_config_nodb",
                    "One weight register array, at the busier-port rule",
                    "rtlcfg_nodb_*: --no-double-buffer on the RTL's "
                    "configuration.  T = M + N, so utilization is capped at "
                    "M / (M + N) whatever BANKS is.")
        L += ["", "## One weight register array (rtlcfg_nodb_*)", "",
              f"* Past the knee, |utilization / stall-free - B/D| on the "
              f"busier port: mean {err.mean():.3f}, max {err.max():.3f}; "
              f"below it the minimum is {frac[load <= 1].min():.3f}, relative "
              f"to the lower stall-free value."]
        key["rtl_config_nodb"] = {
            f"{r['workload']}_N{r['N']}": r["util, decoupled"] for r in rows}

    # ---- tiling along M and N_out
    nt, mt = S.get("rtlcfg_ntile"), S.get("rtlcfg_mtile")
    if nt is not None and mt is not None:
        rows = []
        m_only = {(r["workload"], r["N"], r["array MHz"]): r
                  for r in m_tile_rows(mt)}
        for r in m_tile_rows(nt):
            old = m_only.get((r["workload"], r["N"], r["array MHz"]))
            rows.append({
                "workload": r["workload"], "N": r["N"], "BANKS": r["BANKS"],
                "array MHz": r["array MHz"],
                "M tile": r["M tile rows"], "N_out slice": r["N_out slice"],
                "util, whole layer": r["util, whole layer"],
                "util, M tiles only":
                    f"{old['util, tiled']:.4f}" if old else "does not fit",
                "util, M and N_out tiles": r["util, tiled"],
                "model": r["model, tiled"],
                "time ratio, M tiles only":
                    old["time ratio"] if old else "",
                "time ratio, M and N_out tiles": r["time ratio"],
                "DRAM MiB, whole layer": r["DRAM MiB, whole layer"],
                "DRAM MiB, M and N_out tiles": r["DRAM MiB, tiled"],
            })
        write_table(pd.DataFrame(rows), "rtl_config_n_tiling",
                    "512-word banks, tiled along M only or along M and N_out",
                    "Busier-port rule's banks.  N_out slices are the smallest "
                    "legal size.  A stays in the scratchpad across the slices "
                    "of an M tile, but A alone can exceed the capacity, so "
                    "M tiles are still needed.")
        L += ["", "## Tiling along N_out as well (rtlcfg_ntile)", ""]
        for r in rows:
            if r["array MHz"] == 250:
                shown = r["util, M tiles only"]
                if shown != "does not fit":
                    shown = f"{float(shown):.3f}"
                L.append(f"* {r['workload']} N={r['N']} BANKS={r['BANKS']}, "
                         f"250 MHz, M tile {r['M tile']}, slice "
                         f"{r['N_out slice']}: {r['util, whole layer']:.3f} "
                         f"whole, {shown} M only, "
                         f"{r['util, M and N_out tiles']:.3f} both, layer "
                         f"time x{r['time ratio, M and N_out tiles']:.2f}.")
        key["rtl_config_n_tiling"] = {
            f"{r['workload']}_N{r['N']}_{r['array MHz']:.0f}MHz":
            r["util, M and N_out tiles"] for r in rows}
    ns1 = S.get("ntile")
    if ns1 is not None:
        write_table(pd.DataFrame(m_tile_rows(ns1)), "n_tiling_1rw",
                    "512-word banks tiled along M and N_out, one shared port",
                    "ceil(D/8) banks.")


def rtl_config_tables(S, key):
    """Every hypothesis again under the RTL's configuration, and
    tables/hypotheses_rtl_config.md.  Nothing here feeds the older tables."""
    if S.get("rtlcfg_main") is None or S.get("rtlcfg_knee") is None:
        return
    main, knee = S["main"], S["knee"]
    rm, rk = S["rtlcfg_main"], S["rtlcfg_knee"]
    cb = rtl_frame(S)
    L = []  # lines of the hypothesis check

    def scratch(d):
        """Runs only the banks can limit (no DRAM limit, no lane aliasing)."""
        return d[(d.pred_dram_ceiling >= 0.999) & ~aliasing(d)]

    # ---- model fit
    fits = [fit_row(n, S[n]) for n in
            ("rtlcfg_main", "rtlcfg_knee", "rtlcfg_rulem", "rtlcfg_reuse",
             "rtlcfg_gemv", "rtlcfg_clock", "rtlcfg_extend")
            if S.get(n) is not None]
    t = pd.DataFrame(fits).rename(columns={
        "R^2": "R^2, busier port",
        "mean |err|": "mean |err|, busier port",
        "R^2, eq. (1) as written": "R^2, proposal's eq. (1)",
        "mean |err|, eq. (1) as written": "mean |err|, proposal's eq. (1)"})
    write_table(t, "rtl_config_fit", "Model vs gem5, RTL's configuration",
                "Busier port: min(1, B / max(D_read, D_write)).  Proposal's "
                "eq. (1): min(1, B / D) with one D for all traffic.")
    key["rtl_config_fit"] = {r["suite"]: {k: r[k] for k in r if k != "suite"}
                             for r in fits}

    # ---- H1
    L += ["# Hypothesis checks, RTL's configuration (auto-generated)", "",
          "Simple-dual-port banks, weights crossing the banks once "
          "(rtl/src on the dev_hill branch).  gem5 only: the RTL cannot run "
          "these layers yet.", "",
          "## H1: above 90% until delivered bandwidth reaches B, then ~B/D",
          ""]
    h1 = scratch(cb)
    h1 = h1.assign(frac=h1.pe_util / h1.ideal_util)
    res = {}
    for name, col in (("as the proposal writes it, D = eq. (1)", "model_D"),
                      ("on the busier port, D = max(D_read, D_write)",
                       "model_D_port")):
        load = h1[col] / h1.B_bytes_per_cycle
        below, above = h1[load <= 1], h1[load > 1]
        err = (above.frac - 1 / load[load > 1]).abs()
        # Runs with D > B yet still above 90%: the knee is not where it says.
        early = above[(above.frac >= 0.9) & (load[load > 1] > 1 / 0.9)]
        res[col] = dict(below_min=float(below.frac.min()),
                        above_mean_err=float(err.mean()),
                        above_max_err=float(err.max()),
                        early=len(early), runs=len(h1))
        L += [f"* Load {name}: {len(below)} runs with D <= B, minimum "
              f"utilization / stall-free {below.frac.min():.3f}.  "
              f"{len(above)} runs with D > B: |utilization / stall-free - "
              f"B/D| mean {err.mean():.3f}, max {err.max():.3f}.  Runs still "
              f"at or above 90% past the 90% point the formula predicts: "
              f"{len(early)}."]
    key["rtl_config_h1"] = res

    # The knee, one bank at a time, as a load on the busier port.
    cross, rows = {}, []
    for wl in ex.COMPUTE_BOUND:
        vals = []
        for n in ex.N_VALUES:
            s_ = scratch(rk[(rk.workload == wl) & (rk.N == n)]) \
                .sort_values("banks")
            hit = s_[s_.pe_util >= 0.9]
            if len(hit) and hit.banks.min() > s_.banks.min():
                b1 = int(hit.banks.min())
                r1 = s_[s_.banks == b1].iloc[0]
                r0 = s_[s_.banks < b1].iloc[-1]
                b_eq = r0.banks + (0.9 - r0.pe_util) / \
                    (r1.pe_util - r0.pe_util) * (b1 - r0.banks)
                vals.append(float(r1.model_D_port) / (8 * b_eq))
        cross[wl] = float(np.mean(vals))
        L.append(f"* {wl}: 90% crossing at max(D_read, D_write) / B = "
                 f"{min(vals):.3f}-{max(vals):.3f} (the formula: 1.11).")
    key["rtl_config_crossing"] = cross

    # ---- the research question
    for wl in ex.COMPUTE_BOUND:
        M = ex.M_OF[wl]
        for bk in ex.BANK_VALUES:
            B = 8 * bk
            d0 = main[(main.workload == wl) & (main.banks == bk)] \
                .sort_values("N")
            d1 = rm[(rm.workload == wl) & (rm.banks == bk)].sort_values("N")
            d2 = S["rtlcfg_lockstep"]
            d2 = d2[(d2.workload == wl) & (d2.banks == bk)].sort_values("N")
            ns = np.arange(1.0, 200.0, 0.01)
            T = np.maximum(M, ns)
            D_port = np.maximum((ns * M + ns ** 2) / T, 4 * ns * M / T)
            over = D_port / B > cross[wl]
            rows.append({
                "workload": wl, "BANKS": bk, "B (bytes/cycle, per port)": B,
                "util N=8": float(d1[d1.N == 8].pe_util.iloc[0]),
                "util N=16": float(d1[d1.N == 16].pe_util.iloc[0]),
                "util N=24": float(d1[d1.N == 24].pe_util.iloc[0]),
                "util N=32": float(d1[d1.N == 32].pe_util.iloc[0]),
                "N at 90%, published (1 shared port)":
                    crossing(d0.N, d0.pe_util),
                "N at 90%, interpolated": crossing(d1.N, d1.pe_util),
                "N at 90%, from the measured load":
                    round(float(ns[np.argmax(over)]), 1) if over.any() else "",
                "global stall: util N=8": float(d2[d2.N == 8].steady_util
                                                .iloc[0]),
                "global stall: util N=16": float(d2[d2.N == 16].steady_util
                                                 .iloc[0]),
                "global stall: util N=24": float(d2[d2.N == 24].steady_util
                                                 .iloc[0]),
                "global stall: util N=32": float(d2[d2.N == 32].steady_util
                                                 .iloc[0]),
            })
    write_table(pd.DataFrame(rows), "rtl_config_saturation",
                "Where the scratchpad saturates, RTL's configuration",
                "Decoupled utilization unless the column says global stall "
                "(steady state).  'From the measured load' converts the "
                "knee sweep's crossing on the busier port through "
                "max(D_read, D_write); it is derived, not a run at that N.")
    key["rtl_config_saturation"] = {
        f"{r['workload']}_B{r['BANKS']}":
        r["N at 90%, from the measured load"] for r in rows}

    # ---- H3
    rows = []
    lock = S["rtlcfg_lockstep"]
    for wl in ex.COMPUTE_BOUND:
        M = ex.M_OF[wl]
        for n in ex.N_VALUES:
            s_ = rk[(rk.workload == wl) & (rk.N == n)]
            ceil_d = math.ceil(model.demand(n, M) / 8)
            r = ex.rtl_rule(n, M)

            def at(bk, d=s_):
                v = d[d.banks == bk]
                return float(v.pe_util.iloc[0]) if len(v) else float("nan")
            lk = lock[(lock.workload == wl) & (lock.N == n)]
            rows.append({
                "workload": wl, "N": n, "stall-free":
                float(s_.ideal_util.iloc[0]),
                "ceil(D/8)": ceil_d, "util at ceil(D/8)": at(ceil_d),
                "util at 25% fewer": at(math.floor(0.75 * ceil_d)),
                "busier-port rule": r, "util at rule": at(r),
                "util at rule, 25% fewer": at(math.floor(0.75 * r)),
                "global stall, util at rule": at(r, lk),
                "BRAM36 saved against ceil(D/8)": ceil_d - r,
            })
    t = pd.DataFrame(rows)
    write_table(t, "rtl_config_rule", "The bank rule, RTL's configuration",
                "ceil(D/8) is the proposal's rule; the busier-port rule is "
                "max(ceil(D_read/8), ceil(D_write/8)).  One BRAM36 per bank; "
                "a bandwidth count, not a Vivado number.")
    bound = t[~((t.workload == "bert_ffn1") & (t.N == 32))]
    L += ["", "## H3: the bank rule restores utilization", "",
          f"* ceil(D/8) banks (excluding the DRAM-bound BERT FFN1 N=32): "
          f"utilization {bound['util at ceil(D/8)'].min():.3f}-"
          f"{bound['util at ceil(D/8)'].max():.3f}.",
          f"* 25% fewer than ceil(D/8): {bound['util at 25% fewer'].min():.3f}"
          f"-{bound['util at 25% fewer'].max():.3f} (1 shared port: 0.66-0.80)"
          ", so ceil(D/8) is enough but no longer the smallest count.",
          f"* Busier-port rule: {bound['util at rule'].min():.3f}-"
          f"{bound['util at rule'].max():.3f}; 25% fewer: "
          f"{bound['util at rule, 25% fewer'].min():.3f}-"
          f"{bound['util at rule, 25% fewer'].max():.3f}.",
          f"* Global stall at the busier-port rule: "
          f"{bound['global stall, util at rule'].min():.3f}-"
          f"{bound['global stall, util at rule'].max():.3f}."]
    key["rtl_config_rule"] = {
        c: {"min": float(bound[c].min()), "max": float(bound[c].max())}
        for c in ("util at ceil(D/8)", "util at 25% fewer", "util at rule",
                  "util at rule, 25% fewer", "global stall, util at rule")}

    rulem = S.get("rtlcfg_rulem")
    if rulem is not None:
        rows = []
        for r_ in rulem.itertuples():
            rows.append(dict(
                M=r_.M, N=r_.N, BANKS=r_.banks,
                rule=ex.rtl_rule(r_.N, r_.M),
                ceil_D8=math.ceil(model.demand(r_.N, r_.M) / 8),
                frac=r_.pe_util / r_.ideal_util, pe_util=r_.pe_util))
        d = pd.DataFrame(rows)
        d = d[~aliasing(d.rename(columns={"BANKS": "banks"}))]
        d["kind"] = np.select(
            [d.BANKS == d.rule, d.BANKS == np.floor(0.75 * d.rule),
             d.BANKS == d.ceil_D8, d.BANKS == np.floor(0.75 * d.ceil_D8)],
            ["busier-port rule", "25% fewer than the rule", "ceil(D/8)",
             "25% fewer than ceil(D/8)"], default="other")
        write_table(d.sort_values(["M", "N", "BANKS"])
                    [["M", "N", "BANKS", "kind", "pe_util", "frac"]],
                    "rtl_config_rule_vs_reuse",
                    "The rules across reuse M, RTL's configuration",
                    "Weights free of DRAM.  frac = utilization / stall-free. "
                    "A count that fits two kinds is listed under the first.")
        g = d[d.kind != "other"].groupby("kind").frac.agg(["count", "min",
                                                           "mean", "max"])
        for k, v in g.iterrows():
            L.append(f"* Every reuse M, {k}: {int(v['count'])} runs, "
                     f"utilization / stall-free {v['min']:.3f}-{v['max']:.3f}"
                     f" (mean {v['mean']:.3f}).")
        key["rtl_config_rule_vs_reuse"] = {
            k: {c: float(v[c]) for c in ("count", "min", "mean", "max")}
            for k, v in g.iterrows()}

    # ---- H2
    gm = S.get("rtlcfg_gemv")
    if gm is not None:
        rows = []
        for (n, clk), v in gm.groupby(["N", "clock_hz"]):
            v = v.sort_values("banks")
            cap = float(v.ideal_util.iloc[0])
            top = float(v.pe_util.max())
            flat = v[v.pe_util >= 0.99 * top]
            rows.append({
                "N": n, "array MHz": clk / 1e6, "1/N cap": cap,
                "util at 32 banks": float(v[v.banks == 32].pe_util.iloc[0]),
                "util at 1 bank": float(v[v.banks == 1].pe_util.iloc[0]),
                "fewest BANKS within 1% of the top": int(flat.banks.min()),
                "D_read = N + 1 bytes/cycle": n + 1,
                "banks that needs": math.ceil((n + 1) / 8),
                "model": float(v[v.banks == 32].pred_util.iloc[0]),
            })
        t = pd.DataFrame(rows)
        write_table(t, "rtl_config_gemv", "M = 1, RTL's configuration",
                    "At 250 MHz DRAM is ample and utilization meets the 1/N "
                    "cap; at 2 GHz DRAM sets it.  Above a floor of "
                    "ceil((N+1)/8) banks BANKS stops mattering.")
        lo, hi = t[t["array MHz"] == 250], t[t["array MHz"] == 2000]
        L += ["", "## H2: for M = 1, 1/N and DRAM bandwidth, regardless of "
              "BANKS", "",
              f"* 250 MHz: utilization at 32 banks / the 1/N cap = "
              f"{(lo['util at 32 banks'] / lo['1/N cap']).min():.3f}-"
              f"{(lo['util at 32 banks'] / lo['1/N cap']).max():.3f}.",
              f"* 2 GHz: utilization at 32 banks / model = "
              f"{(hi['util at 32 banks'] / hi['model']).min():.3f}-"
              f"{(hi['util at 32 banks'] / hi['model']).max():.3f}.",
              "* Fewest banks within 1% of the top, 250 MHz: " +
              ", ".join(f"N={r.N}: {r._6}" for r in lo.itertuples()) +
              "; 2 GHz: " +
              ", ".join(f"N={r.N}: {r._6}" for r in hi.itertuples()) + "."]
        key["rtl_config_gemv"] = {
            f"N{r['N']}_{r['array MHz']:.0f}MHz":
            {k: r[k] for k in ("1/N cap", "util at 32 banks", "util at 1 bank",
                               "fewest BANKS within 1% of the top")}
            for r in rows}

    # ---- the global stall
    lk = S["rtlcfg_lockstep"]
    lk = lk[lk.workload == "resnet50_conv3x"]
    dec = cb[cb.workload == "resnet50_conv3x"]
    j = lk.merge(dec[["N", "banks", "steady_util"]], on=["N", "banks"],
                 suffixes=("", "_dec"), how="left")
    t = j[["N", "banks", "steady_util_dec", "steady_util",
           "pred_lockstep_staircase", "pred_lockstep_fixed_addr"]] \
        .rename(columns={"steady_util_dec": "decoupled steady",
                         "steady_util": "global stall steady",
                         "pred_lockstep_staircase": "staircase",
                         "pred_lockstep_fixed_addr": "address-map model"}) \
        .sort_values(["N", "banks"])
    write_table(t, "rtl_config_stall",
                "Decoupled vs. global stall, RTL's configuration (ResNet-50)",
                "staircase = 1 / max(ceil(ceil(N/8) / BANKS), "
                "ceil((N/2) / BANKS)).  Reads and writes no longer meet in "
                "a bank, so activation and partial-sum words stop colliding.")

    # ---- one weight register array, as the RTL has today
    nd = S.get("rtlcfg_nodb")
    if nd is not None:
        j = nd[nd.double_buffer == 0].merge(
            nd[nd.double_buffer == 1][["workload", "N", "banks", "pe_util"]],
            on=["workload", "N", "banks"], suffixes=("", "_db"))
        j["M/T"] = j.M / (j.M + j.N)
        write_table(j[["workload", "M", "N", "banks", "pe_util_db", "pe_util",
                       "M/T"]].rename(columns={
                           "pe_util_db": "shadow registers",
                           "pe_util": "one register array"})
                    .sort_values(["workload", "N", "banks"]),
                    "rtl_config_single_buffer",
                    "One weight register array, RTL's configuration",
                    "rtl/src has one weights register array, loaded after "
                    "reset, which is --no-double-buffer: T = M + N, so "
                    "utilization cannot pass M / (M + N).")
        cbn = j[j.workload.isin(ex.COMPUTE_BOUND) & (j.pred_dram_ceiling >=
                                                     0.999)]
        L += ["", "## What the RTL's one weight register array changes", "",
              "* Compute-bound runs below 90% purely from M / (M + N) < 0.9: "
              + (", ".join(f"{r.workload} N={r.N} BANKS={r.banks}: "
                           f"{r.pe_util:.3f}"
                           for r in cbn[cbn["M/T"] < 0.9].itertuples())
                 or "none") + "."]

    # ---- 512-word banks
    mt = S.get("rtlcfg_mtile")
    if mt is not None:
        rows = m_tile_rows(mt)
        for wl in ex.COMPUTE_BOUND:
            for n in ex.N_VALUES:
                b = ex.rtl_rule(n, ex.M_OF[wl])
                if not ex.m_tile_fits(wl, n, b):
                    rows.append({"workload": wl, "N": n, "BANKS": b,
                                 "array MHz": "any",
                                 "M tile rows": "does not fit"})
        write_table(pd.DataFrame(rows), "rtl_config_m_tiling",
                    "512-word banks, RTL's configuration",
                    "The busier-port rule's banks with --spm-depth 512 "
                    "--m-tile auto.  'does not fit' means one row and the "
                    "weight prefetch exceed BANKS x 512 words.  Note that "
                    "rtl/src currently sizes banks at N words, for one weight "
                    "tile only.")
        L += ["", "## Threat: bank depth (512 words)", ""]
        for r in rows:
            if r["array MHz"] == 250:
                L.append(f"* {r['workload']} N={r['N']} BANKS={r['BANKS']}, "
                         f"250 MHz, M tile {r['M tile rows']}: "
                         f"{r['util, whole layer']:.3f} -> "
                         f"{r['util, tiled']:.3f}, layer time x"
                         f"{r['time ratio']:.2f}.")
            elif r["array MHz"] == "any":
                L.append(f"* {r['workload']} N={r['N']} BANKS={r['BANKS']}: "
                         "does not fit.")

    option_tables(S, key, L)
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, "hypotheses_rtl_config.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("  results/tables/hypotheses_rtl_config.md")


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
    # "unique" stays the count of the default model's suites, so that it
    # compares with results made before the options existed.
    base = every[every.suite.isin(ex.BASELINE_SUITES)]
    key["runs"]["unique"] = int(base.tag.nunique())
    key["runs"]["unique_with_options"] = int(every.tag.nunique())
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
    extension_tables(S, key)
    rtl_config_tables(S, key)
    import json
    with open(os.path.join(DATA, "key_results.json"), "w") as f:
        json.dump(key, f, indent=2, default=str)
    print("  results/data/key_results.json")


# ------------------------------------------------------------------- main
FIGS = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8",
        "fig9", "fig10", "fig11", "fig12", "fig13", "fig14", "fig15",
        "fig16", "fig17", "fig18", "fig19", "fig20", "fig21", "fig22",
        "fig23", "fig24", "fig25", "fig26", "fig27", "tables"]

# The figures the paper uses, per source of results.  rtlcfg leads (the RTL's
# configuration); baseline is the conservative reference with one shared port.
PAPER_FIGS = {
    "rtlcfg": ["fig1", "fig2", "fig5", "fig12", "fig13", "fig15", "fig17",
               "fig18", "fig20", "fig25", "fig26", "fig27"],
    "baseline": ["fig1", "fig2", "fig5", "fig12", "fig13", "fig14", "fig15",
                 "fig17", "fig18", "fig20", "fig22", "fig23"],
}
# Which rtlcfg suite stands in for each suite those figures read.
PAPER_SUITES = {"main": ["rtlcfg_main"], "knee": ["rtlcfg_knee"],
                "idealdram": ["rtlcfg_idealdram"], "reuse": ["rtlcfg_reuse"],
                "clock": ["rtlcfg_clock", "rtlcfg_gemv"],
                "dram": ["rtlcfg_dram"], "rulem": ["rtlcfg_rulem"],
                "extend": ["rtlcfg_extend"]}


def paper_inputs(S):
    """S with the figures' suites replaced by the RTL's configuration, and
    D in those frames taken as the busier port's."""
    S = dict(S)
    for name, sources in PAPER_SUITES.items():
        parts = [S[n] for n in sources if S.get(n) is not None]
        S[name] = pd.concat(parts).assign(
            model_D=lambda d: d.model_D_port) if parts else None
    return S


def main():
    global FORMATS, FIG, TAB, PORT_LOAD
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=FIGS)
    ap.add_argument("--paper", choices=sorted(PAPER_FIGS),
                    help="the paper's figures only, from this source of "
                         "results, into results/paper/<source>/")
    ap.add_argument("--format", nargs="*", default=FORMATS,
                    choices=["png", "pdf", "svg"])
    args = ap.parse_args()
    FORMATS = args.format
    want = set(args.only or FIGS)
    st.setup()

    S = {name: load(name) for name in ex.SUITES}
    if args.paper:
        want = set(PAPER_FIGS[args.paper])
        FIG = os.path.join(RESULTS, "paper", args.paper)
        # Tables those figures write stay with them, not over results/tables.
        TAB = os.path.join(FIG, "tables")
        if args.paper == "rtlcfg":
            PORT_LOAD = True
            S = paper_inputs(S)
        print(f"the paper's figures from {args.paper}, into "
              f"{os.path.relpath(FIG, RESULTS)}/")
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
    if "fig23" in want and need("knee", "lockstep"):
        fig23_lockstep(main_df, S["knee"], S["lockstep"])
    if "fig24" in want and need("knee", "bankports"):
        fig24_bank_ports(S["knee"], S["bankports"])
    if "fig25" in want and need("rtlcfg_main", "rtlcfg_knee"):
        fig25_rtl_config(S)
    if "fig26" in want and need("rtlcfg_main", "rtlcfg_knee",
                                "rtlcfg_lockstep"):
        fig26_rtl_stall(S)
    if "fig27" in want and need("psumopts"):
        fig27_psum_options(S["psumopts"])
    if "tables" in want:
        tables(S)


if __name__ == "__main__":
    main()
