#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : style.py                                                      #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : One visual style for every figure                             #
#-----------------------------------------------------------------------------#
#
# Colours come from a validated reference palette (categorical slots in fixed
# order; a one-hue blue ramp for ordered quantities).  Rules used here:
#   * workloads (identity)        -> categorical slots 1-4, fixed order
#   * BANKS or N (ordered values) -> blue ramp, light = small, dark = large
#   * the analytical model        -> same colour as its series, dashed
#   * every series also has its own marker shape, and every multi-series
#     plot has a legend, so nothing is identified by colour alone
#   * one y-axis per panel; different measures go in different panels

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

WORKLOAD_COLOR = {
    "resnet50_conv3x": "#2a78d6",
    "bert_ffn1": "#eb6834",
    "bert_attn_qk": "#1baf7a",
    "llama7b_gemv": "#eda100",
}
WORKLOAD_MARKER = {
    "resnet50_conv3x": "o",
    "bert_ffn1": "s",
    "bert_attn_qk": "^",
    "llama7b_gemv": "D",
}
WORKLOAD_LABEL = {
    "resnet50_conv3x": "ResNet-50 conv3_x (M=784)",
    "bert_ffn1": "BERT-base FFN1 (M=128)",
    "bert_attn_qk": "BERT attention QK$^T$ (M=128)",
    "llama7b_gemv": "LLaMA-7B decode GEMV (M=1)",
}
WORKLOAD_SHORT = {
    "resnet50_conv3x": "ResNet-50",
    "bert_ffn1": "BERT FFN1",
    "bert_attn_qk": "BERT QK$^T$",
    "llama7b_gemv": "LLaMA GEMV",
}

# Blue ramp steps 250..700 (the lightest step still clears 2:1 on white).
_RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
_RAMP6 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
ORDINAL_MARKERS = ["o", "s", "^", "D", "v", "P"]


def ramp(values):
    """Map an ordered list of values to blue-ramp colours (light->dark)."""
    values = list(values)
    steps = _RAMP4 if len(values) <= 4 else _RAMP6
    if len(values) > len(steps):
        raise ValueError("too many ordered series for one ramp")
    # spread across the ramp so 2-3 series still get well-separated steps
    idx = [round(i * (len(steps) - 1) / max(1, len(values) - 1))
           for i in range(len(values))]
    return {v: steps[i] for v, i in zip(values, idx)}


def markers(values):
    return {v: ORDINAL_MARKERS[i % len(ORDINAL_MARKERS)]
            for i, v in enumerate(values)}


COMPONENT_COLOR = {  # stacked cycle breakdown (adjacent categorical slots)
    "useful (rows streaming)": "#2a78d6",
    "idle slots (fill/drain, M < N)": "#eb6834",
    "spm read stall": "#1baf7a",
    "spm write stall": "#eda100",
    "DMA stall": "#e87ba4",
}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "text.color": INK,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "svg.hashsalt": "eel5764",   # stable element ids in SVG output
    })


def threshold(ax, y=0.9, text="90% utilization", x=None):
    ax.axhline(y, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    if text:
        xl = ax.get_xlim()
        ax.annotate(text, (x if x is not None else xl[0], y),
                    xytext=(2, 2), textcoords="offset points", color=MUTED,
                    fontsize=7, va="bottom", ha="left")


def model_line(ax, x, y, color, **kw):
    ax.plot(x, y, color=color, lw=1.0, ls=(0, (3, 2)), alpha=0.9,
            zorder=2, **kw)


def end_label(ax, x, y, text, color=INK2, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=7, color=color, va="center", ha="left")
