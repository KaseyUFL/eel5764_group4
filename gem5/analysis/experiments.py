# EEL5764 Group 4 -- experiment definitions for the saturation study.
#
# Each suite is a list of run points.  collect.py runs them with gem5 and
# plot_curves.py turns the resulting CSVs into the proposal's figures.
# Unless a suite says otherwise, a run is host-driven: the RISC-V host
# program (RiscvTimingSimpleCPU, L1/L2, DDR4) launches the layer on the
# accelerator, and statistics cover exactly the layer.
#
# Proposal sweeps
#   suite       what it answers (proposal section)                  runs
#   ----------  --------------------------------------------------  ----
#   main        Sweep 1 x Sweep 2: N{8,16,24,32} x BANKS{4..32} x     64
#               4 workloads -- the saturation curves (II.C, IV.C)
#   nodb        double_buffer=False at BANKS=32 (IV.C)                16
#   ideal       infinite-bandwidth scratchpad per (workload, N)       16
#               (IV.A: must reproduce the compute-bound count)
#   banks       finer BANKS {2..64}, compute-bound layers (II.C)      48
#   reuse       reuse_M* GEMMs, M = 1..512 (II.C research question)   80
#   sw          the same layers in software on the RISC-V core         8
#               (Table II "speedup")
#
# Extra evidence for the hypotheses
#   knee        H1/H3: BANKS swept one bank at a time across the     126
#               knee (0.4 D/8 .. 1.5 D/8), compute-bound layers
#   idealdram   H1: the main grid with weights free of DRAM, so      48
#               only the scratchpad can limit utilization
#   extend      H1: N = 40..64, beyond the proposal's range, to       12
#               test the predicted saturation of 32 banks (N ~ 51)
#   sens        H1: one-at-a-time changes of lookahead, bank          66
#               latency, write buffer, weight buffer and DMA depth
#   clock       H2: array clock 250 MHz .. 1 GHz (DRAM supply per     60
#               array cycle changes 8x), and BANKS down to 1
#   dram        H2: DDR3-1600 / DDR5-6400 / DDR5-8400 instead of      15
#               DDR4-2400
#   bankrule    H3: exact ceil(D/8) banks, its power-of-two          44
#               rounding and 25% headroom; write buffer 4 vs 8
#   rulem       H3: the rule at every reuse M (banks = ceil(D/8)      48
#               and 25% fewer), DRAM taken out
#
# Design alternatives and checks
#   policy      reads-before-writes bank queues vs oldest-first at    64
#               aliasing bank counts (BANKS = N - 1) and elsewhere
#   lane        strict per-lane arbitration, compute-bound layers     32
#   rtlval      reduced *_rtl layers at every (N, BANKS), for the     64
#               gem5 <-> RTL comparison (phase 2, fig12)
#   standalone  the main grid without the CPU: checks the host path   64
#               changes nothing inside the compute phase

import math

WORKLOADS = ["resnet50_conv3x", "bert_ffn1", "bert_attn_qk", "llama7b_gemv"]
COMPUTE_BOUND = ["resnet50_conv3x", "bert_ffn1"]
N_VALUES = [8, 16, 24, 32]
N_EXTENDED = [40, 48, 56, 64]
BANK_VALUES = [4, 8, 16, 32]
FINE_BANKS = [2, 4, 8, 16, 32, 64]
REUSE_M = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
REUSE_K = 1024         # K and N_out of the reuse_M* GEMMs
REUSE_BANKS = [8, 32]  # B = 64 bytes/cycle (proposal's reference) and 256
RULE_M = [8, 16, 32, 64, 128, 512]
M_OF = {"resnet50_conv3x": 784, "bert_ffn1": 128, "bert_attn_qk": 128,
        "llama7b_gemv": 1}
CLOCKS = ["250MHz", "500MHz", "1GHz"]       # plus the default 2 GHz
DRAM_TYPES = ["DDR3_1600_8x8", "DDR5_6400_4x8", "DDR5_8400_4x8"]

# One-at-a-time sensitivity: (option, [alternative values]); the defaults
# are lookahead 8, spm_latency 1, psum_queue 8, weight_buffer 16KiB,
# dma_outstanding 64.
SENS_KNOBS = [
    ("lookahead", [4, 16]),
    ("spm_latency", [2, 4]),
    ("psum_queue", [16, 32]),
    ("weight_buffer", ["4KiB", "64KiB"]),
    ("dma_outstanding", [32, 128]),
]
SENS_POINTS = [(16, [8, 11, 16]), (32, [16, 21, 32])]  # (N, BANKS list)

# Options that map straight to systolic_se.py flags (collect.py).
OPT_FLAGS = {
    "psum_queue": "--psum-queue", "clock": "--clock",
    "mem_type": "--mem-type", "lookahead": "--lookahead",
    "spm_latency": "--spm-latency", "weight_buffer": "--weight-buffer",
    "dma_outstanding": "--dma-outstanding", "bank_policy": "--bank-policy",
}
OPT_TAG = {"psum_queue": "pq", "clock": "c", "mem_type": "", "lookahead":
           "la", "spm_latency": "lat", "weight_buffer": "wb",
           "dma_outstanding": "dma", "bank_policy": ""}


def next_pow2(x):
    return 1 << math.ceil(math.log2(max(1, x)))


def demand(N, M):
    return (5 * N * M + N * N) / max(M, N)


def _pt(workload, N, banks, db=True, arb="bank", ideal_spm=False,
        ideal_dram=False, mode="accel", **opts):
    opts = {k: v for k, v in opts.items() if v is not None}
    tag = f"{workload}_N{N}_B{banks}"
    tag += "" if db else "_nodb"
    tag += "_lane" if arb == "lane" else ""
    tag += "_idealspm" if ideal_spm else ""
    tag += "_idealdram" if ideal_dram else ""
    for k in sorted(opts):
        tag += f"_{OPT_TAG[k]}{opts[k]}"
    tag += {"accel": "", "sw": "_sw", "standalone": "_standalone"}[mode]
    return dict(tag=tag, workload=workload, N=N, banks=banks,
                double_buffer=db, arbitration=arb, ideal_spm=ideal_spm,
                ideal_dram=ideal_dram, mode=mode, opts=opts)


def suite_main():
    return [_pt(w, n, b) for w in WORKLOADS for n in N_VALUES
            for b in BANK_VALUES]


def suite_nodb():
    return [_pt(w, n, 32, db=False) for w in WORKLOADS for n in N_VALUES]


def suite_ideal():
    return [_pt(w, n, 32, ideal_spm=True) for w in WORKLOADS
            for n in N_VALUES]


def suite_banks():
    return [_pt(w, n, b) for w in COMPUTE_BOUND for n in N_VALUES
            for b in FINE_BANKS]


def knee_banks(n, M):
    """Every bank count from 0.4 to 1.5 x the rule's D/8."""
    d8 = demand(n, M) / 8
    return list(range(max(1, math.floor(0.4 * d8)), math.ceil(1.5 * d8) + 1))


def suite_knee():
    return [_pt(w, n, b) for w in COMPUTE_BOUND for n in N_VALUES
            for b in knee_banks(n, M_OF[w])]


def suite_idealdram():
    return [_pt(w, n, b, ideal_dram=True)
            for w in ("resnet50_conv3x", "bert_ffn1", "bert_attn_qk")
            for n in N_VALUES for b in BANK_VALUES]


def suite_extend():
    return [_pt("resnet50_conv3x", n, b) for n in N_EXTENDED
            for b in (16, 32, 64)]


def suite_sens():
    pts = []
    for n, bank_list in SENS_POINTS:
        for b in bank_list:
            pts.append(_pt("resnet50_conv3x", n, b))       # baseline
            for knob, values in SENS_KNOBS:
                for v in values:
                    pts.append(_pt("resnet50_conv3x", n, b, **{knob: v}))
    return pts


def suite_clock():
    pts = []
    # GEMV: DRAM supply per array cycle is 8x / 4x / 2x the 2 GHz case.
    for n in N_VALUES:
        for c in CLOCKS:
            pts.append(_pt("llama7b_gemv", n, 8, clock=c))
        # "regardless of BANKS" has a floor at B = D = N + 5: go down to 1
        for b in (1, 2, 4, 32):
            pts.append(_pt("llama7b_gemv", n, b, clock="250MHz"))
        for b in (1, 2):
            pts.append(_pt("llama7b_gemv", n, b))
    # Scratchpad-bound layers must not care about the clock.
    for w in ("resnet50_conv3x", "bert_ffn1", "bert_attn_qk"):
        for n in N_VALUES:
            for b in (8, 32):
                pts.append(_pt(w, n, b, clock="250MHz"))
    return pts


def suite_dram():
    pts = [_pt("llama7b_gemv", n, 8, mem_type=t) for t in DRAM_TYPES
           for n in N_VALUES]
    # The one DRAM-limited "compute-bound" point of the main sweep.
    pts += [_pt("bert_ffn1", 32, 32, mem_type=t) for t in DRAM_TYPES]
    return pts


def bank_rule_counts(n, M):
    """(label, BANKS) for the H3 rule variants at one (N, M)."""
    D = demand(n, M)
    exact = math.ceil(D / 8)
    return [("ceil(D/8)", exact),
            ("pow2", next_pow2(exact)),
            ("headroom", math.ceil(D / (0.8 * 8)))]


def suite_bankrule():
    pts = []
    for w in COMPUTE_BOUND:
        for n in N_VALUES:
            counts = sorted({b for _, b in bank_rule_counts(n, M_OF[w])})
            for b in counts:
                for pq in (4, None):       # None = the default (8)
                    pts.append(_pt(w, n, b, psum_queue=pq))
    return pts


def rule_m_banks(n, m):
    """(rule, 25% fewer) bank counts for reuse M at array size n."""
    rule = math.ceil(demand(n, m) / 8)
    return rule, max(1, math.floor(0.75 * rule))


def suite_rulem():
    pts = []
    for m in RULE_M:
        for n in N_VALUES:
            for b in sorted(set(rule_m_banks(n, m))):
                pts.append(_pt(f"reuse_M{m}", n, b, ideal_dram=True))
    return pts


def suite_reuse():
    return [_pt(f"reuse_M{m}", n, b) for m in REUSE_M for n in N_VALUES
            for b in REUSE_BANKS]


def resonant_banks(n, lo=2):
    """Bank counts below N that divide N - 1: there every partial-sum lane
    hits the same bank in the same cycle (see EVALUATION.txt)."""
    return [b for b in range(lo, n) if (n - 1) % b == 0]


POLICY_CASES = (
    # (N, BANKS, extra options): bank counts that alias, the rule's counts,
    # the short-lookahead case of the sensitivity suite, saturated points.
    [(n, n - 1, {}) for n in N_VALUES] +
    [(n, n - 1, {"psum_queue": 16}) for n in N_VALUES] +
    [(n, math.ceil(demand(n, 784) / 8), {}) for n in N_VALUES] +
    [(16, 11, {"lookahead": 4}), (32, 21, {"lookahead": 4}),
     (16, 8, {}), (32, 16, {})]
)


def suite_policy():
    """Reads-before-writes bank queues vs the default oldest-first."""
    pts = []
    for w in COMPUTE_BOUND:
        for n, b, extra in POLICY_CASES:
            pts.append(_pt(w, n, b, **extra))                  # fifo
            pts.append(_pt(w, n, b, bank_policy="reads_first", **extra))
    return pts


def suite_lane():
    return [_pt(w, n, b, arb="lane") for w in COMPUTE_BOUND
            for n in N_VALUES for b in BANK_VALUES]


def suite_sw():
    # N and BANKS are irrelevant in software; the accelerator stays idle.
    return [_pt(w, 8, 8, mode="sw") for w in WORKLOADS] + \
           [_pt(w + "_rtl", 8, 8, mode="sw") for w in WORKLOADS]


def suite_rtlval():
    return [_pt(w + "_rtl", n, b) for w in WORKLOADS for n in N_VALUES
            for b in BANK_VALUES]


def suite_standalone():
    return [_pt(w, n, b, mode="standalone") for w in WORKLOADS
            for n in N_VALUES for b in BANK_VALUES]


SUITES = {
    "main": suite_main,
    "nodb": suite_nodb,
    "ideal": suite_ideal,
    "banks": suite_banks,
    "reuse": suite_reuse,
    "sw": suite_sw,
    "knee": suite_knee,
    "idealdram": suite_idealdram,
    "extend": suite_extend,
    "sens": suite_sens,
    "clock": suite_clock,
    "dram": suite_dram,
    "bankrule": suite_bankrule,
    "rulem": suite_rulem,
    "policy": suite_policy,
    "lane": suite_lane,
    "rtlval": suite_rtlval,
    "standalone": suite_standalone,
}
