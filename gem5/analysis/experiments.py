#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : experiments.py                                                #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : Experiment suite definitions (saturation study)               #
#-----------------------------------------------------------------------------#
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
#
# Extensions of the model (each behind an option, default off)
#   rtlval_idealdram  rtlval without DRAM stalls, since the RTL        64
#               testbench preloads the weights
#   rtlval_lockstep  the same with the lockstep stall model            64
#   lockstep    the main grid with the lockstep stall model            64
#   lockcheck   existing options as a lockstep stand-in, and lockstep  24
#               without DRAM at the staircase's and the rule's banks
#   bankports   bankrule + knee with simple-dual-port banks           170
#   weightstore bank crossings of a weight tile, low reuse             60
#   bankhash    XOR skew hash at aliasing and non-aliasing bank        28
#               counts
#   mtile       512-word banks with M tiling at the rule's banks,      32
#               and the whole layer on-chip, at 2 GHz and 250 MHz
#   hostext     host metrics of the setup alone; prepacked weights      8
#   dramwide    DDR5-8400 on a wider memory bus                         6
#
# The RTL's configuration (rtl/src on the dev_hill branch): simple-dual-port
# banks, weights crossing the banks once.  Every hypothesis experiment again:
#   rtlcfg_main     the main grid                                       64
#   rtlcfg_knee     H1/H3: BANKS one at a time across the new knee      100
#   rtlcfg_gemv     H2: M = 1 at 2 GHz and 250 MHz, BANKS down to 1      48
#   rtlcfg_rulem    H3: both rules, and 25% fewer, at every reuse M      82
#   rtlcfg_reuse    the reuse sweep                                      80
#   rtlcfg_extend   N = 40..64                                           12
#   rtlcfg_clock    250 MHz, so DRAM cannot limit the array              36
#   rtlcfg_nodb     one weight register array, as the RTL has, and       64
#                   shadow registers at the same points
#   rtlcfg_mtile    512-word banks with M tiling, whole layer as        28
#                   reference, at 2 GHz and 250 MHz
#   rtlcfg_lockstep the RTL's global stall: main grid + ResNet-50 knee  107
#   rtlcfg_rtlval   reduced layers without DRAM, for the RTL comparison  64
#   rtlcfg_rtlval_lockstep  the same with the global stall               64
#   rtlcfg_idealdram the main grid with weights free of DRAM             48
#   rtlcfg_dram     other DRAM types                                     15
#
# Open choices in the RTL, each ready either way:
#   rtlcfg_rmw_main, _knee, _lockstep   partial sums read back through the
#                   read port while accumulating across K tiles
#   rtlcfg_nodb_main, _knee, _lockstep, _rtlval, _rtlval_lockstep
#                   one weight register array instead of shadow registers
#   rtlcfg_ntile, ntile  512-word banks tiled along M and N_out
#   psumopts        where partial sums live, four options side by side,
#                   BANKS one at a time, both stall models, no DRAM

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
    "stall_model": "--stall-model", "bank_ports": "--bank-ports",
    "weight_store": "--weight-store", "bank_hash": "--bank-hash",
    "spm_depth": "--spm-depth", "m_tile": "--m-tile",
    "membus_width": "--membus-width",
    "n_tile": "--n-tile", "psum_rmw": "--psum-rmw",
    "psum_store": "--psum-store",
    "split_host_stats": "--split-host-stats",
    "prepacked_weights": "--prepacked-weights",
}
OPT_TAG = {"psum_queue": "pq", "clock": "c", "mem_type": "", "lookahead":
           "la", "spm_latency": "lat", "weight_buffer": "wb",
           "dma_outstanding": "dma", "bank_policy": "",
           "stall_model": "", "bank_ports": "", "weight_store": "ws",
           "bank_hash": "", "spm_depth": "depth", "m_tile": "mt",
           "membus_width": "mb", "n_tile": "nt", "psum_rmw": "rmw",
           "psum_store": "ps", "split_host_stats": "split",
           "prepacked_weights": "prepack"}
# Options that are flags: the value only says the flag is present.
OPT_FLAG_ONLY = {"split_host_stats", "prepacked_weights", "psum_rmw"}


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
        tag += (f"_{OPT_TAG[k]}" if k in OPT_FLAG_ONLY
                else f"_{OPT_TAG[k]}{opts[k]}")
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


def suite_rtlval_idealdram():
    return [_pt(w + "_rtl", n, b, ideal_dram=True) for w in WORKLOADS
            for n in N_VALUES for b in BANK_VALUES]


def suite_rtlval_lockstep():
    return [_pt(w + "_rtl", n, b, ideal_dram=True, stall_model="lockstep")
            for w in WORKLOADS for n in N_VALUES for b in BANK_VALUES]


def suite_lockstep():
    return [_pt(w, n, b, stall_model="lockstep") for w in WORKLOADS
            for n in N_VALUES for b in BANK_VALUES]


# Where the lockstep staircase is below 1 (N, BANKS), and BANKS = ceil(5N/8),
# where the evenly spread staircase reaches 1.
LOCK_POINTS = [(8, 4), (16, 4), (16, 8), (24, 4), (24, 8), (32, 4), (32, 8),
               (32, 16)]
LOCK_RULE = [(n, math.ceil(5 * n / 8)) for n in N_VALUES]
# Far more banks than words: fixed addresses still collide now and then.
LOCK_WIDE = [(n, 2 * n) for n in N_VALUES]


def suite_lockcheck():
    w = "resnet50_conv3x"
    # Do the existing options already give lockstep?
    pts = [_pt(w, n, b, arb="lane", lookahead=1, psum_queue=1)
           for n, b in LOCK_POINTS]
    pts += [_pt(w, n, b, stall_model="lockstep", ideal_dram=True)
            for n, b in LOCK_POINTS + LOCK_RULE + LOCK_WIDE]
    return pts


def _unique(pts):
    return list({p["tag"]: p for p in pts}.values())


def suite_bankports():
    pts = suite_bankrule() + suite_knee()
    for p in pts:
        p.update(_pt(p["workload"], p["N"], p["banks"], bank_ports="1r1w",
                     **{k: v for k, v in p["opts"].items()}))
    return _unique(pts)


def suite_weightstore():
    pts = []
    for store in ("spm", "fill_bypass", "separate"):
        # 250 MHz array: DRAM supply is 8x, so the banks set the limit.
        for m in (1, 8):
            for n in (8, 16):
                for b in (1, 2, 4, 8):
                    pts.append(_pt(f"reuse_M{m}", n, b, clock="250MHz",
                                   weight_store=store))
        # 2 GHz: the same at M = 1 with DRAM back in the picture.
        for b in (1, 2, 4, 8):
            pts.append(_pt("reuse_M1", 8, b, weight_store=store))
    return pts


def suite_bankhash():
    pts = []
    for w in COMPUTE_BOUND:
        counts = [(n, n - 1) for n in (16, 24, 32)]
        counts += [(n, math.ceil(demand(n, M_OF[w]) / 8)) for n in N_VALUES]
        for n, b in counts:
            pts.append(_pt(w, n, b))
            pts.append(_pt(w, n, b, bank_hash="xor"))
    return _unique(pts)


def rule_banks(n, M):
    return math.ceil(demand(n, M) / 8)


def suite_mtile():
    # At 2 GHz the extra weight reloads run into the DRAM ceiling; at an
    # FPGA-like 250 MHz only the scratchpad is left.  The whole layer on-chip
    # is the reference at each clock.
    pts = []
    for w in COMPUTE_BOUND:
        for n in N_VALUES:
            b = rule_banks(n, M_OF[w])
            for clock in (None, "250MHz"):
                pts.append(_pt(w, n, b, clock=clock))
                pts.append(_pt(w, n, b, clock=clock, spm_depth=512,
                               m_tile="auto"))
    return pts


def suite_hostext():
    pts = [_pt(w, 16, 8, split_host_stats=True) for w in WORKLOADS]
    pts += [_pt("llama7b_gemv", n, 8, prepacked_weights=True)
            for n in N_VALUES]
    return pts


def suite_dramwide():
    pts = [_pt("llama7b_gemv", n, 8, mem_type="DDR5_8400_4x8",
               membus_width=64) for n in N_VALUES]
    pts.append(_pt("bert_ffn1", 32, 32, mem_type="DDR5_8400_4x8",
                   membus_width=64))
    # DDR4 was never held back by the bus: this one must not move much.
    pts.append(_pt("llama7b_gemv", 32, 8, membus_width=64))
    return pts


# ---------------------------------------------------- the RTL's configuration
# These settings hold if the RTL keeps partial sums in the banks, written once,
# with shadow weight registers (the design as proposed); the rtlcfg_rmw_* and
# rtlcfg_nodb_* suites cover the other answers, and psumopts compares them.
# rtl/src/sram_bank.sv has independent read and write channels (1r1w).  The
# testbench writes the weights while computation is disabled and top.sv reads
# them once into the weights register array that feeds the MACs: one crossing
# of the banks during compute.  top.sv has no queues between banks and array,
# only a global stall, hence the lockstep suites.
RTL_OPTS = dict(bank_ports="1r1w", weight_store="fill_bypass")


def rtl_rule(n, M):
    """The bank rule for the RTL's configuration: the busier port."""
    return math.ceil(max(n * M + n * n, 4 * n * M) / (8 * max(M, n)))


def rtl_knee_banks(n, M):
    r = rtl_rule(n, M)
    return list(range(max(1, math.floor(0.4 * r)), math.ceil(1.5 * r) + 1))


def suite_rtlcfg_main():
    return [_pt(w, n, b, **RTL_OPTS) for w in WORKLOADS for n in N_VALUES
            for b in BANK_VALUES]


def suite_rtlcfg_knee():
    return [_pt(w, n, b, **RTL_OPTS) for w in COMPUTE_BOUND for n in N_VALUES
            for b in rtl_knee_banks(n, M_OF[w])]


def suite_rtlcfg_gemv():
    return [_pt("llama7b_gemv", n, b, clock=c, **RTL_OPTS)
            for n in N_VALUES for b in (1, 2, 4, 8, 16, 32)
            for c in (None, "250MHz")]


def suite_rtlcfg_rulem():
    pts = []
    for m in RULE_M:
        for n in N_VALUES:
            r = rtl_rule(n, m)
            ceil_d = math.ceil(demand(n, m) / 8)
            counts = {r, max(1, math.floor(0.75 * r)), ceil_d,
                      max(1, math.floor(0.75 * ceil_d))}
            for b in sorted(counts):
                pts.append(_pt(f"reuse_M{m}", n, b, ideal_dram=True,
                               **RTL_OPTS))
    return pts


def suite_rtlcfg_reuse():
    return [_pt(f"reuse_M{m}", n, b, **RTL_OPTS) for m in REUSE_M
            for n in N_VALUES for b in REUSE_BANKS]


def suite_rtlcfg_extend():
    return [_pt("resnet50_conv3x", n, b, **RTL_OPTS) for n in N_EXTENDED
            for b in (16, 32, 64)]


def suite_rtlcfg_clock():
    return [_pt(w, n, b, clock="250MHz", **RTL_OPTS)
            for w in ("resnet50_conv3x", "bert_ffn1", "bert_attn_qk")
            for n in N_VALUES for b in (8, 16, 32)]


def suite_rtlcfg_nodb():
    # Each point with one weight register array and with shadow registers.
    pts = []
    for w in WORKLOADS:
        for n in N_VALUES:
            for b in sorted({rtl_rule(n, M_OF[w]), 32}):
                pts.append(_pt(w, n, b, db=False, **RTL_OPTS))
                pts.append(_pt(w, n, b, **RTL_OPTS))
    return pts


def m_tile_fits(w, n, b, depth=512):
    """Whether one row of the layer plus its weight prefetch fits in BANKS x
    depth words, computed as systolic_se.py does; at fewer banks than this
    the layer cannot run at all."""
    K, NOUT = {"resnet50_conv3x": (1152, 128), "bert_ffn1": (768, 3072)}[w]
    capacity = depth * 8 * b
    slots = min(max(2, 16384 // (n * n)), max(2, capacity // 4 // (n * n)))
    align = lambda x: (x + 63) // 64 * 64  # noqa: E731
    nkb, nnb = math.ceil(K / n), math.ceil(NOUT / n)
    return (align(align(nkb * n) + slots * n * n) + nnb * n * 4) <= capacity


def suite_rtlcfg_mtile():
    pts = []
    for w in COMPUTE_BOUND:
        for n in N_VALUES:
            b = rtl_rule(n, M_OF[w])
            if not m_tile_fits(w, n, b):
                continue
            for clock in (None, "250MHz"):
                pts.append(_pt(w, n, b, clock=clock, **RTL_OPTS))
                pts.append(_pt(w, n, b, clock=clock, spm_depth=512,
                               m_tile="auto", **RTL_OPTS))
    return pts


def suite_rtlcfg_lockstep():
    pts = [_pt(w, n, b, stall_model="lockstep", **RTL_OPTS)
           for w in WORKLOADS for n in N_VALUES for b in BANK_VALUES]
    pts += [_pt("resnet50_conv3x", n, b, stall_model="lockstep", **RTL_OPTS)
            for n in N_VALUES for b in rtl_knee_banks(n, M_OF[
                "resnet50_conv3x"])]
    # The rule's count of the other compute-bound layer, off the grid at N=24.
    pts += [_pt(w, n, rtl_rule(n, M_OF[w]), stall_model="lockstep", **RTL_OPTS)
            for w in COMPUTE_BOUND for n in N_VALUES]
    return _unique(pts)


def suite_rtlcfg_rtlval():
    return [_pt(w + "_rtl", n, b, ideal_dram=True, **RTL_OPTS)
            for w in WORKLOADS for n in N_VALUES for b in BANK_VALUES]


def suite_rtlcfg_rtlval_lockstep():
    return [_pt(w + "_rtl", n, b, ideal_dram=True, stall_model="lockstep",
                **RTL_OPTS)
            for w in WORKLOADS for n in N_VALUES for b in BANK_VALUES]


def suite_rtlcfg_idealdram():
    return [_pt(w, n, b, ideal_dram=True, **RTL_OPTS)
            for w in ("resnet50_conv3x", "bert_ffn1", "bert_attn_qk")
            for n in N_VALUES for b in BANK_VALUES]


def suite_rtlcfg_dram():
    pts = [_pt("llama7b_gemv", n, 8, mem_type=t, **RTL_OPTS)
           for t in DRAM_TYPES for n in N_VALUES]
    pts += [_pt("bert_ffn1", 32, 32, mem_type=t, **RTL_OPTS)
            for t in DRAM_TYPES]
    return pts


# ------ partial sums read back through the read port while accumulating
def rtl_rmw_rule(n, M):
    """The read port now carries N activations + 4N old partial sums."""
    return math.ceil((5 * n * M + n * n) / (8 * max(M, n)))


def rtl_rmw_knee_banks(n, M):
    r = rtl_rmw_rule(n, M)
    return list(range(max(1, math.floor(0.4 * r)), math.ceil(1.5 * r) + 1))


def suite_rtlcfg_rmw_main():
    return [_pt(w, n, b, psum_rmw=True, **RTL_OPTS) for w in WORKLOADS
            for n in N_VALUES for b in BANK_VALUES]


def suite_rtlcfg_rmw_knee():
    return [_pt(w, n, b, psum_rmw=True, **RTL_OPTS) for w in COMPUTE_BOUND
            for n in N_VALUES for b in rtl_rmw_knee_banks(n, M_OF[w])]


def suite_rtlcfg_rmw_lockstep():
    pts = [_pt("resnet50_conv3x", n, b, psum_rmw=True, stall_model="lockstep",
               **RTL_OPTS)
           for n in N_VALUES for b in BANK_VALUES]
    pts += [_pt(w, n, b, psum_rmw=True, stall_model="lockstep", **RTL_OPTS)
            for w in COMPUTE_BOUND for n in N_VALUES
            for b in (n // 2, math.ceil(5 * n / 8), math.ceil(3 * n / 4))]
    return _unique(pts)


# ------ one weight register array instead of shadow registers
def suite_rtlcfg_nodb_main():
    return [_pt(w, n, b, db=False, **RTL_OPTS) for w in WORKLOADS
            for n in N_VALUES for b in BANK_VALUES]


def suite_rtlcfg_nodb_knee():
    return [_pt(w, n, b, db=False, **RTL_OPTS) for w in COMPUTE_BOUND
            for n in N_VALUES for b in rtl_knee_banks(n, M_OF[w])]


def suite_rtlcfg_nodb_lockstep():
    return [_pt(w, n, b, db=False, stall_model="lockstep", **RTL_OPTS)
            for w in COMPUTE_BOUND for n in N_VALUES
            for b in sorted(set(BANK_VALUES) | {rtl_rule(n, M_OF[w])})]


def suite_rtlcfg_nodb_rtlval():
    return [_pt(w + "_rtl", n, b, db=False, ideal_dram=True, **RTL_OPTS)
            for w in WORKLOADS for n in N_VALUES for b in BANK_VALUES]


def suite_rtlcfg_nodb_rtlval_lockstep():
    return [_pt(w + "_rtl", n, b, db=False, ideal_dram=True,
                stall_model="lockstep", **RTL_OPTS)
            for w in WORKLOADS for n in N_VALUES for b in BANK_VALUES]


# ------ 512-word banks tiled along M and N_out
def suite_rtlcfg_ntile():
    pts = []
    for w in COMPUTE_BOUND:
        for n in N_VALUES:
            b = rtl_rule(n, M_OF[w])
            for clock in (None, "250MHz"):
                pts.append(_pt(w, n, b, clock=clock, **RTL_OPTS))
                pts.append(_pt(w, n, b, clock=clock, spm_depth=512,
                               m_tile="auto", n_tile="auto", **RTL_OPTS))
    return pts


def suite_ntile():
    """The same with one shared port, at the bank counts of ceil(D/8)."""
    pts = []
    for w in COMPUTE_BOUND:
        for n in N_VALUES:
            b = rule_banks(n, M_OF[w])
            for clock in (None, "250MHz"):
                pts.append(_pt(w, n, b, clock=clock))
                pts.append(_pt(w, n, b, clock=clock, spm_depth=512,
                               m_tile="auto", n_tile="auto"))
    return pts


# ------ where partial sums live
PSUM_OPTIONS = {
    "A": dict(weight_store="fill_bypass"),
    "B": dict(weight_store="fill_bypass", bank_ports="1r1w"),
    "B+rmw": dict(weight_store="fill_bypass", bank_ports="1r1w",
                  psum_rmw=True),
    "C": dict(weight_store="fill_bypass", bank_ports="1r1w",
              psum_store="external"),
}


def psumopts_banks(n):
    return sorted(set(range(max(1, n // 8), math.ceil(3 * n / 4) + 3)) |
                  set(BANK_VALUES))


def suite_psumopts():
    return [_pt(w, n, b, ideal_dram=True, stall_model=s, **o)
            for w in COMPUTE_BOUND for n in N_VALUES
            for b in psumopts_banks(n)
            for o in PSUM_OPTIONS.values()
            for s in (None, "lockstep")]


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
    # From here on: options of the model that are off by default.
    "rtlval_idealdram": suite_rtlval_idealdram,
    "rtlval_lockstep": suite_rtlval_lockstep,
    "lockstep": suite_lockstep,
    "lockcheck": suite_lockcheck,
    "bankports": suite_bankports,
    "weightstore": suite_weightstore,
    "bankhash": suite_bankhash,
    "mtile": suite_mtile,
    "hostext": suite_hostext,
    "dramwide": suite_dramwide,
    "rtlcfg_main": suite_rtlcfg_main,
    "rtlcfg_knee": suite_rtlcfg_knee,
    "rtlcfg_gemv": suite_rtlcfg_gemv,
    "rtlcfg_rulem": suite_rtlcfg_rulem,
    "rtlcfg_reuse": suite_rtlcfg_reuse,
    "rtlcfg_extend": suite_rtlcfg_extend,
    "rtlcfg_clock": suite_rtlcfg_clock,
    "rtlcfg_nodb": suite_rtlcfg_nodb,
    "rtlcfg_mtile": suite_rtlcfg_mtile,
    "rtlcfg_lockstep": suite_rtlcfg_lockstep,
    "rtlcfg_rtlval": suite_rtlcfg_rtlval,
    "rtlcfg_rtlval_lockstep": suite_rtlcfg_rtlval_lockstep,
    "rtlcfg_idealdram": suite_rtlcfg_idealdram,
    "rtlcfg_dram": suite_rtlcfg_dram,
    "rtlcfg_rmw_main": suite_rtlcfg_rmw_main,
    "rtlcfg_rmw_knee": suite_rtlcfg_rmw_knee,
    "rtlcfg_rmw_lockstep": suite_rtlcfg_rmw_lockstep,
    "rtlcfg_nodb_main": suite_rtlcfg_nodb_main,
    "rtlcfg_nodb_knee": suite_rtlcfg_nodb_knee,
    "rtlcfg_nodb_lockstep": suite_rtlcfg_nodb_lockstep,
    "rtlcfg_nodb_rtlval": suite_rtlcfg_nodb_rtlval,
    "rtlcfg_nodb_rtlval_lockstep": suite_rtlcfg_nodb_rtlval_lockstep,
    "rtlcfg_ntile": suite_rtlcfg_ntile,
    "ntile": suite_ntile,
    "psumopts": suite_psumopts,
}

# The suites of the default model (no option that is off by default).
BASELINE_SUITES = list(SUITES)[:list(SUITES).index("rtlval_idealdram")]
