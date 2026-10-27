EEL5764 Group 4 -- measurements and saturation curves
======================================================

This folder turns gem5 runs of the SystolicAccel model (../gem5_model)
into the figures and tables the proposal promises (Sec. II.C, IV, V), plus
extra experiments that test each hypothesis harder.  Every run is
host-driven: a RISC-V program on the RiscvTimingSimpleCPU launches the
layer, and statistics cover exactly the layer.

Outputs go to ../results/ (or $RESULTS_DIR).  git ignores them: the
repository holds only what generates them, and the project documents
carry the results.  results/data/key_results.json has every headline
number, machine-readable.


1. Commands (from the repository root)
--------------------------------------

Everything, from scratch (about an hour with 12 runs in parallel;
cached afterwards):

  make all              # = make build data figures

Step by step:

  make build            # gem5 + host program + workloads (once)
  make list             # suites and run counts
  make data             # every suite -> results/data/<suite>.csv
  make data SUITES="main knee"     # only some suites
  make data FORCE=1     # rerun after changing the model (see below)
  make data SIM_JOBS=8  # fewer parallel gem5 runs
  make figures          # -> results/figures, results/tables,
                        #    results/data/key_results.json
  make figures PLOTS="fig1 fig17 tables"
  make check            # results/ vs the last git commit, number by number

The scripts behind the targets (after  source scripts/env.sh ):

  $PYTHON analysis/collect.py [SUITE ...] [-j JOBS] [--force] [--list]
  $PYTHON analysis/plot_curves.py [--only fig1 ... tables] [--format pdf]
  $PYTHON analysis/compare_results.py REF_RESULTS_DIR [NEW_RESULTS_DIR]

The collection can be interrupted (Ctrl-C, sleep, reboot) and restarted:
finished runs are kept in build/runs/.  Its progress is printed, one line
per run.

After you rebuild gem5 or change a model default (systolic_se.py,
SystolicAccel.py, the host program), use FORCE=1 (--force).  Otherwise
cached runs from the old build are reused.


2. Files
--------

  experiments.py     the experiment suites (which N, BANKS, workloads,
                     options).  Edit this to add points.
  collect.py         runs the suites with gem5 in parallel and writes
                     results/data/<suite>.csv (one row per run, every
                     metric)
  parse_stats.py     one gem5 output directory -> one CSV row
  model.py           the analytical model, eq. (1)-(2) plus the DRAM
                     ceiling and the DMA-fill term
  plot_curves.py     results/data/*.csv -> results/figures/,
                     results/tables/, results/data/key_results.json
  style.py           colours, fonts and line styles shared by all figures
  compare_results.py two results directories, number by number (make check);
                     simulation data must match exactly, derived tables
                     to within floating-point round-off

  build/runs/<tag>/  raw gem5 output of each run (stats.txt, config.ini,
                     simout.txt, simerr.txt, run.json, systolic_run.json);
                     not in git.  Tags look like
                     resnet50_conv3x_N16_B11[_nodb][_lane]
                     [_idealspm][_idealdram][_c250MHz][_DDR5_8400_4x8]
                     [_la4][_lat2][_pq16][_wb4KiB][_dma32][_sw|_standalone]
  results/data/<suite>.csv  parsed results (columns:
                     gem5_model/README.txt section 7 and parse_stats.py)
  results/data/key_results.json  headline numbers, machine-readable
  results/figures/   PNG (220 dpi) + PDF (vector, for the paper)
  results/tables/    Markdown (paste into notes) + CSV (for the paper)

Suites (make list); 772 unique runs:

  Proposal sweeps
  main        64  4 workloads x N{8,16,24,32} x BANKS{4,8,16,32}  Sweeps 1+2
  nodb        16  double_buffer=False, BANKS=32                   Sec. IV.C
  ideal       16  infinite-bandwidth scratchpad per (wl, N)       Sec. IV.A
  banks       48  ResNet-50 + BERT FFN1, BANKS{2..64}             bank rule
  reuse       80  reuse_M* GEMMs (M = 1..512, 1024 x 1024),       reuse M
                  N{8..32}, BANKS{8,32}
  sw           8  the layers in software on the RISC-V host       speedup

  Extra evidence for the hypotheses
  knee       126  H1/H3: BANKS in steps of 1 from 0.4 to 1.5 x D/8,
                  ResNet-50 + BERT FFN1, every N
  idealdram   48  H1: main grid (3 layers) with weights free of DRAM, so
                  only the scratchpad can limit utilization
  extend      12  H1: ResNet-50 at N = 40..64 (outside the proposal's
                  range) to test the predicted saturation of 32 banks
  sens        66  H1: lookahead, bank latency, write buffer, weight buffer
                  and DMA depth changed one at a time
  clock       60  H2: array clock 250 MHz / 500 MHz / 1 GHz (DRAM supply
                  per array cycle 8x / 4x / 2x), BANKS down to 1
  dram        15  H2: DDR3-1600, DDR5-6400, DDR5-8400 instead of DDR4-2400
  bankrule    44  H3: exact ceil(D/8), power-of-two and 25%-headroom bank
                  counts, write buffer 4 vs 8 entries/lane
  rulem       48  H3: the rule at every reuse M (and 25% fewer banks)

  Design alternatives and checks
  policy      64  bank queues that serve reads before writes vs the default
                  oldest-first, at aliasing bank counts (BANKS = N - 1),
                  the rule's counts, a short lookahead and saturated points
  lane        32  ResNet-50 + BERT FFN1, per-lane arbitration
  rtlval      64  reduced *_rtl layers for the RTL comparison (phase 2)
  standalone  64  the main grid without the CPU (consistency)

  The slow runs are the 74 LLaMA GEMV runs (1-2 min each: the host packs
  16 MiB of weights before the layer) and the software BERT FFN1 baseline
  (50-60 min, and it sets the total).  The rest take seconds.


3. What each figure shows
-------------------------

  fig1_saturation_vs_N         THE saturation curves: PE utilization vs N,
                               one line per BANKS, model dashed, 90% line;
                               one panel per workload
  fig2_universal_saturation    every scratchpad run on one axis: load D/B
                               vs utilization / stall-free utilization,
                               against min(1, B/D).  Left: DDR4 weights
                               (main + knee).  Right: weights free of DRAM
  fig3_delivered_bandwidth     delivered bytes/cycle vs N against demand
                               D(N) (eq. 1) and supply B (eq. 2): the
                               plateau at B is the saturation
  fig4_bank_provisioning       utilization vs BANKS (powers of two) per N
  fig5_utilization_vs_reuse    utilization vs M (1..512) per N, for B = 64
                               and B = 256 bytes/cycle
  fig6_gemv_dram_bound         M = 1 at 2 GHz: utilization identical for
                               every BANKS; DDR4 bus at its limit
  fig7_stall_breakdown         fraction of cycles: useful / idle slots / SPM
                               read stall / SPM write stall / DMA stall
  fig8_double_buffering        double- vs single-buffered weights
  fig9_arbitration_lane_vs_bank  per-lane ports vs per-bank queues
  fig10_spm_amat_vs_load       scratchpad AMAT vs load D/B
  fig11_model_vs_gem5          predicted vs measured utilization, main sweep
  fig12_gem5_vs_rtl            (after phase 2) gem5 vs Questa cycle counts,
                               +/-5% band
  fig13_edp                    energy-delay product per layer: DRAM energy
                               now, + array energy once
                               results/data/vivado.csv exists
  fig14_bank_rule              H3: utilization at exact ceil(D/8) banks,
                               its power-of-two rounding and 25% headroom,
                               with a 4- vs 8-entry psum write buffer
  fig15_layer_phases           host-driven layer time split into A load /
                               compute / C store
  fig16_speedup                speedup over the same GEMM in software on
                               the RISC-V host
  fig17_knee_bank_by_bank      H1/H3: utilization vs BANKS in steps of one
                               bank across the knee, ceil(D/8) marked
  fig18_gemv_dram_supply       H2: M = 1 utilization (a) per array clock,
                               (b) all clocks and DRAM types on one
                               supply/need axis, (c) vs BANKS at 250 MHz
  fig19_sensitivity            H1: utilization under one-at-a-time design
                               knob changes
  fig20_rule_vs_reuse          H3: utilization at BANKS = ceil(D(N,M)/8) and
                               at 25% fewer banks, for M = 8..512
  fig21_extended_N             H1: ResNet-50 up to N = 64; eq. (1)'s
                               saturation point of 32 banks (N = 50.6)
  fig22_bank_policy            the dip at BANKS = N - 1 (all partial-sum
                               lanes on one bank) and its removal by
                               reads-first queues + a 16-entry write buffer

Tables:

  table1_predicted_scaling     the proposal's Table I, recomputed
  table2_metrics               the proposal's Table II metrics for all 64
                               sweep points: utilization, stalls, delivered
                               BW (spm*.bwRead/bwWrite), bank balance
                               (spm_xbar.pktCount), AMAT, layer time,
                               speedup, host CPI/IPC, DRAM busUtil,
                               DRAM energy, EDP (DRAM part), verify
  saturation_points            the research question: at what N does
                               utilization fall below 90%, per workload and
                               BANKS (gem5 and eq. 1-2)
  knee_bank_by_bank            per (layer, N): utilization at ceil(D/8)-1,
                               ceil(D/8), ceil(D/8)+1; smallest BANKS for
                               90% and for 99% of stall-free; D/B at 90%
  fit_quality                  model vs gem5 per suite: R^2, RMSE, errors,
                               with and without the DMA-fill term
  bank_policy                  oldest-first vs reads-first bank queues
  extended_N                   ResNet-50 at N = 8..64, 16/32/64 banks
  sensitivity                  largest utilization change per knob sweep
  gemv_dram_supply             every M = 1 run: utilization, 1/N cap,
                               DRAM supply / need, DRAM bus utilization
  clock_independence           250 MHz vs 2 GHz, scratchpad-bound layers
  bert_n32_dram_types          the DRAM-limited BERT point per DRAM type
  bank_matching                ceil(D/8) rule at power-of-two counts
  rule_vs_reuse                the rule (and 25% fewer banks) per (M, N)
  bram_estimate                BRAM36 estimate per (layer, N) -- NOT Vivado
  ideal_baseline               ideal-scratchpad cycles vs the analytical
                               count (the Sec. IV.A baseline check)
  speedup                      speedup per (workload, N, BANKS) and the
                               software baseline times
  consistency_host_vs_standalone  compute phase with and without the CPU
  edp                          EDP terms per run
  hypotheses.md                auto-generated H1/H2/H3 checks


4. Adding RTL and Vivado results later
--------------------------------------

  RTL cycle counts (phase 2 cross-validation):
    cp analysis/templates/rtl_cycles.csv.example results/data/rtl_cycles.csv
    # fill in: workload,N,banks,cycles  (workload = the *_rtl names)
    make data SUITES=rtlval                    # gem5 side
    make figures PLOTS=fig12
    -> results/figures/fig12_gem5_vs_rtl,
       results/tables/rtl_crossvalidation (error %)

  Vivado (post-route) per array size, for EDP (proposal Sec. V):
    cp analysis/templates/vivado.csv.example results/data/vivado.csv
    # fill in: N,power_W,fmax_MHz,dsp,lut,ff,bram
    make figures PLOTS=fig13
    EDP = (DRAM rank*.totalEnergy + power x totalCycles / Fmax) x simSeconds
    (mixing the 2 GHz gem5 clock with the FPGA's Fmax needs a decision:
    see the proposal review, item A3)

  The .example files (analysis/templates/) contain made-up numbers that
  only show the format.  Do not copy them without replacing the values.


5. Notes
--------

  * Colours: workloads use a fixed four-colour categorical order (blue,
    orange, aqua, yellow).  Ordered quantities (BANKS, N or clock) use one
    blue ramp, light = small and dark = large, and every series also has
    its own marker.  The analytical model is always the dashed line in
    the colour of its series.
  * The model (analysis/model.py) has one calibrated
    constant per DRAM type: the fraction of its peak bandwidth it sustains
    under the weight DMA (DDR4-2400: 70%), read from one run.  Everything
    else is eq. (1)-(2), the schedule, and the DMA-fill term (a weight
    tile crosses the scratchpad twice).
  * Every run checks its output bit-exactly, twice (accelerator and RISC-V
    host).  plot_curves.py warns if any CSV row has verify_mismatches != 0.
  * Results are deterministic and do not depend on where the
    repository is cloned: rerunning gives identical numbers (make check).
    The raw runs' paths are stored relative to the repository root.
