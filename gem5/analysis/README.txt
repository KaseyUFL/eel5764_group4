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

Everything, from scratch (about two hours with 12 runs in parallel;
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
  make check REF=dir    # results/ vs a reference folder, number by number
  make regress          # default model vs analysis/golden_stats.csv (1 min)

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
  regress.py         reruns 12 representative points of the default model
  regress.sh         and diffs them against golden_stats.csv, exactly
  golden_stats.csv   (make regress).  Run it after every model change.
  trace_diff.py      first cycle where two --bank-trace files disagree, e.g.
                     gem5 vs RTL

  build/runs/<tag>/  raw gem5 output of each run (stats.txt, config.ini,
                     simout.txt, simerr.txt, run.json, systolic_run.json);
                     not in git.  Tags look like
                     resnet50_conv3x_N16_B11[_nodb][_lane]
                     [_idealspm][_idealdram][_c250MHz][_DDR5_8400_4x8]
                     [_la4][_lat2][_pq16][_wb4KiB][_dma32][_lockstep][_1r1w]
                     [_wsseparate][_xor][_depth512][_mtauto][_mb64][_split]
                     [_prepack][_sw|_standalone]
  results/data/<suite>.csv  parsed results (columns:
                     gem5_model/README.txt section 7 and parse_stats.py)
  results/data/key_results.json  headline numbers, machine-readable
  results/figures/   PNG (220 dpi) + PDF (vector, for the paper)
  results/tables/    Markdown (paste into notes) + CSV (for the paper)

Suites (make list); 3,573 unique runs (772 of the default model):

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

  Options of the model that are off by default
  rtlval_idealdram 64  rtlval without DRAM: the RTL testbench preloads the
                       weights
  rtlval_lockstep  64  the same with the lockstep stall model
  lockstep    64  the main grid with the lockstep stall model
  lockcheck   24  existing options as a lockstep stand-in; lockstep without
                  DRAM at the staircase's, the rule's and twice N banks
  bankports  149  bankrule + knee with simple-dual-port banks
  weightstore 60  1, 2 or 0 bank crossings of a weight tile at low reuse
  bankhash    28  XOR skew hash at aliasing and rule bank counts
  mtile       32  512-word banks with M tiling at the rule's banks, and the
                  whole layer on-chip, at 2 GHz and 250 MHz
  hostext      8  host metrics of the setup alone; prepacked weights
  dramwide     6  DDR5-8400 on a 64-byte memory bus

  The RTL's configuration: --bank-ports 1r1w --weight-store fill_bypass
  (gem5_model/README.txt 6.10)
  rtlcfg_main     64  the main grid
  rtlcfg_knee    100  H1/H3: BANKS one at a time across the new knee
  rtlcfg_gemv     48  H2: M = 1 at 2 GHz and 250 MHz, BANKS down to 1
  rtlcfg_rulem    82  H3: both bank rules, and 25% fewer, at every reuse M
  rtlcfg_reuse    80  the reuse sweep
  rtlcfg_extend   12  N = 40..64
  rtlcfg_clock    36  250 MHz, so DRAM cannot limit the array
  rtlcfg_nodb     64  one weight register array, and shadow registers, at
                      the same points
  rtlcfg_mtile    28  512-word banks with M tiling, at 2 GHz and 250 MHz
  rtlcfg_lockstep 107 the global stall: main grid + ResNet-50 knee
  rtlcfg_rtlval   64  reduced layers without DRAM
  rtlcfg_rtlval_lockstep 64  the same with the global stall
  rtlcfg_idealdram 48  the main grid with weights free of DRAM
  rtlcfg_dram     15  other DRAM types

  Open choices in the RTL, each ready either way
  rtlcfg_rmw_main 64, rtlcfg_rmw_knee 132, rtlcfg_rmw_lockstep 37
                      partial sums read back through the read port
  rtlcfg_nodb_main 64, rtlcfg_nodb_knee 100, rtlcfg_nodb_lockstep 34,
  rtlcfg_nodb_rtlval 64, rtlcfg_nodb_rtlval_lockstep 64
                      one weight register array
  rtlcfg_ntile    32  512-word banks tiled along M and N_out
  ntile           32  the same with one shared port
  psumopts      1088  where partial sums live: four options, BANKS one at a
                      time, both stall models, ResNet-50 + BERT FFN1, no DRAM

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
  fig23_lockstep_vs_decoupled  steady-state utilization vs load D/B: the
                               decoupled model, the lockstep model and the
                               lines min(1, B/D) and 1 / ceil(words/BANKS)
  fig24_bank_ports             utilization vs BANKS, one shared port per
                               bank against simple dual port
  fig25_rtl_config_saturation  the RTL's configuration on the proposal's load
                               axis and on the load of the busier port
  fig26_rtl_config_stall       the RTL's configuration: decoupled queues
                               against a global stall (ResNet-50)
  fig27_psum_options           where partial sums live sets the bank count

  make paper-figures builds the paper's set from the rtlcfg_* suites into
  results/paper/rtlcfg/ (fig1, 2, 5, 12, 13, 15, 17, 18, 20, 25, 26, 27).  In
  that set the load axis is the busier port and the rule is that port's.
  make paper-figures PAPER_SOURCE=baseline builds the one-shared-port set into
  results/paper/baseline/ (fig1, 2, 5, 12, 13, 14, 15, 17, 18, 20, 22, 23).

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
  saturation_from_load         the N where a B crosses 90%, from the load the
                               knee suite measures rather than interpolating
                               between N = 8 and 16
  rtl_gem5_columns             gem5 cycles for the RTL comparison: without
                               DRAM (the match for a preloading testbench),
                               with DRAM, and lockstep
  lockstep_vs_decoupled        both stall models over the main grid, whole
                               run and steady state, against the staircase
                               and the address-map model
  lockstep_check               the existing options and the new stall model
                               at the staircase's points
  bank_rule_1r1w               the rule with one read and one write port
  weight_store                 1, 2 and 0 bank crossings of a weight tile
  bank_hash                    XOR skew hash vs modulo
  m_tiling                     512-word banks: M tile, utilization and layer
                               time against the whole layer on-chip
  host_options                 --split-host-stats and --prepacked-weights
                               against the default
  dram_wide_bus                DDR5-8400 on a 64-byte memory bus
  hypotheses_rtl_config.md     auto-generated H1/H2/H3 checks for the RTL's
                               configuration, with the proposal's D and with
                               the busier port's
  rtl_config_fit               model vs gem5 per rtlcfg suite, both forms
  rtl_config_saturation        the research question, RTL's configuration
  rtl_config_rule              both bank rules per (layer, N), and 25% fewer
  rtl_config_rule_vs_reuse     the same at every reuse M
  rtl_config_gemv              M = 1 at both clocks, and the BANKS floor
  rtl_config_stall             decoupled vs global stall, with the staircase
                               and the address-map model
  rtl_config_single_buffer     one weight register array against shadow ones
  rtl_config_m_tiling          512-word banks at the busier-port rule's counts
  psum_options                 where partial sums live, four options side by
                               side: banks each needs, both stall models
  psum_options_grid            the same at the main grid's bank counts
  rtl_config_rmw               partial sums read back: N/2 and ceil(5N/8) banks
  rtl_config_nodb              one weight register array, at the rule's banks
  rtl_config_n_tiling          512-word banks: M tiles only, M and N_out tiles
  n_tiling_1rw                 M and N_out tiles with one shared port
  hypotheses.md                auto-generated H1/H2/H3 checks


4. Adding RTL and Vivado results later
--------------------------------------

  RTL cycle counts (phase 2 cross-validation):
    cp analysis/templates/rtl_cycles.csv.example results/data/rtl_cycles.csv
    # fill in: workload,N,banks,cycles  (workload = the *_rtl names)
    make data SUITES=rtlval                    # gem5 side
    make data SUITES="rtlcfg_rtlval rtlcfg_rtlval_lockstep"
    make figures PLOTS=fig12   # uses the global-stall runs; set
                               # RTL_STALL_MODEL=decoupled, single_buffer or
                               # baseline for the other gem5 columns
    -> results/figures/fig12_gem5_vs_rtl,
       results/tables/rtl_crossvalidation (error %)

  Per-cycle comparison, when totals disagree:
    make run WL=bert_attn_qk_rtl N=8 BANKS=4 \
        ARGS='--ideal-dram --bank-trace build/gem5_trace.csv'
    # have the testbench $fwrite the same columns (gem5_model/README.txt 6.8)
    $PYTHON analysis/trace_diff.py build/gem5_trace.csv rtl_trace.csv

  Vivado (post-route) per array size and bank count, for EDP (proposal
  Sec. V):
    cp analysis/templates/vivado.csv.example results/data/vivado.csv
    # fill in: N,banks,fmax_mhz,array_power_w,bram_energy_pj_per_access
    make figures PLOTS=fig13
    EDP = (DRAM rank*.totalEnergy + array_power_w x activeCycles / Fmax
           + (spm_reads_total + spm_writes_total) x pJ per access)
          x layer time
    Blank cells count as no number yet.  (Mixing the 2 GHz gem5 clock with
    the FPGA's Fmax needs a decision: see the proposal review, item A3.)
    The older columns N,power_W,fmax_MHz,... still work, without the BRAM
    term.

  rtl_cycles.csv.example contains made-up numbers that only show the format.
  Do not copy it without replacing the values.


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
