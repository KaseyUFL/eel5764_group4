EEL5764 Group 4 -- gem5 model of the weight-stationary systolic array
=======================================================================

"Where Does Scratchpad SRAM Saturate?  Scaling N x N Systolic Arrays from
8x8 to 32x32 in RTL and gem5" -- gem5 half (proposal Sec. IV.A, Table II).

The model is a RISC-V system.  A RiscvTimingSimpleCPU host at 2 GHz with
32 KiB L1s, a 512 KiB L2 and DDR4-2400 runs a RISC-V program.  That
program prepares a layer in DRAM, programs the SystolicAccel through
memory-mapped registers, starts it, waits for DONE and checks the result.
(A CPU-less "standalone" mode is kept for quick debugging.)

Paths below are relative to the repository root.  Every command can be
run through make (see ../README.md); the scripts behind the make targets
are shown too, for debugging.  `source scripts/env.sh` sets the paths the
scripts use (GEM5_BIN, PYTHON, ...).


0. TL;DR
--------

  make build                                   # gem5 + host + workloads
  make run WL=bert_attn_qk N=16 BANKS=8        # one layer
  make smoke                                   # 8 small runs, must verify
  make data figures                            # every measurement, curve
                                               # and table (../analysis)


1. Folder layout
----------------

  gem5_model/
    README.txt                  this file
    src/accel/                  the SystolicAccel SimObject (EXTRAS=...)
      SystolicAccel.py            parameters and ports
      systolic_accel.hh/.cc       timing + functional model, registers,
                                  DMA engines
      SConscript                  SimObject, source, debug flags
                                  SystolicAccel and SystolicAccelDma
    host/
      systolic_host.c             RISC-V host program (driver + checker)
      systolic_host               the binary (after make host; not in git)
    configs/
      systolic_se.py              builds the system and runs one layer
    patches/
      riscv_se_uncacheable_tlb.patch   the one change to gem5 (5.3)

  Used from elsewhere in the repository:
    scripts/build_gem5.sh       fetch + patch + build gem5.opt (make gem5)
    scripts/build_host.sh       cross-compiles the host program (make host)
    scripts/run_workload.sh     one workload at one (N, BANKS) point
    scripts/run_all_workloads.sh  all 4 workloads, banked + ideal SPM
    analysis/experiments.py     every experiment sweep (make data)
    analysis/parse_stats.py     stats.txt -> CSV (all Table II metrics)
    analysis/model.py           eq. (1)/(2) predictions, Table I

  gem5 itself is not modified apart from the patch.  The accelerator is
  compiled in through EXTRAS into build/RISCV_SYSTOLIC of the gem5
  checkout; a stock build/RISCV there is untouched.


2. Environment and build
------------------------

2.1 Settings.  scripts/env.sh sets the paths; machine-specific settings
go in env.local.sh at the repository root (copy env.local.sh.example).
The gem5 used for results/ was built with a conda env (Python 3.10) and
conda-forge gcc 13, through the gem5_build_env hook shown in the example.
Without the hook, the system gcc and python3-config are used.

2.2 gem5 (first time 20-30 min on 24 threads; afterwards 1-2 min):

  make gem5                       # = bash scripts/build_gem5.sh
  make gem5 BUILD_JOBS=8          # fewer jobs if RAM is tight

  It clones gem5 v25.1.0.1 into third_party/gem5 (or uses GEM5_DIR),
  applies patches/*.patch, and runs
    scons defconfig build/RISCV_SYSTOLIC build_opts/RISCV   (first time)
    scons build/RISCV_SYSTOLIC/gem5.opt EXTRAS=<repo>/gem5_model/src

  Rebuild after editing systolic_accel.*, SystolicAccel.py or the patch.
  Config edits (systolic_se.py) need no rebuild.

2.3 RISC-V host program:

  make host                       # = bash scripts/build_host.sh

  It uses the first compiler it finds: $RISCV_CC,
  riscv64-linux-gnu-gcc (sudo apt install gcc-riscv64-linux-gnu), zig's
  clang from third_party/zig-venv (make zig: pip-installs ziglang 0.16.0,
  no sudo), or zig from ~/riscv-tools/venv.  The output is a static
  RISC-V Linux (musl) binary that gem5 SE runs.  The results were
  produced with zig 0.16.0.

2.4 gem5 version.  gem5 v25.1.0.1, not v24 as the proposal says.

2.5 Sanity check -- stock RISC-V hello world on this binary:

  source scripts/env.sh
  $GEM5_BIN -d build/m5out/hello $GEM5_DIR/configs/deprecated/example/se.py \
      --cpu-type=RiscvTimingSimpleCPU --caches --l2cache \
      --cmd=$GEM5_DIR/tests/test-progs/hello/bin/riscv/linux/hello


3. Workloads
------------

  make workloads                  # = $PYTHON workloads/gen_workloads.py

  full layers:     resnet50_conv3x  bert_ffn1  bert_attn_qk  llama7b_gemv
  reduced (RTL):   <name>_rtl       (same M, K and N_out <= 96)
  reuse sweep:     reuse_M1 ... reuse_M512   (M x 1024 x 1024, synthetic)

See workloads/README.txt for shapes and sources.


4. Running the model
--------------------

4.1 One layer:  run_workload.sh WORKLOAD N BANKS [systolic_se options]

  make run WL=resnet50_conv3x N=16 BANKS=8
  make run WL=bert_ffn1 N=32 BANKS=21                    # 21 banks
  make run WL=bert_ffn1 N=16 BANKS=8 ARGS=--no-double-buffer
  make run WL=bert_ffn1 N=16 BANKS=8 ARGS='--mode sw'
  make run WL=bert_ffn1 N=16 BANKS=8 ARGS=--standalone

  or, with the same arguments:

  source scripts/env.sh
  bash scripts/run_workload.sh llama7b_gemv 8 4

  Output: build/m5out/<WL>_N<N>_B<BANKS>/ (override with OUT=dir).
  What you see:
    HOST: accelerator layer M=784 K=1152 N_out=128, array N=16
    info: SystolicAccel: layer 1 start: GEMM 784x1152x128, 576 tiles ...
    info: SystolicAccel done: 569751 cycles (ideal 452399), PE
          utilization 0.7926, ... layer 748485 cycles (load 135057,
          store 43677), verify: PASS
    HOST: result mode=accel mismatches=0 checked=100352 ...
    HOST: PASS
  Two independent checks: the accelerator compares its result with a
  reference GEMM built from what the host put in DRAM ("verify: PASS"),
  and the RISC-V host compares C in DRAM with C_golden.bin
  ("HOST: PASS").

  Wall time (host-driven, N=16): bert_attn_qk <1 s, resnet50 ~5 s,
  bert_ffn1 ~10 s, llama7b_gemv ~40 s.  GEMV is slow only because the
  host packs 16 MiB of weights before the layer (50-70 M RISC-V
  instructions, most of them cache misses).  Software mode (--mode sw)
  runs the whole GEMM on the CPU: ~8 min (ResNet), ~20 min (BERT FFN1).

4.2 All four workloads, banked vs. ideal scratchpad:

  make smoke                      # reduced layers, N=8, 4 banks (~10 s)
  bash scripts/run_all_workloads.sh 16 8
  bash scripts/run_all_workloads.sh 16 8 --rtl

  Each prints a table and fails unless every run verified.

4.3 Proposal measurements, curves and tables:
  see analysis/README.txt  (make data figures)

4.4 systolic_se.py options (--help for the full list):
    --workload NAME        workload folder under workloads/out/gem5
    -N/--array-dim N       array size, multiple of 8                  [16]
    -B/--banks BANKS       any count >= 1; B = 8*BANKS bytes/cycle      [8]
    --bank-mapping auto|interleave|modulo                           [auto]
                           auto = interleave for powers of two, else
                           modulo (bank = word mod BANKS)
    --arbitration bank|lane   per-bank queues / per-lane ports      [bank]
    --bank-policy fifo|reads_first   order inside a bank queue      [fifo]
                           reads_first serves reads ahead of queued
                           writes (see 6.4)
    --psum-queue 8         partial-sum write buffer entries per lane
    --no-double-buffer     no shadow weight registers (T = M + N)
    --mode accel|sw        offload, or run the GEMM in software  [accel]
    --standalone           no CPU (accelerator started by the config)
    --ideal-spm            infinite-bandwidth scratchpad (baseline)
    --ideal-dram           weights appear in the SPM for free
    --clock 2GHz           accelerator + scratchpad clock.  Use e.g.
                           250MHz for an FPGA-like array; the host, the
                           memory bus and DRAM keep their own timing, so
                           DRAM then delivers 8x more bytes per array cycle
    --cpu-clock 2GHz       host CPU, caches, memory bus, I/O cache
    --l1-size 32KiB  --l2-size 512KiB
    --spm-latency 1        bank latency (accelerator cycles)
    --lookahead 8          read-ahead depth (array cycles)
    --weight-buffer 16KiB  SPM bytes for DMA'd weight tiles
    --weight-slots K       override: prefetch depth in tiles
    --dma-outstanding 64   in-flight 64 B DRAM reads
    --dma-lanes 8          DMA lanes (lane arbitration only)
    --mem-type DDR4_2400_8x8   any gem5 DRAM interface, e.g.
                           DDR3_1600_8x8, DDR5_6400_4x8, DDR5_8400_4x8
    --dram-size 1GiB
    --max-cycles C         stop the accelerator early (debugging)
    --no-verify            skip the accelerator's own check
    --gemm M K N_OUT       standalone only: synthetic random GEMM


5. The system (configs/systolic_se.py)
--------------------------------------

  RiscvTimingSimpleCPU 2 GHz -- L1I/L1D 32 KiB -- L2 512 KiB --+
                                                               |
                          membus (SystemXBar) -----------------+--- DDR4-2400
                            |                   |                  (1 GiB)
                  accel.pio (0xA0000000)   iocache (32 KiB)
                            |                   |
                     +-----------------------------------+
                     |   SystolicAccel (N x N, INT8)     |
                     +-----------------------------------+
                            | spm_ports
                     spm_xbar (Noncoherent, 2x clock)
                            |
                     spm[0 .. BANKS-1]  SimpleMemory, 1 cycle,
                                        one 8-byte word per cycle

5.0 Clocks.  There are two clock domains.  The system domain (host CPU,
L1/L2, memory bus, I/O cache, DRAM controller) runs at --cpu-clock.  The
accelerator and its scratchpad banks run at --clock, and spm_xbar at twice
that.  Scratchpad behaviour is counted in array cycles, so it does not
depend on --clock; what changes is how many bytes DRAM delivers per array
cycle.

5.1 Memory map.  DRAM is 1 GiB.  The SE page allocator may only use the
lower 512 MiB (system.mem_ranges).  The upper 512 MiB (physical
0x20000000) is the accelerator's DMA buffer; the config maps it into the
host program at virtual 0x200000000000 (cacheable).  The registers are
at physical 0xA0000000, mapped at virtual 0x300000000000 (uncacheable).
The scratchpad is at 0x80000000; only the accelerator can reach it.
The host program runs with the workload directory as its working
directory and gets file names (A.bin, W.bin, C_golden.bin), not paths.
Its arguments are copied onto its stack, so absolute paths would make the
stack layout, and with it the host's timing, depend on where the
repository is cloned.

5.2 Registers (64-bit; SystolicAccel::Reg and host/systolic_host.c):
    0x00 CTRL          W  1 = start the layer
    0x08 STATUS        R  bit0 busy, bit1 done, bit2 error
    0x10 M, 0x18 K, 0x20 N_OUT            RW  GEMM shape
                          (K a multiple of 8, N_out even)
    0x28 A_ADDR        RW  A, row-major INT8        (64-byte aligned)
    0x30 W_ADDR        RW  packed weight tiles      (64-byte aligned)
    0x38 C_ADDR        RW  C, row-major INT32       (64-byte aligned)
    0x40 LAYER_CYCLES  R   last layer, all phases
    0x48 ARRAY_DIM     R   N
    0x50 COMPUTE_CYCLES R  compute phase
    0x58 VERIFY        R   the accelerator's own check (0 = pass)

5.3 The gem5 patch.  RISC-V's SE-mode TLB ignored Process.map(...,
cacheable=False), so every register access became a cacheable line fill
(x86's SE TLB honours the flag).  src/arch/riscv/tlb.cc now sets
UNCACHEABLE | STRICT_ORDER on requests to such pages, as x86 does.  This
is the only change to gem5 itself (patches/riscv_se_uncacheable_tlb.patch,
applied by make gem5).  It would be a reasonable upstream patch.

5.4 Coherent DMA through an I/O cache.  The host writes A and the packed
weights with ordinary stores, so the data may still be dirty in its L1/L2
when the accelerator reads it.  The accelerator also writes C into lines
the host may hold.  gem5's classic caches cannot merge a DMA write into
another cache's dirty line from a port on the memory bus.  So, like
gem5's full-system IOCache (and a real SoC's coherent accelerator port),
the DMA port goes through a 32 KiB cache with 128 MSHRs.  C is written as
whole-line WriteLineReq packets, which invalidate the host's copies.

5.5 Scratchpad banks.  Each bank is a SimpleMemory owning every
BANKS-th 8-byte word.  Power-of-two BANKS use the crossbar's address
interleaving.  Any other count (e.g. ceil(D/8) = 6, 11, 21) uses modulo
mapping: each bank is a contiguous range, and the accelerator's
bank-select logic maps word w to bank w mod BANKS, row w div BANKS.  A
real design would use counters in the address generators, not a divider.
Bank bandwidth is exactly one 8-byte word per accelerator cycle: 62
ticks/byte, rather than gem5's rounded 63.  spm_xbar runs at 2x the clock
so it never limits a bank.


6. How the accelerator model works (src/accel/systolic_accel.cc)
----------------------------------------------------------------

6.1 A host-driven layer has three phases:
  LoadA    the DMA reads A (row-major) from DRAM, 64 B at a time, and
           scatters it into the SPM blocked by K-block (N columns per
           row); padded columns are zeroed.  The weight DMA may already
           prefetch tiles.
  Compute  the array runs the static schedule (6.2) with stalls.
  StoreC   the INT32 accumulators are read from the SPM and written to
           DRAM row-major, one 64-byte line at a time.
  Standalone runs have only the compute phase.  A is preloaded at tick 0.

6.2 Tiling and static schedule.  C = A W is cut into N x N weight
tiles: ceil(K/N) K-blocks x ceil(Nout/N) output blocks, ordered
nb-outer, kb-inner.  All M rows stream through each tile.  Partial sums
accumulate in the SPM across kb; as eq. (1) assumes, the read-modify-
write is local and the array only writes 4 bytes per output.  The
schedule matches the team's demo RTL testbench (tb_systolic.sv, kept
with the RTL work), with T = max(M, N) (or M + N
single-buffered):
    weight row r of tile t          cycle tT + 1 + r
    activation row m of tile t      cycle S_t + m,  S_t = (t+1)T
    psum lane j (cols 2j, 2j+1)     cycle S_t + m + N + 2j + 1
  Stall-free length ("ideal"): nTiles*T + M - 1 + 2N.

6.3 Global stall.  The array advances one schedule cycle per clock only
if (a) that cycle's activation and weight words are back from the banks,
(b) the weight tile has arrived from DRAM, and (c) the psums leaving
the array have room in the write buffer (psum_queue_depth x N/2 entries).
Otherwise it freezes for a clock and the cause is counted.  Reads are
issued up to --lookahead cycles early.

6.4 Arbitration.  "bank" (default): one request queue per bank, each
bank serves its oldest request per cycle, and supply = 8*BANKS
bytes/cycle (eq. 2).  "lane": one port per edge lane, and lanes contend
in the crossbar (a lockstep design).  With lanes, a lost arbitration is
never made up.

Within a bank queue the default order is oldest first ("fifo").
With --bank-policy reads_first a read goes ahead of queued writes:
the array stalls on a late read, while a write only waits in the
write buffer.  This matters when BANKS divides N - 1.  In one cycle,
partial-sum lane j writes a row two cycles older than lane j-1 (a row
is N/2 words) and one word further along it, so neighbouring lanes are
N - 1 words apart.  With a bank count that divides N - 1, every lane
addresses the SAME bank in the same cycle, and reads queue behind a
burst of N/2 writes (utilization 0.46 at N = 32 with 31 banks).
reads_first plus a 16-entry write buffer restores 0.99.  Powers of two
never divide N - 1 (it is odd), and neither do the rule's counts 6, 11,
16, 21.

6.5 Write buffer.  The default depth changed from 4 to 8 entries per
psum lane.  With 4, the banks' short-term load imbalance near D ~ B
filled the buffer and stalled the array (N=24 at 16 banks: 0.83).  With 8
the same point reaches 0.995, and the bank-matching rule holds.  8
entries x 8 B x N/2 lanes is 1 KiB at N=32.

6.6 Weight DMA.  Tiles are pre-packed tile-major in DRAM, by the host
program (or offline).  After packing, the host flushes its caches (it
writes through a 4 MiB scratch buffer), so the DMA really reads the
weights from DRAM and not from the host's L2.  Each tile is written into
the scratchpad by the DMA and then read out by the array, so weights
cross the banks twice: scratchpad demand is (5NM + 2N^2)/max(M, N), not
eq. (1)'s (5NM + N^2)/max(M, N).  The difference is 1% at M = 784 and
N = 32, 5% at M = 128, and a factor of nearly two at M = 1.  The DMA reads 64-byte lines into a ring of
weight_slots tile buffers in the SPM (--weight-buffer / N^2 tiles; about
64 lines must be in flight to reach the ~14 GB/s DDR4 sustains).

6.7 Checks.  Packets carry real data.  At the end of a layer the
accelerator compares its accumulators with a reference GEMM computed
from the operands actually in DRAM (or the files, standalone), plus the
SPM contents and (host-driven) C in DRAM.  The host program then
compares C with C_golden.bin.


7. Statistics and the proposal's Table II
-----------------------------------------

Host-driven stats.txt has three dumps, because the host calls
m5_dump_reset_stats() before START and after DONE:
    dump 0  setup: read files, pack weights
    dump 1  THE LAYER  <- every metric below comes from here
    dump 2  verification and exit
parse_stats.py picks the right dump using systolic_run.json, which the
config writes next to stats.txt.

  PE utilization     system.accel.peUtilization (compute phase)
                     system.accel.layerUtilization (incl. DMA phases)
  Stall cycles       system.accel.spmStallCycles (+Read/Write parts),
                     dmaStallCycles
  Delivered BW       system.spm*.bwRead::total, bwWrite::total (bytes/s;
                     parse_stats divides by the clock -> bytes/cycle)
  Bank balance       system.spm_xbar.pktCount_* per bank (requests +
                     responses), system.accel.bankAccesses::<bank>
  Scratchpad AMAT    system.accel.avgSpmLatency (cycles)
  Layer time         simSeconds / simTicks of dump 1,
                     system.accel.layerCycles, loadCycles, storeCycles
  Speedup            layer time of --mode sw / layer time (analysis)
  Host CPI / IPC     system.cpu.cpi, system.cpu.ipc (dump 1)
  DRAM pressure      system.mem_ctrl.dram.busUtil, avgRdBWSys
  EDP                sum of system.mem_ctrl.dram.rank*.totalEnergy x
                     simSeconds (+ array energy from Vivado, later)
  Verification       system.accel.verifyMismatches (0), "HOST: PASS"


8. Debugging
------------

  * Trace the accelerator:
      DEBUG=SystolicAccel,SystolicAccelDma bash \
          scripts/run_workload.sh bert_attn_qk_rtl 8 4
    (the trace is build/m5out/<run>/trace.out)
    Add NoncoherentXBar (scratchpad packets, "RETRY" = bank busy),
    MemCtrl or Cache for the DRAM side, Exec for the RISC-V
    instructions.  Traces get large: use _rtl layers.
  * The host's printf output (HOST: ...) is in simout.txt with -re.
  * verify FAIL -- the first mismatches are printed as warnings:
      array  model output vs reference   (datapath / tiling)
      spm    SPM contents vs model       (psum write path)
      dram   C in DRAM vs model          (C store DMA)
      golden files disagree              (rerun make workloads)
  * "HOST: FAIL" with verify PASS: host-side layout bug
    (host/systolic_host.c packing or addresses).
  * "bad layer registers": K not a multiple of 8, N_out odd, or an
    address not 64-byte aligned.
  * gem5.debug for gdb (in $GEM5_DIR, after source scripts/env.sh):
      scons --ignore-style build/RISCV_SYSTOLIC/gem5.debug \
          EXTRAS=$SYSTOLIC_ROOT/gem5_model/src -j22


9. Not done yet
---------------
  * RTL cross-validation (phase 2): run the rtlval suite and compare with
    Questa cycle counts (fig12, from results/data/rtl_cycles.csv).
  * Array energy / Fmax / hardware cost need Vivado (fig13 then includes
    them).
  * Dual-port (1 read + 1 write) BRAM banks are not modeled: each bank
    serves one access per cycle.
