# EEL5764 Group 4 -- gem5 system for the SystolicAccel (RISC-V, SE mode).
#
#  Default (host-driven, proposal Sec. IV.A):
#
#   RiscvTimingSimpleCPU 2 GHz ---- L1I 32 KiB / L1D 32 KiB ---- L2 512 KiB
#                                                                    |
#                                               membus (SystemXBar) -+
#                                                |        |          |
#                                  accel.pio (registers)  |   DDR4-2400 MemCtrl
#                                                |        |
#                            +-------------------------------+
#                            |  SystolicAccel (N x N, INT8)  |--- dram_port
#                            +-------------------------------+   (DMA: A, W, C)
#                               spm_ports |
#                                         v
#                                spm_xbar (Noncoherent) -> spm[0..BANKS-1]
#                                SimpleMemory, one 8-byte word/cycle each
#
#  A RISC-V host program (gem5_model/host/systolic_host) reads the
#  workload files, puts A and the packed weight tiles in a DRAM buffer that
#  is mapped into its address space, writes the accelerator's registers,
#  starts the layer, polls for DONE and checks C against the golden file.
#  The layer is bracketed by m5 dump/reset pseudo-instructions, so
#  stats.txt dump #1 is exactly the layer (--mode sw runs the same GEMM in
#  software on the CPU instead, for the speedup metric).
#
#  --standalone: no CPU; the accelerator gets the layer from parameters,
#  starts at tick 0 and ends the simulation (fast; used while debugging).
#
#   $GEM5_BIN -d build/m5out/test \
#       gem5_model/configs/systolic_se.py --workload bert_attn_qk \
#       --array-dim 16 --banks 8
#
#  (scripts/run_workload.sh wraps this; GEM5_BIN is set by scripts/env.sh)
#
# Run with --help for all options.

import argparse
import json
import math
import os
import sys

import m5
from m5.objects import *
from m5.util.convert import toFrequency, toLatency, toMemorySize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_WL_DIR = os.environ.get(
    "SYSTOLIC_WORKLOADS",
    os.path.join(ROOT, "workloads", "out", "gem5"),
)
DEFAULT_HOST_BIN = os.path.join(ROOT, "gem5_model", "host", "systolic_host")

SPM_BASE = 0x80000000
PIO_ADDR = 0xA0000000
BANK_WORD = 8  # bytes per bank per cycle (one BRAM36 in 72-bit SDP mode)
# Host-driven memory map: the process allocator only uses the lower half of
# DRAM; the upper half is the accelerator's DMA buffer, mapped into the
# host program at SHARED_VA (cacheable, kept coherent by the memory bus).
SHARED_VA = 0x200000000000
PIO_VA = 0x300000000000


def fail(msg):
    """Print an error and stop.  (In a gem5 config, sys.exit("text") makes
    gem5 abort with a pybind11 cast error instead of printing the text.)"""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


p = argparse.ArgumentParser(description="SystolicAccel system")
g = p.add_argument_group("workload")
g.add_argument("--workload", default="bert_attn_qk",
               help="Directory name under --workload-dir (see "
                    "workloads/out/workloads.json)")
g.add_argument("--workload-dir", default=DEFAULT_WL_DIR)
g.add_argument("--gemm", nargs=3, type=int, metavar=("M", "K", "N_OUT"),
               help="Standalone only: synthetic GEMM with random INT8 data")
g.add_argument("--seed", type=int, default=1,
               help="Random-data seed for --gemm")
g.add_argument("--no-verify", action="store_true",
               help="Skip the accelerator's own end-of-layer check")
g.add_argument("--c-out", default="",
               help="Write the simulated INT32 C matrix to this file")

g = p.add_argument_group("host (RISC-V)")
g.add_argument("--standalone", action="store_true",
               help="No CPU: the accelerator runs the layer on its own")
g.add_argument("--mode", choices=["accel", "sw"], default="accel",
               help="accel: offload the layer; sw: GEMM in software on the "
                    "RISC-V core (speedup baseline)")
g.add_argument("--host-bin", default=DEFAULT_HOST_BIN,
               help="RISC-V host program (build: make host)")
g.add_argument("--cpu-clock", default="2GHz",
               help="Host CPU, caches and memory bus clock")
g.add_argument("--l1-size", default="32KiB")
g.add_argument("--l2-size", default="512KiB")

g = p.add_argument_group("accelerator / scratchpad")
g.add_argument("--array-dim", "-N", type=int, default=16,
               help="N (array is N x N, multiple of 8)")
g.add_argument("--banks", "-B", type=int, default=8,
               help="Scratchpad banks; B = 8*BANKS bytes/cycle.  Any "
                    "count; non-powers of two use modulo bank mapping")
g.add_argument("--bank-mapping", choices=["auto", "interleave", "modulo"],
               default="auto",
               help="auto: interleave for powers of two, else modulo")
g.add_argument("--arbitration", choices=["bank", "lane"], default="bank",
               help="bank: per-bank request queues (supply = 8*BANKS "
                    "bytes/cycle); lane: one port per edge lane, lanes "
                    "contend for banks (lockstep-RTL-like)")
g.add_argument("--bank-policy", choices=["fifo", "reads_first"],
               default="fifo",
               help="Order within a bank's request queue: oldest first, or "
                    "reads ahead of queued writes")
g.add_argument("--no-double-buffer", action="store_true",
               help="Disable shadow weight registers (T = M + N)")
g.add_argument("--clock", default="2GHz",
               help="Accelerator + scratchpad clock (e.g. 250MHz for an "
                    "FPGA-like array).  The host, caches, memory bus and "
                    "DRAM keep --cpu-clock / their own timing")
g.add_argument("--spm-latency", type=int, default=1,
               help="Bank access latency in accelerator cycles")
g.add_argument("--lookahead", type=int, default=8,
               help="Array cycles of read-ahead in the address generators")
g.add_argument("--psum-queue", type=int, default=8,
               help="Partial-sum write buffer entries per psum lane")
g.add_argument("--weight-buffer", default="16KiB",
               help="Scratchpad bytes reserved for DMA'd weight tiles; "
                    "sets the prefetch depth to buffer / N^2 tiles (>= 2)")
g.add_argument("--weight-slots", type=int, default=0,
               help="Override: weight tile buffers (DMA prefetch depth)")
g.add_argument("--dma-lanes", type=int, default=8,
               help="8-byte DMA write lanes into the scratchpad")
g.add_argument("--dma-outstanding", type=int, default=64,
               help="Max in-flight 64-byte DRAM reads")
g.add_argument("--ideal-spm", action="store_true",
               help="Infinite-bandwidth scratchpad (analytical baseline)")
g.add_argument("--ideal-dram", action="store_true",
               help="Weights appear in the scratchpad for free")
g.add_argument("--max-cycles", type=int, default=0,
               help="Stop the accelerator after this many cycles (0 = off)")

g = p.add_argument_group("memory system")
g.add_argument("--mem-type", default="DDR4_2400_8x8",
               help="DRAM interface class (DDR4_2400_8x8, LPDDR5_..., ...)")
g.add_argument("--dram-size", default="1GiB")
args = p.parse_args()

N = args.array_dim
BANKS = args.banks
host = not args.standalone
if N < 8 or N % 8:
    fail("--array-dim must be a multiple of 8")
if BANKS < 1:
    fail("--banks must be >= 1")
pow2 = BANKS & (BANKS - 1) == 0
mapping = args.bank_mapping
if mapping == "auto":
    mapping = "interleave" if pow2 else "modulo"
if mapping == "interleave" and not pow2:
    fail("--bank-mapping interleave needs a power-of-two --banks")

# ---------------------------------------------------------------- workload
if args.gemm:
    if host:
        fail("--gemm is standalone-only (the host program reads files)")
    M, K, NOUT = args.gemm
    wl = {"name": f"gemm_{M}x{K}x{NOUT}"}
    a_file = w_file = c_file = ""
else:
    wdir = os.path.abspath(os.path.join(args.workload_dir, args.workload))
    meta_path = os.path.join(wdir, "meta.json")
    if not os.path.exists(meta_path):
        fail(f"No workload at {meta_path}. Run 'make workloads' "
                 f"first.")
    with open(meta_path) as f:
        wl = json.load(f)
    M, K, NOUT = wl["M"], wl["K"], wl["N_out"]
    a_file = os.path.join(wdir, wl["A"]["file"])
    w_file = os.path.join(wdir, wl["W"]["file"])
    c_file = os.path.join(wdir, wl["C_golden"]["file"])
if host and (K % 8 or NOUT % 2):
    fail("host-driven layers need K % 8 == 0 and an even N_out")

# Weight prefetch depth: a fixed-size staging buffer holds more, smaller
# tiles at small N.  Deep prefetch is what lets the DMA keep enough DRAM
# reads in flight (Little's law) and ride out DDR4 refresh (~350 ns).
if args.weight_slots:
    slots = args.weight_slots
else:
    slots = max(2, int(toMemorySize(args.weight_buffer)) // (N * N))


def align(x, a):
    return (x + a - 1) // a * a


# Scratchpad capacity: blocked A + weight slots + INT32 accumulators, the
# same layout SystolicAccel computes.  Each bank gets a power-of-two share.
nkb, nnb = -(-K // N), -(-NOUT // N)
need = align(align(nkb * M * N, 64) + slots * N * N, 64) + nnb * M * N * 4
bank_bytes = max(8 * 1024, 1 << math.ceil(math.log2(-(-need // BANKS))))
spm_size = bank_bytes * BANKS

freq = toFrequency(args.clock)
dbuf = not args.no_double_buffer
T = max(M, N) if dbuf else M + N
d_model = (5 * N * M + N * N) / max(M, N)
print(f"[systolic_se] workload {wl['name']}: M={M} K={K} N_out={NOUT}; "
      f"{'host-driven (RISC-V, mode=' + args.mode + ')' if host else 'standalone'}")
print(f"[systolic_se] N={N} BANKS={BANKS} (B={BANKS * BANK_WORD} bytes/cycle, "
      f"{mapping}) double_buffer={dbuf} T={T} tiles={nkb * nnb} "
      f"arbitration={args.arbitration} weight_slots={slots}")
print(f"[systolic_se] eq.(1) D={d_model:.1f} bytes/cycle, "
      f"predicted util~{min(1.0, BANKS * BANK_WORD / d_model) * M / T:.3f}; "
      f"SPM {spm_size // 1024} KiB")

# ------------------------------------------------------------------ system
system = System()
# Two clock domains.  The system domain (host CPU, caches, memory bus, I/O
# cache, DRAM controller) runs at --cpu-clock.  The accelerator and its
# scratchpad run at --clock, so slowing the array to an FPGA-like
# frequency does not also slow the bus that feeds it.
system.clk_domain = SrcClockDomain(clock=args.cpu_clock,
                                   voltage_domain=VoltageDomain())
system.accel_clk_domain = SrcClockDomain(
    clock=args.clock, voltage_domain=system.clk_domain.voltage_domain
)
system.mem_mode = "timing"
dram_size = int(toMemorySize(args.dram_size))
dram_range = AddrRange(dram_size)
shared_pa = dram_size // 2
shared_size = dram_size - shared_pa
# The SE page allocator hands out pages from system.mem_ranges only, so
# limiting it to the lower half keeps the DMA buffer free.
system.mem_ranges = [AddrRange(shared_pa)] if host else [dram_range]

system.membus = SystemXBar()
system.system_port = system.membus.cpu_side_ports
mem_cls = getattr(m5.objects, args.mem_type)
system.mem_ctrl = MemCtrl(dram=mem_cls(range=dram_range))
# Peak data-bus bandwidth of this DRAM model: bytes per beat x beats per
# clock / tCK (DDR4-2400 x64: 8 B x 2 / 0.833 ns = 19.2 GB/s).
dram_peak_bps = (int(mem_cls.device_bus_width) * int(mem_cls.devices_per_rank)
                 / 8 * int(mem_cls.beats_per_clock)
                 / toLatency(str(mem_cls.tCK)))
system.mem_ctrl.port = system.membus.mem_side_ports

if host:
    class L1(Cache):
        assoc = 8
        tag_latency = 2
        data_latency = 2
        response_latency = 2
        mshrs = 16
        tgts_per_mshr = 20

    class L2(Cache):
        assoc = 16
        tag_latency = 12
        data_latency = 12
        response_latency = 12
        mshrs = 32
        tgts_per_mshr = 12

    system.cpu = RiscvTimingSimpleCPU()
    system.cpu.icache = L1(size=args.l1_size)
    system.cpu.dcache = L1(size=args.l1_size)
    system.cpu.icache_port = system.cpu.icache.cpu_side
    system.cpu.dcache_port = system.cpu.dcache.cpu_side
    system.l2bus = L2XBar()
    system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
    system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
    system.l2 = L2(size=args.l2_size)
    system.l2.cpu_side = system.l2bus.mem_side_ports
    system.l2.mem_side = system.membus.cpu_side_ports
    system.cpu.createInterruptController()

    host_bin = os.path.abspath(args.host_bin)
    if not os.path.exists(host_bin):
        fail(f"host program not found: {host_bin}\n"
                 "build it: make host")
    shared_need = (align(M * K, 4096) + align(nkb * nnb * N * N, 4096) +
                   M * NOUT * 4)
    if shared_need > shared_size:
        fail(f"layer needs {shared_need} bytes of DMA buffer; raise "
                 f"--dram-size (buffer = upper half of DRAM)")
    # The program runs in the workload directory and gets file names, not
    # paths: argv is copied onto the simulated stack, so absolute paths
    # would make the stack layout, and with it the timing, depend on where
    # the repository was cloned.
    process = Process(
        executable=host_bin,
        cwd=wdir,
        cmd=["systolic_host", args.mode,
             os.path.basename(a_file), os.path.basename(w_file),
             os.path.basename(c_file), str(M), str(K), str(NOUT),
             hex(SHARED_VA), hex(shared_pa), hex(PIO_VA)],
    )
    system.workload = SEWorkload.init_compatible(host_bin)
    system.cpu.workload = process
    system.cpu.createThreads()
elif nkb * nnb * N * N > dram_size:
    fail("Packed weights do not fit in --dram-size")

# Banked scratchpad.  The crossbar runs at 2x the accelerator clock: a
# gem5 crossbar layer is busy for (1 + payload) of its own cycles per data
# packet, so at 2x it passes one 8-byte word per bank (and per lane) per
# accelerator cycle.  Each bank's own bandwidth limit is exactly 8 bytes
# per accelerator cycle.
system.spm_clk_domain = SrcClockDomain(
    clock=f"{int(2 * freq)}Hz", voltage_domain=system.clk_domain.voltage_domain
)
system.spm_xbar = NoncoherentXBar(
    width=BANK_WORD,
    frontend_latency=1,
    forward_latency=0,
    response_latency=1,
    clk_domain=system.spm_clk_domain,
)
lat_ns = args.spm_latency / freq * 1e9
# SimpleMemory stores its bandwidth as an integer number of ticks per byte
# and stays busy for size * that after each access.  8 B/cycle at 2 GHz is
# 62.5 ticks/byte, which gem5 rounds to 63, so a bank would be busy 504
# ticks -- past the next 500-tick edge -- and lose a cycle on every
# back-to-back access.  Round DOWN instead: the bank frees within the
# cycle, and the one-request-per-cycle ports keep it at one 8-byte word
# per cycle.
period_ticks = round(1e12 / freq)
ticks_per_byte = period_ticks // BANK_WORD
bank_bw = f"{1e12 / ticks_per_byte:.0f}B/s"
banks = []
bits = int(math.log2(BANKS)) if pow2 else 0
for i in range(BANKS):
    if mapping == "modulo":
        # Each bank owns a contiguous range; the accelerator's bank-select
        # logic maps 8-byte word w to bank w % BANKS, row w // BANKS.
        rng = AddrRange(SPM_BASE + i * bank_bytes, size=bank_bytes)
    elif bits:
        rng = AddrRange(SPM_BASE, size=spm_size, intlvHighBit=2 + bits,
                        intlvBits=bits, intlvMatch=i)
    else:
        rng = AddrRange(SPM_BASE, size=spm_size)
    banks.append(SimpleMemory(range=rng, latency=f"{lat_ns}ns",
                              latency_var="0ns", bandwidth=bank_bw,
                              clk_domain=system.accel_clk_domain))
system.spm = banks
for b in system.spm:
    b.port = system.spm_xbar.mem_side_ports

# Accelerator
system.accel = SystolicAccel(
    clk_domain=system.accel_clk_domain,
    array_dim=N,
    double_buffer=dbuf,
    host_driven=host,
    pio_addr=PIO_ADDR,
    c_out_file=args.c_out, seed=args.seed, verify=not args.no_verify,
    spm_base=SPM_BASE, spm_size=f"{spm_size}B", w_dram_base=0,
    lookahead=args.lookahead, psum_queue_depth=args.psum_queue,
    weight_slots=slots, dma_lanes=args.dma_lanes,
    dma_max_outstanding=args.dma_outstanding,
    ideal_spm=args.ideal_spm, ideal_dram=args.ideal_dram,
    max_cycles=args.max_cycles,
    spm_banks=BANKS, bank_mapping=mapping, arbitration=args.arbitration,
    bank_policy=args.bank_policy,
)
if not host:
    system.accel.gemm_m = M
    system.accel.gemm_k = K
    system.accel.gemm_n = NOUT
    system.accel.a_file = a_file
    system.accel.w_file = w_file
    system.accel.c_golden_file = c_file
if args.arbitration == "bank":
    nports = BANKS
else:
    nports = N // 8 + N // 8 + N // 2 + args.dma_lanes
for i in range(nports):
    system.accel.spm_ports[i] = system.spm_xbar.cpu_side_ports
if host:
    # Coherent DMA: like gem5's full-system IOCache, the accelerator's DMA
    # port goes through a small cache on the memory bus, so its reads see
    # data the host still holds dirty in L1/L2, and its writes of C
    # invalidate the host's copies.  128 MSHRs keep 64+ DMA reads in
    # flight (the DDR4 bandwidth needs them).
    system.iocache = Cache(
        size="32KiB", assoc=8, tag_latency=2, data_latency=2,
        response_latency=2, mshrs=128, tgts_per_mshr=12,
        addr_ranges=[dram_range],
    )
    system.accel.dram_port = system.iocache.cpu_side
    system.iocache.mem_side = system.membus.cpu_side_ports
    system.accel.pio = system.membus.mem_side_ports
else:
    system.accel.dram_port = system.membus.cpu_side_ports

# Run description for parse_stats.py (which stats dump is the layer, etc.)
run_info = {
    "workload": wl["name"], "M": M, "K": K, "N_out": NOUT, "N": N,
    "banks": BANKS, "bank_mapping": mapping, "double_buffer": dbuf,
    "arbitration": args.arbitration, "bank_policy": args.bank_policy,
    "ideal_spm": args.ideal_spm,
    "ideal_dram": args.ideal_dram, "host_driven": host,
    "mode": args.mode if host else "standalone",
    "layer_dump": 1 if host else 0, "psum_queue": args.psum_queue,
    "weight_slots": slots, "clock_hz": freq,
    "cpu_clock_hz": toFrequency(args.cpu_clock),
    "mem_type": args.mem_type, "dram_peak_bps": dram_peak_bps,
    "lookahead": args.lookahead, "spm_latency": args.spm_latency,
    "dma_outstanding": args.dma_outstanding,
    "weight_buffer_bytes": slots * N * N,
}
with open(os.path.join(m5.options.outdir, "systolic_run.json"), "w") as f:
    json.dump(run_info, f, indent=2)

# ---------------------------------------------------------------- simulate
root = Root(full_system=False, system=system)
m5.instantiate()
if host:
    process.map(SHARED_VA, shared_pa, shared_size, True)
    process.map(PIO_VA, PIO_ADDR, 4096, False)
ev = m5.simulate()
cause = ev.getCause()
print(f"[systolic_se] exit @ tick {m5.curTick()}: {cause}")
if host:
    ok = ev.getCode() == 0
    print(f"[systolic_se] host program exit code {ev.getCode()}")
else:
    ok = cause == "systolic accelerator finished"
if ok:
    print("[systolic_se] systolic accelerator finished")
else:
    sys.exit(1)
