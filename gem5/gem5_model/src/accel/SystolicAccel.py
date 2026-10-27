# EEL5764 Group 4 -- SystolicAccel SimObject declaration.
#
# A weight-stationary INT8 N x N systolic array that streams one GEMM
#   C[M x Nout] = A[M x K] * W[K x Nout]
# through a banked scratchpad (spm_ports -> spm_xbar -> BANKS SimpleMemory)
# and refills weight tiles from DRAM (dram_port -> membus -> DDR4).
#
# Each spm_ports entry is a request queue that sends at most one 8-byte
# word per accelerator cycle.  With arbitration="bank" (default) port i
# feeds bank i, so every bank serves its oldest request each cycle.  With
# arbitration="lane" the ports are the array's edge lanes, which contend
# for banks inside spm_xbar:
#   [0, N/8)                      activation read lanes   (left edge, N B)
#   [N/8, N/4)                    weight read lanes       (top edge,  N B)
#   [N/4, 3N/4)                   partial-sum write lanes (bottom,   4N B)
#   [3N/4, 3N/4 + dma_lanes)      DMA write lanes (DRAM -> SPM weight slot)

from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *


class SystolicAccel(ClockedObject):
    type = "SystolicAccel"
    cxx_header = "accel/systolic_accel.hh"
    cxx_class = "gem5::SystolicAccel"

    system = Param.System(Parent.any, "System (for the requestor id)")

    # ---- hardware (compile/synthesis-time in RTL) ----
    array_dim = Param.Unsigned(8, "N: array is N x N PEs (multiple of 8)")
    double_buffer = Param.Bool(
        True, "Shadow weight registers: next tile loads while one streams"
    )

    # ---- workload (runtime): C[M x Nout] = A[M x K] * W[K x Nout] ----
    gemm_m = Param.Unsigned(0, "M: activation rows per weight tile")
    gemm_k = Param.Unsigned(0, "K: reduction dimension")
    gemm_n = Param.Unsigned(0, "Nout: output columns")
    a_file = Param.String("", "INT8 A, row-major M x K ('' = random)")
    w_file = Param.String("", "INT8 W, row-major K x Nout ('' = random)")
    c_golden_file = Param.String("", "INT32 golden C, row-major M x Nout")
    c_out_file = Param.String("", "If set, dump the computed INT32 C here")
    seed = Param.Unsigned(1, "Seed for random A/W when no files given")
    verify = Param.Bool(True, "Check C against a reference GEMM at the end")

    # ---- memory map ----
    spm_base = Param.Addr(0x80000000, "Scratchpad base address")
    spm_size = Param.MemorySize("16MiB", "Scratchpad size (all banks)")
    w_dram_base = Param.Addr(0x0, "DRAM address of the packed weight tiles")

    # ---- microarchitecture knobs ----
    lookahead = Param.Unsigned(
        8, "Read-ahead depth in array cycles (hides SPM + xbar latency)"
    )
    psum_queue_depth = Param.Unsigned(
        8,
        "Partial-sum write buffer entries per psum lane (N/2 lanes, 8 B "
        "each) before the array stalls",
    )
    weight_slots = Param.Unsigned(
        2, "Weight tile buffers in SPM (DMA prefetch depth)"
    )
    dma_lanes = Param.Unsigned(8, "8-byte DMA write lanes into the SPM")
    dma_max_outstanding = Param.Unsigned(
        64, "Max in-flight 64-byte DRAM reads for weight refill"
    )
    ideal_spm = Param.Bool(
        False, "Infinite-bandwidth, zero-latency scratchpad (baseline)"
    )
    ideal_dram = Param.Bool(
        False, "Weights appear in the SPM instantly (no DRAM traffic)"
    )
    max_cycles = Param.UInt64(0, "Abort after this many cycles (0 = off)")

    spm_banks = Param.Unsigned(8, "Scratchpad banks (8-byte words)")
    bank_mapping = Param.String(
        "interleave",
        "'interleave': power-of-two BANKS, the crossbar's address "
        "interleaving selects the bank. 'modulo': any BANKS; each bank "
        "is a contiguous range and the accelerator maps word w to bank "
        "w mod BANKS",
    )
    bank_policy = Param.String(
        "fifo",
        "Order within a bank's request queue. 'fifo': oldest first. "
        "'reads_first': reads go ahead of queued writes (the array stalls "
        "on late reads; writes wait in the write buffer)",
    )
    arbitration = Param.String(
        "bank",
        "'bank': one request queue per bank, each bank serves its oldest "
        "request every cycle (supply = 8*BANKS bytes/cycle, eq. 2). "
        "'lane': one queue per edge lane, 1 word/lane/cycle, lanes contend "
        "for banks in the crossbar (lockstep-RTL-like)",
    )

    spm_ports = VectorRequestPort(
        "Request ports to the scratchpad crossbar: spm_banks ports in "
        "'bank' arbitration, N/8+N/8+N/2+dma_lanes ports in 'lane'"
    )
    dram_port = RequestPort("DMA port to the memory bus (weight refill)")

    # ---- host-driven mode (proposal phase 3) ----
    host_driven = Param.Bool(
        False,
        "Wait for a RISC-V host to program the registers and set START; "
        "DMA A in and C out; do not end the simulation",
    )
    pio_addr = Param.Addr(0xA0000000, "Control register base address")
    pio_size = Param.Addr(0x1000, "Control register window size")
    pio_latency = Param.Cycles(2, "Register access latency")
    pio = ResponsePort("Control registers (connect to the memory bus)")
