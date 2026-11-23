#-----------------------------------------------------------------------------#
# EEL5764 Computer Architecture                                               #
#-----------------------------------------------------------------------------#
# Project     : Parameterized SRAM Scratchpad Array                           #
# Filename    : SystolicAccel.py                                              #
# Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke   #
# Group       : #4                                                            #
# Description : SystolicAccel SimObject declaration                           #
#-----------------------------------------------------------------------------#
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
    stall_model = Param.String(
        "decoupled",
        "'decoupled': per-bank queues, read-ahead and a write buffer sit "
        "between banks and array. 'lockstep': a schedule step issues all "
        "its words at once and the next step waits until every bank has "
        "served its share, so a step takes as many cycles as the busiest "
        "bank has words",
    )
    bank_ports = Param.String(
        "1rw",
        "'1rw': reads and writes share one 8-byte port per bank. '1r1w': "
        "simple dual port, one read port and one write port per bank "
        "working in the same cycle (spm_read_alias must be set)",
    )
    spm_read_alias = Param.Addr(
        0,
        "1r1w only: address offset of the read-port SimpleMemory objects "
        "from the write-port ones (they hold the same data)",
    )
    weight_store = Param.String(
        "spm",
        "Where weight tiles cross the banks. 'spm': DMA write and array "
        "read (2 crossings, D_fill). 'fill_bypass': the DMA deposits the "
        "tile over a dedicated path, the array reads it through the banks "
        "(1 crossing, eq. 1). 'separate': a dedicated staging buffer "
        "outside the banks (0 crossings)",
    )
    bank_hash = Param.String(
        "modulo",
        "Word to bank map. 'modulo': word w is in bank w mod BANKS. 'xor': "
        "with row r = w div BANKS, bank (w + (r ^ (r >> s))) mod BANKS, "
        "s = ceil(log2(BANKS)). Needs bank_mapping='modulo'",
    )
    m_tile = Param.Unsigned(
        0,
        "Host-driven: process A in tiles of this many rows (0 = off). Each "
        "tile loads A, runs every weight tile and stores C. Rows * K and "
        "rows * N_out * 4 must be multiples of the 64-byte line",
    )
    n_tile = Param.Unsigned(
        0,
        "Host-driven: process the output columns in slices of this many "
        "columns (0 = off, a multiple of N and of 16). Each slice runs its "
        "weight tiles over the A in the scratchpad and stores its C",
    )
    psum_rmw = Param.Bool(
        False,
        "Accumulating across K tiles reads the old partial sum through the "
        "bank's read port before writing the new one (no read on a row's "
        "first K tile)",
    )
    psum_store = Param.String(
        "spm",
        "'spm': partial sums are written through the bank ports. 'external': "
        "they accumulate outside the counted banks and use no bank port "
        "during compute",
    )
    bank_trace_file = Param.String(
        "", "If set, write one CSV row per compute-phase cycle here"
    )
    w_prepacked_file = Param.String(
        "", "Host-driven: packed weight tiles to load into DRAM at time 0"
    )
    w_prepacked_addr = Param.Addr(
        0, "DRAM address for w_prepacked_file"
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
