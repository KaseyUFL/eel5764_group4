//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : systolic_accel.cc                                           //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //
// Group       : #4                                                          //
// Description : SystolicAccel implementation                                //
//---------------------------------------------------------------------------//
/*
 * Static schedule (array cycles, before any stall), tile t = 0..nTiles-1,
 * T = max(M, N) with double buffering or M + N without it:
 *
 *   weight row r of tile t is read from the SPM at   W_t + r,  W_t = tT + 1
 *   activation row m of tile t is read at            S_t + m,
 *       S_t = (t + 1) T       (double buffered; matches tb_systolic.sv)
 *       S_t = t T + N + 1     (single buffered: load, then stream)
 *   psum lane j (columns 2j, 2j+1) of that row is written at
 *                                                    S_t + m + N + 2j + 1
 *
 * Every accelerator clock the model tries to execute the next array cycle.
 * It executes only if that cycle's activation/weight words have returned
 * from the SPM and every psum lane that must write has queue space;
 * otherwise the whole array stalls for this clock (a global stall).
 * Reads are issued up to `lookahead` array cycles early, which models the
 * address-generation pipeline in front of the SRAM.
 *
 * With stall_model = "lockstep" the same schedule runs without any of that
 * elasticity: a step's activation, weight and partial-sum words all go to
 * the banks at once, the next step waits until every bank has accepted its
 * share, and the only thing between address generation and the array is the
 * fixed bank read pipeline.  A step takes as many cycles as the busiest bank
 * has words.
 *
 * Tiles are ordered nb-outer / kb-inner so each output block accumulates in
 * the SPM accumulator region across the K dimension.  Weight tiles are
 * DMA'd from DRAM (tile-major packed) into `weight_slots` SPM buffers.
 *
 * Host-driven layers add two DMA phases around the compute phase:
 *   LoadA   A (row-major in DRAM) -> SPM, blocked by K-block, zero padded
 *   StoreC  C (SPM accumulators) -> DRAM, row-major INT32
 * The weight DMA may already prefetch tiles during LoadA.
 */

#include "accel/systolic_accel.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/SystolicAccel.hh"
#include "debug/SystolicAccelDma.hh"
#include "sim/sim_exit.hh"
#include "sim/system.hh"

namespace gem5
{

namespace
{

uint64_t
ceilDiv(uint64_t a, uint64_t b)
{
    return b ? (a + b - 1) / b : 0;
}

Addr
align64(Addr a)
{
    return (a + 63) & ~Addr(63);
}

void
readFile(const std::string &path, void *dst, size_t bytes)
{
    std::ifstream f(path, std::ios::binary);
    fatal_if(!f, "SystolicAccel: cannot open '%s'", path);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    fatal_if(sz != bytes, "SystolicAccel: '%s' is %d bytes, expected %d",
             path, sz, bytes);
    f.seekg(0);
    f.read(static_cast<char *>(dst), bytes);
}

} // anonymous namespace

// ---------------------------------------------------------------------
// Ports

SystolicAccel::LanePort::LanePort(const std::string &name,
                                  SystolicAccel &owner, unsigned idx)
    : RequestPort(name, idx), owner(owner), idx(idx)
{
}

bool
SystolicAccel::LanePort::trySend()
{
    // One request per queue per accelerator clock.
    if (waitRetry || queue.empty() || owner.clockEdge() == lastSend)
        return false;

    PacketPtr pkt = queue.front();
    auto *st = pkt->findNextSenderState<AccessState>();
    if (st)
        st->sent = curTick();
    // The memory turns the packet into a response inside sendTimingReq.
    const bool is_read = pkt->isRead();
    if (!sendTimingReq(pkt)) {
        waitRetry = true;
        owner.stats.laneRetries++;
        return false;
    }
    if (st && st->bank >= 0)
        owner.stats.bankAccesses[st->bank]++;
    if (st && owner.lockstep) {
        if (st->stream == Stream::Act || st->stream == Stream::Weight ||
            st->stream == Stream::PsumRead)
            owner.stepReadsUnsent--;
        else if (st->stream == Stream::Psum)
            owner.stepWritesUnsent--;
    }
    if (st && st->bank >= 0 && owner.traceOut &&
        owner.phase == Phase::Compute) {
        TraceOp &op = owner.traceOps[st->bank];
        if (is_read) {
            op.read = true;
            op.readRow = st->row;
        } else {
            op.write = true;
            op.writeRow = st->row;
        }
    }
    queue.pop_front();
    if (frontReads)
        frontReads--;
    lastSend = owner.clockEdge();
    return true;
}

void
SystolicAccel::LanePort::push(PacketPtr pkt, bool is_read)
{
    // "reads_first" bank policy: a read goes ahead of every queued write
    // (and behind earlier reads).  The array stalls on a late read, while
    // a write only waits in the write buffer, so a burst of writes to one
    // bank should not delay the reads behind it.  Reordering is safe: no
    // read issued here depends on a write that is still queued.
    if (owner.readsFirst && is_read) {
        queue.insert(queue.begin() + frontReads, pkt);
        frontReads++;
    } else {
        queue.push_back(pkt);
    }
}

bool
SystolicAccel::LanePort::recvTimingResp(PacketPtr pkt)
{
    return owner.recvSpmResp(pkt);
}

void
SystolicAccel::LanePort::recvReqRetry()
{
    // Synchronous hardware only issues on clock edges: just re-arm the
    // port; the next tick() (this edge or the next one) resends.
    waitRetry = false;
}

void
SystolicAccel::DramPort::send(PacketPtr pkt)
{
    if (blocked.empty() && sendTimingReq(pkt))
        return;
    blocked.push_back(pkt);
}

bool
SystolicAccel::DramPort::recvTimingResp(PacketPtr pkt)
{
    return owner.recvDramResp(pkt);
}

void
SystolicAccel::DramPort::recvReqRetry()
{
    while (!blocked.empty() && sendTimingReq(blocked.front()))
        blocked.pop_front();
}

Tick
SystolicAccel::PioPort::recvAtomic(PacketPtr pkt)
{
    return owner.pioAccess(pkt);
}

AddrRangeList
SystolicAccel::PioPort::getAddrRanges() const
{
    AddrRangeList l;
    if (owner.hostDriven)
        l.push_back(RangeSize(owner.pioAddr, owner.pioSize));
    return l;
}

// ---------------------------------------------------------------------
// Construction

SystolicAccel::SystolicAccel(const Params &p)
    : ClockedObject(p),
      N(p.array_dim),
      doubleBuffer(p.double_buffer),
      actLanes(p.array_dim / 8), wgtLanes(p.array_dim / 8),
      psumLanes(p.array_dim / 2), dmaLanes(p.dma_lanes),
      lookahead(p.lookahead), psumQueueDepth(p.psum_queue_depth),
      weightSlots(p.weight_slots),
      dmaMaxOutstanding(p.dma_max_outstanding),
      spmBanks(p.spm_banks),
      perBank(p.arbitration == "bank"),
      moduloBanks(p.bank_mapping == "modulo"),
      readsFirst(p.bank_policy == "reads_first"),
      lockstep(p.stall_model == "lockstep"),
      sdpPorts(p.bank_ports == "1r1w"),
      xorHash(p.bank_hash == "xor"),
      weightStore(p.weight_store == "separate" ? WeightStore::Separate :
                  p.weight_store == "fill_bypass" ? WeightStore::FillBypass :
                                                    WeightStore::Spm),
      hashShift(p.spm_banks > 1 ? ceilLog2(p.spm_banks) : 1),
      mTile(p.m_tile), nTile(p.n_tile), psumRmw(p.psum_rmw),
      psumExternal(p.psum_store == "external"),
      idealSpm(p.ideal_spm), idealDram(p.ideal_dram), verify(p.verify),
      hostDriven(p.host_driven),
      maxCycles(p.max_cycles),
      spmBase(p.spm_base), spmSize(p.spm_size),
      bankBytes(p.spm_size / std::max(1u, p.spm_banks)),
      readAlias(p.bank_ports == "1r1w" ? p.spm_read_alias : 0),
      pioAddr(p.pio_addr), pioSize(p.pio_size),
      pioLatency(p.pio_latency),
      requestorId(p.system->getRequestorId(this)),
      dramPort(name() + ".dram_port", *this),
      pioPort(name() + ".pio", *this),
      tickEvent([this]{ tick(); }, name() + ".tick", false,
                Event::CPU_Tick_Pri),
      startEvent([this]{ startLayer(); }, name() + ".start"),
      stats(*this)
{
    fatal_if(N < 8 || N % 8, "array_dim (%d) must be a multiple of 8", N);
    fatal_if(!lookahead, "lookahead must be >= 1");
    fatal_if(!weightSlots, "weight_slots must be >= 1");
    fatal_if(!dmaLanes, "dma_lanes must be >= 1");
    fatal_if(!psumQueueDepth, "psum_queue_depth must be >= 1");
    fatal_if(!dmaMaxOutstanding, "dma_max_outstanding must be >= 1");
    fatal_if(p.arbitration != "bank" && p.arbitration != "lane",
             "arbitration must be 'bank' or 'lane', not '%s'",
             p.arbitration);
    fatal_if(p.bank_mapping != "interleave" && p.bank_mapping != "modulo",
             "bank_mapping must be 'interleave' or 'modulo'");
    fatal_if(p.bank_policy != "fifo" && p.bank_policy != "reads_first",
             "bank_policy must be 'fifo' or 'reads_first'");
    fatal_if(p.stall_model != "decoupled" && p.stall_model != "lockstep",
             "stall_model must be 'decoupled' or 'lockstep'");
    fatal_if(lockstep && idealSpm,
             "stall_model='lockstep' needs a real scratchpad, not ideal_spm");
    fatal_if(p.bank_ports != "1rw" && p.bank_ports != "1r1w",
             "bank_ports must be '1rw' or '1r1w'");
    fatal_if(sdpPorts && !p.spm_read_alias,
             "bank_ports='1r1w' needs spm_read_alias");
    fatal_if(p.weight_store != "spm" && p.weight_store != "fill_bypass" &&
             p.weight_store != "separate",
             "weight_store must be 'spm', 'fill_bypass' or 'separate'");
    fatal_if(p.bank_hash != "modulo" && p.bank_hash != "xor",
             "bank_hash must be 'modulo' or 'xor'");
    fatal_if(xorHash && !moduloBanks,
             "bank_hash='xor' needs bank_mapping='modulo'");
    fatal_if(mTile && !hostDriven, "m_tile needs a host-driven layer");
    fatal_if(nTile && !hostDriven, "n_tile needs a host-driven layer");
    fatal_if(p.psum_store != "spm" && p.psum_store != "external",
             "psum_store must be 'spm' or 'external'");
    fatal_if(psumRmw && psumExternal,
             "psum_rmw reads through the bank ports, psum_store='external' "
             "does not use them");
    fatal_if(!spmBanks, "spm_banks must be >= 1");
    fatal_if(!moduloBanks && (spmBanks & (spmBanks - 1)),
             "spm_banks (%d) is not a power of two: use "
             "bank_mapping='modulo'", spmBanks);
    fatal_if(moduloBanks && (spmSize % (8 * spmBanks)),
             "spm_size must be a multiple of 8 * spm_banks");

    const unsigned want = perBank ? spmBanks * (sdpPorts ? 2 : 1)
                                  : actLanes + wgtLanes + psumLanes +
                                    dmaLanes;
    fatal_if(p.port_spm_ports_connection_count != want,
             "SystolicAccel (N=%d, %s arbitration) needs exactly %d "
             "spm_ports (bank: one per bank, two with bank_ports=1r1w; "
             "lane: N/8 act + N/8 weight + N/2 psum + dma_lanes), got %d",
             N, p.arbitration, want,
             p.port_spm_ports_connection_count);
    for (unsigned i = 0; i < want; ++i) {
        lanes.emplace_back(new LanePort(
            csprintf("%s.spm_ports%d", name(), i), *this, i));
    }

    slots.resize(lookahead);
    for (auto &s : slots) {
        s.act.resize(N);
        s.wgt.resize(N);
    }
    psumRing.resize(2 * N + 2);
    for (auto &r : psumRing)
        r.vals.resize(N);
    dmaRemaining.assign(weightSlots, 0);
    dmaLinesLeft.assign(weightSlots, 0);
    if (weightStore == WeightStore::Separate)
        wStage.resize(uint64_t(weightSlots) * N * N);
    if (!p.bank_trace_file.empty()) {
        traceOut = std::make_unique<std::ofstream>(p.bank_trace_file);
        fatal_if(!*traceOut, "SystolicAccel: cannot write '%s'",
                 p.bank_trace_file);
        traceOps.resize(spmBanks);
        writeTraceHeader();
    }

    regs[RegArrayDim / 8] = N;
    if (!hostDriven) {
        fatal_if(!p.gemm_m || !p.gemm_k || !p.gemm_n,
                 "standalone mode needs gemm_m/k/n > 0");
        configure(p.gemm_m, p.gemm_k, p.gemm_n);
        wDramBase = p.w_dram_base;
    }
}

void
SystolicAccel::configure(uint64_t m, uint64_t k, uint64_t n)
{
    M = m;
    K = k;
    Nout = n;
    nKb = ceilDiv(K, N);
    nNb = ceilDiv(Nout, N);
    nTiles = nKb * nNb;
    T = doubleBuffer ? std::max<uint64_t>(M, N) : M + N;

    // SPM layout: [A blocked by kb][weight slots][C accumulators]
    aBase = spmBase;
    wBase = align64(aBase + nKb * M * N);
    const uint64_t slot_bytes = weightStore == WeightStore::Separate
        ? 0 : uint64_t(weightSlots) * N * N;
    cBase = align64(wBase + slot_bytes);
    spmEnd = cBase + nNb * M * N * 4;
    fatal_if(spmEnd > spmBase + spmSize,
             "Layer %dx%dx%d needs %d bytes of scratchpad but spm_size is "
             "%d", M, K, Nout, spmEnd - spmBase, spmSize);

    // Last activation row enters at S_last + M - 1; the last psum lane
    // (j = N/2 - 1) writes it 2N - 1 cycles later.
    const uint64_t s_last = doubleBuffer ? nTiles * T
                                         : (nTiles - 1) * T + N + 1;
    schedEnd = s_last + M - 1 + 2 * N;

    const uint64_t s_first = doubleBuffer ? T : N + 1;
    ssStart = s_first + 2 * N - 1;
    ssEnd = s_last + M - 1;

    acc.assign(nNb * M * N, 0);
    dmaDone.assign(nTiles, false);
}

void
SystolicAccel::resetRunState()
{
    sched = genCycle = 0;
    for (auto &s : slots)
        s.cycle = UINT64_MAX;
    for (auto &r : psumRing)
        r.cycle = UINT64_MAX;
    outstandingWrites = 0;
    cyclesElapsed = layerCycles = phaseCycles = 0;
    dmaIssueTile = 0;
    dmaIssueLine = 0;
    weightTilesConsumed = 0;
    dmaLaneRR = 0;
    wAct.assign(N * N, 0);
    wNext.assign(N * N, 0);
    aLines = aLinesIssued = aLinesDone = aPending = 0;
    cChunks = cChunksIssued = cDramPending = 0;
    cInFlight.clear();
    stepReadsUnsent = stepWritesUnsent = 0;
}

Port &
SystolicAccel::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "spm_ports" && idx >= 0 && idx < (PortID)lanes.size())
        return *lanes[idx];
    if (if_name == "dram_port")
        return dramPort;
    if (if_name == "pio")
        return pioPort;
    return ClockedObject::getPort(if_name, idx);
}

void
SystolicAccel::init()
{
    ClockedObject::init();
    for (auto &l : lanes)
        fatal_if(!l->isConnected(), "%s is not connected", l->name());
    fatal_if(!dramPort.isConnected(), "dram_port is not connected");
    if (hostDriven) {
        fatal_if(!pioPort.isConnected(),
                 "host_driven needs the pio port connected");
        pioPort.sendRangeChange();
    }
}

void
SystolicAccel::startup()
{
    if (hostDriven) {
        inform("SystolicAccel: N=%d, host-driven, registers at %#x",
               N, pioAddr);
        loadPrepackedWeights();
        return;
    }
    loadInputs();
    resetRunState();
    phase = Phase::Compute;
    stats.layers++;
    inform("SystolicAccel: N=%d %s-buffered, %s arbitration over %d "
           "SPM ports, GEMM %dx%dx%d, %d tiles (%d kb x %d nb), T=%d, "
           "ideal schedule %d cycles, SPM A@%#x W@%#x C@%#x end %#x",
           N, doubleBuffer ? "double" : "single",
           perBank ? "per-bank" : "per-lane", lanes.size(), M, K, Nout,
           nTiles, nKb, nNb, T, schedEnd, aBase, wBase, cBase, spmEnd);
    schedule(tickEvent, clockEdge());
}

// ---------------------------------------------------------------------
// Control registers

Tick
SystolicAccel::pioAccess(PacketPtr pkt)
{
    const Addr off = pkt->getAddr() - pioAddr;
    const unsigned size = pkt->getSize();
    panic_if(size > 8 || (off & 7) + size > 8,
             "SystolicAccel: unaligned register access at %#x", off);
    const Addr reg = off & ~Addr(7);
    const unsigned shift = (off & 7) * 8;

    if (pkt->isRead()) {
        uint64_t v = 0;
        if (reg == RegStatus)
            v = status;
        else if (reg < RegEnd)
            v = regs[reg / 8];
        pkt->setUintX(v >> shift, ByteOrder::little);
        stats.pioReads++;
    } else if (pkt->isWrite()) {
        const uint64_t v = pkt->getUintX(ByteOrder::little);
        stats.pioWrites++;
        if (reg == RegCtrl) {
            if (v & 1) {
                if (phase != Phase::Idle && phase != Phase::Done) {
                    warn("SystolicAccel: START while busy ignored");
                } else {
                    status = 1;     // busy
                    schedule(startEvent, clockEdge(Cycles(1)));
                }
            }
        } else if (reg >= RegM && reg <= RegCAddr) {
            const uint64_t mask = size == 8 ? ~0ULL
                                            : ((1ULL << (8 * size)) - 1);
            regs[reg / 8] = (regs[reg / 8] & ~(mask << shift)) |
                            ((v & mask) << shift);
        }
    }
    pkt->makeAtomicResponse();
    return cyclesToTicks(pioLatency);
}

void
SystolicAccel::startLayer()
{
    const uint64_t m = regs[RegM / 8], k = regs[RegK / 8],
                   n = regs[RegNout / 8];
    aDramBase = regs[RegAAddr / 8];
    wDramBase = regs[RegWAddr / 8];
    cDramBase = regs[RegCAddr / 8];
    if (!m || !k || !n || k % 8 || n % 2 || (aDramBase | wDramBase |
                                              cDramBase) & 63) {
        warn("SystolicAccel: bad layer registers (M=%d K=%d N_out=%d, "
             "K must be a multiple of 8, N_out even, addresses 64-byte "
             "aligned)", m, k, n);
        status = 4 | 2;     // error + done
        return;
    }
    // A row-major A and C must continue on a cache line at the next M tile.
    if (mTile && m > mTile && ((mTile * k) % lineSize ||
                               (mTile * n * 4) % lineSize)) {
        warn("SystolicAccel: m_tile=%d does not keep A and C line-aligned "
             "for K=%d N_out=%d", mTile, k, n);
        status = 4 | 2;     // error + done
        return;
    }
    if (nTile && (nTile % N || nTile % 16 || n % 16 || (n % nTile) % 16)) {
        warn("SystolicAccel: n_tile=%d must be a multiple of N and of 16, and "
             "every slice of N_out=%d a multiple of 16 columns", nTile, n);
        status = 4 | 2;     // error + done
        return;
    }
    layerM = m;
    layerNout = n;
    nColsDone = 0;
    wDramLayerBase = wDramBase;
    mRowsDone = 0;
    computeCyclesTotal = 0;
    verifyBaseline = stats.verifyMismatches.value();
    layerCycles = 0;
    stats.layers++;
    beginMTile(mTile ? std::min<uint64_t>(mTile, m) : m);
    inform("SystolicAccel: layer %d start: GEMM %dx%dx%d, %d tiles, "
           "T=%d, A@%#x W@%#x C@%#x", (uint64_t)stats.layers.value(),
           M, K, Nout, nTiles, T, aDramBase, wDramBase, cDramBase);
    if (!tickEvent.scheduled())
        schedule(tickEvent, clockEdge());
}

void
SystolicAccel::beginMTile(uint64_t rows, bool load_a)
{
    const uint64_t layer_cycles = layerCycles;
    configure(rows, regs[RegK / 8],
              nTile ? std::min(nTile, layerNout - nColsDone) : layerNout);
    resetRunState();
    layerCycles = layer_cycles;
    if (load_a)
        stats.mTiles++;
    stats.nSlices++;

    // The ideal-memory baselines read operands without timing, so take a
    // functional copy of what the host put in DRAM.
    if (idealSpm || idealDram) {
        aRaw.resize(M * K);
        wPacked.resize(nTiles * N * N);
        const size_t chunk = lineSize;  // see doVerify()
        for (size_t o = 0; o < aRaw.size(); o += chunk)
            functionalRead(dramPort, aDramBase + o, &aRaw[o],
                           std::min(chunk, aRaw.size() - o));
        for (size_t o = 0; o < wPacked.size(); o += chunk)
            functionalRead(dramPort, wDramBase + o, &wPacked[o],
                           std::min(chunk, wPacked.size() - o));
    }

    if (!load_a) {
        // The next slice of output columns over the A already in the SPM.
        phase = Phase::Compute;
        phaseCycles = 0;
        return;
    }
    aLines = ceilDiv(M * K, lineSize);
    // Zero the padded columns of the last K-block (K not a multiple of
    // N): the SPM may still hold the previous layer's data there.
    if (K % N) {
        static const uint8_t zeros[8] = {};
        const uint64_t kb = nKb - 1;
        for (uint64_t mm = 0; mm < M; ++mm) {
            for (unsigned col = K % N; col < N; col += 8) {
                const Addr a = aBase + (kb * M + mm) * N + col;
                auto *st = new AccessState(Stream::ALoad, 0, 0, curTick());
                dmaQ(dmaLaneRR++, a).push(spmWrite(a, zeros, st), false);
                aPending++;
            }
        }
    }
    phase = Phase::LoadA;
}

void
SystolicAccel::loadPrepackedWeights()
{
    const auto &p = params();
    if (p.w_prepacked_file.empty())
        return;
    std::ifstream f(p.w_prepacked_file, std::ios::binary);
    fatal_if(!f, "SystolicAccel: cannot open '%s'", p.w_prepacked_file);
    std::vector<char> line(lineSize);
    for (Addr a = p.w_prepacked_addr;; a += lineSize) {
        f.read(line.data(), lineSize);
        if (f.gcount() == 0)
            break;
        functionalWrite(dramPort, a, line.data(), f.gcount());
    }
}

// ---------------------------------------------------------------------
// Standalone input loading (functional, at time 0)

void
SystolicAccel::functionalWrite(RequestPort &port, Addr addr,
                               const void *data, unsigned size)
{
    auto req = std::make_shared<Request>(addr, size, 0, requestorId);
    Packet pkt(req, MemCmd::WriteReq);
    pkt.dataStatic(static_cast<const uint8_t *>(data));
    port.sendFunctional(&pkt);
}

void
SystolicAccel::functionalRead(RequestPort &port, Addr addr, void *data,
                              unsigned size)
{
    auto req = std::make_shared<Request>(addr, size, 0, requestorId);
    Packet pkt(req, MemCmd::ReadReq);
    pkt.dataStatic(static_cast<uint8_t *>(data));
    port.sendFunctional(&pkt);
}

void
SystolicAccel::loadInputs()
{
    const auto &p = params();
    aRaw.resize(M * K);
    wRaw.resize(K * Nout);
    std::mt19937 rng(p.seed);
    std::uniform_int_distribution<int> dist(-128, 127);
    if (!p.a_file.empty())
        readFile(p.a_file, aRaw.data(), aRaw.size());
    else
        for (auto &v : aRaw) v = dist(rng);
    if (!p.w_file.empty())
        readFile(p.w_file, wRaw.data(), wRaw.size());
    else
        for (auto &v : wRaw) v = dist(rng);

    // A blocked by K-block, zero padded: row (kb, m) holds
    // A[m][kb*N .. kb*N + N - 1].  Written in 8-byte bank words.
    std::vector<uint8_t> row(N);
    for (uint64_t kb = 0; kb < nKb; ++kb) {
        for (uint64_t m = 0; m < M; ++m) {
            for (unsigned i = 0; i < N; ++i) {
                uint64_t k = kb * N + i;
                row[i] = k < K ? aRaw[m * K + k] : 0;
            }
            Addr base = aBase + (kb * M + m) * N;
            for (unsigned w = 0; w < N / 8; ++w)
                spmFunctionalWrite(base + 8 * w, &row[8 * w]);
        }
    }

    // W packed tile-major (tile t = nb * nKb + kb), each tile N x N with
    // row r = k offset (PE row), column j = output column.
    wPacked.assign(nTiles * N * N, 0);
    for (uint64_t t = 0; t < nTiles; ++t) {
        uint64_t nb = t / nKb, kb = t % nKb;
        for (unsigned r = 0; r < N; ++r) {
            uint64_t k = kb * N + r;
            if (k >= K)
                continue;
            for (unsigned j = 0; j < N; ++j) {
                uint64_t n = nb * N + j;
                if (n < Nout)
                    wPacked[(t * N + r) * N + j] = wRaw[k * Nout + n];
            }
        }
    }
    if (!idealDram) {
        const size_t chunk = 4096;
        for (size_t off = 0; off < wPacked.size(); off += chunk) {
            functionalWrite(dramPort, wDramBase + off, &wPacked[off],
                            std::min(chunk, wPacked.size() - off));
        }
    }
}

// ---------------------------------------------------------------------
// Schedule helpers

SystolicAccel::CycleInfo
SystolicAccel::decode(uint64_t c) const
{
    CycleInfo ci;
    if (c >= 1) {
        uint64_t t = (c - 1) / T, off = (c - 1) % T;
        if (t < nTiles) {
            if (off < N) {
                ci.wTile = t;
                ci.r = off;
            } else if (!doubleBuffer && off - N < M) {
                ci.actTile = t;
                ci.m = off - N;
            }
        }
    }
    if (doubleBuffer && c >= T) {
        uint64_t t = c / T - 1, m = c % T;
        if (t < nTiles && m < M) {
            ci.actTile = t;
            ci.m = m;
        }
    }
    return ci;
}

Addr
SystolicAccel::aRowAddr(uint64_t tile, uint64_t m) const
{
    return aBase + ((tile % nKb) * M + m) * N;
}

Addr
SystolicAccel::wRowAddr(uint64_t tile, uint64_t r) const
{
    return wBase + (tile % weightSlots) * N * N + r * N;
}

Addr
SystolicAccel::cRowAddr(uint64_t tile, uint64_t m) const
{
    return cBase + ((tile / nKb) * M + m) * N * 4;
}

PacketPtr
SystolicAccel::makeRead(Addr addr, unsigned size, AccessState *st)
{
    auto req = std::make_shared<Request>(addr, size, 0, requestorId);
    PacketPtr pkt = new Packet(req, MemCmd::ReadReq);
    pkt->allocate();
    pkt->pushSenderState(st);
    return pkt;
}

PacketPtr
SystolicAccel::makeWrite(Addr addr, unsigned size, const void *data,
                         AccessState *st)
{
    auto req = std::make_shared<Request>(addr, size, 0, requestorId);
    PacketPtr pkt = new Packet(req, MemCmd::WriteReq);
    pkt->allocate();
    pkt->setData(static_cast<const uint8_t *>(data));
    pkt->pushSenderState(st);
    return pkt;
}

// ---------------------------------------------------------------------
// Main loop

void
SystolicAccel::tick()
{
    layerCycles++;
    phaseCycles++;

    switch (phase) {
      case Phase::LoadA:
        aLoadAdvance();
        dmaAdvance();       // weight prefetch overlaps the A load
        for (auto &l : lanes)
            l->trySend();
        stats.loadCycles++;
        if (aLinesDone == aLines && aPending == 0) {
            DPRINTF(SystolicAccel, "A loaded in %d cycles\n", phaseCycles);
            phase = Phase::Compute;
            phaseCycles = 0;
        }
        break;

      case Phase::Compute: {
        if (sched < schedEnd) {
            dmaAdvance();
            generateReads();
        }
        for (auto &l : lanes)
            l->trySend();
        const uint64_t step = sched;
        const bool steady = inSteadyState();
        bool stalled = false;
        if (steady)
            stats.steadyStateCycles++;
        else
            stats.fillDrainCycles++;
        if (sched < schedEnd) {
            bool dma_blocked = false;
            if (tryExecute(dma_blocked)) {
                stats.activeCycles++;
                lastStall = StallCause::None;
                if (steady && decode(step).actTile >= 0)
                    stats.steadyStateRows++;
            } else {
                stalled = true;
                stats.stallCycles++;
                if (dma_blocked) {
                    stats.dmaStallCycles++;
                    lastStall = StallCause::Dma;
                } else {
                    stats.spmStallCycles++;
                }
            }
        } else {
            stats.drainCycles++;
            lastStall = StallCause::Drain;
        }
        if (traceOut)
            writeTraceRow(step, stalled);
        cyclesElapsed++;

        if (sched >= schedEnd && drained()) {
            stats.totalCycles += cyclesElapsed;
            stats.scheduleCycles += schedEnd;
            stats.realMacs += double(M) * K * Nout;
            if (mRowsDone == 0)
                stats.modelDemand = double(5 * N * M + N * N) /
                                    double(std::max<uint64_t>(M, N));
            computeCyclesTotal += cyclesElapsed;
            regs[RegComputeCycles / 8] = computeCyclesTotal;
            if (hostDriven) {
                if (idealSpm) {
                    // No timing psum writes happened; put C in the SPM
                    // so the store phase has something to move.
                    for (uint64_t i = 0; i < acc.size(); i += 2)
                        spmFunctionalWrite(cBase + i * 4, &acc[i]);
                }
                cChunks = ceilDiv(M * Nout * 4, lineSize);
                phase = Phase::StoreC;
                phaseCycles = 0;
            } else {
                layerDone(true);
                return;
            }
        } else if (maxCycles && cyclesElapsed >= maxCycles) {
            warn("SystolicAccel: max_cycles (%d) reached at array cycle %d "
                 "of %d; stopping early (output not verified)", maxCycles,
                 sched, schedEnd);
            stats.totalCycles += cyclesElapsed;
            stats.scheduleCycles += schedEnd;
            layerDone(false);
            return;
        }
        break;
      }

      case Phase::StoreC:
        cStoreAdvance();
        for (auto &l : lanes)
            l->trySend();
        stats.storeCycles++;
        if (cChunksIssued == cChunks && cInFlight.empty() &&
            cDramPending == 0 && !dramPort.busy()) {
            if (mRowsDone + M >= layerM && nColsDone + Nout >= layerNout) {
                layerDone(true);
                return;
            }
            if (verify)
                doVerify();
            if (nColsDone + Nout < layerNout) {
                // Next output slice: same A, the tiles of the next columns.
                wDramBase += ceilDiv(Nout, N) * nKb * N * N;
                nColsDone += Nout;
                beginMTile(M, false);
            } else {
                // Next M tile: row-major A and C continue where this one
                // ended, and the weights start over.
                mRowsDone += M;
                aDramBase += M * K;
                cDramBase += M * layerNout * 4;
                wDramBase = wDramLayerBase;
                nColsDone = 0;
                beginMTile(std::min<uint64_t>(mTile, layerM - mRowsDone));
            }
        }
        break;

      default:
        return;
    }
    schedule(tickEvent, clockEdge(Cycles(1)));
}

void
SystolicAccel::aLoadAdvance()
{
    // One 64-byte DRAM read per clock, as long as reads can be tracked.
    if (aLinesIssued >= aLines || dramOutstanding >= dmaMaxOutstanding ||
        dramPort.busy())
        return;
    const uint64_t off = aLinesIssued * lineSize;
    const unsigned size = std::min<uint64_t>(lineSize, M * K - off);
    auto *st = new AccessState(Stream::ALoadRead, aLinesIssued, 0,
                               curTick());
    dramPort.send(makeRead(aDramBase + off, size, st));
    dramOutstanding++;
    aLinesIssued++;
}

void
SystolicAccel::cStoreAdvance()
{
    // Gather one 64-byte chunk of row-major C per clock from the SPM.
    if (cChunksIssued >= cChunks || cInFlight.size() >= dmaMaxOutstanding)
        return;
    const uint64_t c = cChunksIssued++;
    const uint64_t total = M * Nout * 4;
    CChunk &ch = cInFlight[c];
    ch.bytes = std::min<uint64_t>(lineSize, total - c * lineSize);
    ch.data.assign(ch.bytes, 0);
    ch.pending = ch.bytes / 8;
    for (unsigned p = 0; p < ch.pending; ++p) {
        const uint64_t e = c * (lineSize / 4) + 2 * p;  // element index
        const uint64_t m = e / Nout, n = e % Nout;
        const Addr a = cBase + (((n / N) * M + m) * N + n % N) * 4;
        auto *st = new AccessState(Stream::CStoreRead, c, p, curTick());
        dmaReadQ(dmaLaneRR++, a).push(spmRead(a, st), true);
    }
}

void
SystolicAccel::dmaAdvance()
{
    const unsigned lines_per_tile = N * N / lineSize;
    while (dmaIssueTile < nTiles &&
           dmaIssueTile < weightTilesConsumed + weightSlots) {
        const uint64_t t = dmaIssueTile;
        const unsigned slot = t % weightSlots;
        if (idealDram) {
            if (weightStore == WeightStore::Separate) {
                std::memcpy(&wStage[uint64_t(slot) * N * N],
                            &wPacked[t * N * N], N * N);
            } else if (!idealSpm) {
                Addr base = wRowAddr(t, 0);
                for (unsigned off = 0; off < N * N; off += 8)
                    spmFunctionalWrite(base + off,
                                       &wPacked[t * N * N + off]);
            }
            dmaDone[t] = true;
            dmaIssueTile++;
            continue;
        }
        if (dramOutstanding >= dmaMaxOutstanding || dramPort.busy())
            break;
        if (dmaIssueLine == 0) {
            dmaRemaining[slot] = N * N / 8;
            dmaLinesLeft[slot] = lines_per_tile;
            DPRINTF(SystolicAccelDma, "DMA tile %d -> slot %d start\n",
                    t, slot);
        }
        Addr addr = wDramBase + t * N * N + dmaIssueLine * lineSize;
        auto *st = new AccessState(Stream::DramRead, t, dmaIssueLine,
                                   curTick());
        dramPort.send(makeRead(addr, lineSize, st));
        dramOutstanding++;
        if (++dmaIssueLine == lines_per_tile) {
            dmaIssueLine = 0;
            dmaIssueTile++;
        }
        break;  // one DRAM request per accelerator clock
    }
}

void
SystolicAccel::generateReads()
{
    // Lockstep: the next step waits until the banks took all of this one.
    // A step's partial sums come from rows at least N + 1 steps older, which
    // have executed as long as issue stays within N + 1 steps of the array.
    if (lockstep && (stepReadsUnsent || stepWritesUnsent))
        return;
    const uint64_t depth = lockstep ? std::min(lookahead, N + 1) : lookahead;
    while (genCycle < schedEnd && genCycle < sched + depth) {
        const CycleInfo ci = decode(genCycle);
        // Weight reads for a tile can only start once its DMA landed.
        if (ci.wTile >= 0 && !dmaDone[ci.wTile])
            break;

        Slot &s = slots[genCycle % lookahead];
        s.cycle = genCycle;
        s.pending = 0;

        if (ci.actTile >= 0) {
            stats.actBytes += N;
            if (idealSpm) {
                uint64_t kb = ci.actTile % nKb;
                for (unsigned i = 0; i < N; ++i) {
                    uint64_t k = kb * N + i;
                    s.act[i] = k < K ? aRaw[ci.m * K + k] : 0;
                }
            } else {
                Addr base = aRowAddr(ci.actTile, ci.m);
                for (unsigned j = 0; j < actLanes; ++j) {
                    auto *st = new AccessState(Stream::Act, genCycle, j,
                                               curTick());
                    actQ(j, base + 8 * j).push(
                        spmRead(base + 8 * j, st), true);
                    s.pending++;
                    if (lockstep)
                        stepReadsUnsent++;
                }
            }
        }
        if (ci.wTile >= 0) {
            stats.weightBytes += N;
            if (idealSpm) {
                std::memcpy(s.wgt.data(),
                            &wPacked[(ci.wTile * N + ci.r) * N], N);
            } else if (weightStore == WeightStore::Separate) {
                // The staging buffer has its own port: no bank traffic.
                std::memcpy(s.wgt.data(),
                            &wStage[((ci.wTile % weightSlots) * N + ci.r) * N],
                            N);
            } else {
                Addr base = wRowAddr(ci.wTile, ci.r);
                for (unsigned j = 0; j < wgtLanes; ++j) {
                    auto *st = new AccessState(Stream::Weight, genCycle, j,
                                               curTick());
                    wgtQ(j, base + 8 * j).push(
                        spmRead(base + 8 * j, st), true);
                    s.pending++;
                    if (lockstep)
                        stepReadsUnsent++;
                }
            }
        }
        if (psumRmw && !idealSpm) {
            // Accumulating across K tiles: the partial sum leaving a lane in
            // this step must be added to the stored one, so that word is one
            // more read the step needs.  A row's first K tile has nothing to
            // read.  The read data itself is not used here: the array model
            // keeps the running sum, and verification checks the result.
            for (unsigned j = 0; j < psumLanes; ++j) {
                const uint64_t lag = N + 2 * j + 1;
                if (genCycle < lag)
                    break;
                const CycleInfo pi = decode(genCycle - lag);
                if (pi.actTile < 0 || uint64_t(pi.actTile) % nKb == 0)
                    continue;
                const Addr a = cRowAddr(pi.actTile, pi.m) + 8 * j;
                auto *st = new AccessState(Stream::PsumRead, genCycle, j,
                                           curTick());
                psumReadQ(j, a).push(spmRead(a, st), true);
                s.pending++;
                stats.psumReadBytes += 8;
                if (lockstep)
                    stepReadsUnsent++;
            }
        }
        if (lockstep) {
            sendPsums(genCycle);
            genCycle++;
            break;
        }
        genCycle++;
    }
}

bool
SystolicAccel::tryExecute(bool &dma_blocked)
{
    const uint64_t c = sched;
    const Slot &s = slots[c % lookahead];
    if (lockstep) {
        // The step is frozen until the banks took all of its words and its
        // read data is back.  Its reads are charged first: a stall with no
        // read left to serve can only be partial sums.
        const bool issued = s.cycle == c;
        if (!issued && !stepReadsUnsent && !stepWritesUnsent) {
            dma_blocked = true;
            return false;
        }
        const bool newest = c + 1 == genCycle || !issued;
        if (stepReadsUnsent && newest) {
            stats.spmReadStallCycles++;
            lastStall = StallCause::Read;
            return false;
        }
        if (stepWritesUnsent && newest) {
            stats.spmWriteStallCycles++;
            lastStall = StallCause::Write;
            return false;
        }
        if (s.pending) {
            stats.spmReadStallCycles++;
            lastStall = StallCause::Read;
            return false;
        }
        execute(c);
        sched++;
        return true;
    }
    if (s.cycle != c) {
        // Reads for this cycle could not even be issued: the weight tile
        // it needs has not arrived from DRAM yet.
        dma_blocked = true;
        DPRINTF(SystolicAccel, "cycle %d: stall (weights not in SPM)\n", c);
        return false;
    }
    if (s.pending) {
        stats.spmReadStallCycles++;
        lastStall = StallCause::Read;
        DPRINTF(SystolicAccel, "cycle %d: stall (%d reads pending)\n", c,
                s.pending);
        return false;
    }
    if (!idealSpm) {
        // Partial sums that must leave the array this cycle need room in
        // the write buffer: psum_queue_depth entries per lane (lane mode)
        // or psum_queue_depth * N/2 entries shared by all banks.
        unsigned due = 0;
        for (unsigned j = 0; j < psumLanes; ++j) {
            const uint64_t lag = N + 2 * j + 1;
            if (c < lag)
                break;
            const PsumRow &pr = psumRing[(c - lag) % psumRing.size()];
            if (pr.cycle != c - lag)
                continue;
            due++;
            if (!perBank && psumQ(j, spmBase).queue.size() >=
                                psumQueueDepth) {
                stats.spmWriteStallCycles++;
                lastStall = StallCause::Write;
                DPRINTF(SystolicAccel, "cycle %d: stall (psum lane %d "
                        "full)\n", c, j);
                return false;
            }
        }
        if (perBank &&
            outstandingWrites + due > uint64_t(psumQueueDepth) * psumLanes) {
            stats.spmWriteStallCycles++;
            lastStall = StallCause::Write;
            DPRINTF(SystolicAccel, "cycle %d: stall (write buffer full, "
                    "%d outstanding)\n", c, outstandingWrites);
            return false;
        }
    }
    execute(c);
    sched++;
    return true;
}

void
SystolicAccel::execute(uint64_t c)
{
    const CycleInfo ci = decode(c);
    const Slot &s = slots[c % lookahead];

    // Weight row r of the next tile goes into the shadow registers.
    if (ci.wTile >= 0) {
        std::memcpy(&wNext[ci.r * N], s.wgt.data(), N);
        if (ci.r == N - 1)
            weightTilesConsumed = ci.wTile + 1;
    }

    // An activation row enters the left edge.  The first row of a tile
    // carries the `first` flag that swaps shadow -> active weights.
    if (ci.actTile >= 0) {
        if (ci.m == 0)
            wAct = wNext;
        const uint64_t nb = ci.actTile / nKb;
        int32_t *acc_row = &acc[(nb * M + ci.m) * N];
        for (unsigned j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (unsigned i = 0; i < N; ++i)
                sum += int32_t(int8_t(s.act[i])) *
                       int32_t(wAct[i * N + j]);
            acc_row[j] = int32_t(uint32_t(acc_row[j]) + uint32_t(sum));
        }
        PsumRow &pr = psumRing[c % psumRing.size()];
        pr.cycle = c;
        pr.addr = cRowAddr(ci.actTile, ci.m);
        std::copy(acc_row, acc_row + N, pr.vals.begin());
        stats.usefulRows++;
        DPRINTF(SystolicAccel, "cycle %d: tile %d row %d enters\n", c,
                ci.actTile, ci.m);
    }

    if (!lockstep)
        sendPsums(c);
}

void
SystolicAccel::sendPsums(uint64_t c)
{
    // Partial sums leave the bottom edge, skewed one column per cycle;
    // lane j carries columns 2j and 2j+1 (one 8-byte bank word).
    for (unsigned j = 0; j < psumLanes; ++j) {
        const uint64_t lag = N + 2 * j + 1;
        if (c < lag)
            break;
        PsumRow &pr = psumRing[(c - lag) % psumRing.size()];
        if (pr.cycle != c - lag)
            continue;
        stats.psumBytes += 8;
        if (psumExternal) {
            // No bank port: it reaches the C region without timing.
            spmFunctionalWrite(pr.addr + 8 * j, &pr.vals[2 * j]);
        } else if (!idealSpm) {
            auto *st = new AccessState(Stream::Psum, c, j, curTick());
            psumQ(j, pr.addr + 8 * j).push(
                spmWrite(pr.addr + 8 * j, &pr.vals[2 * j], st), false);
            outstandingWrites++;
            if (lockstep)
                stepWritesUnsent++;
        }
    }
}

void
SystolicAccel::writeTraceHeader()
{
    *traceOut << "cycle,sched_step,stalled,stall_cause";
    for (unsigned b = 0; b < spmBanks; ++b)
        *traceOut << ",bank" << b << "_op,bank" << b << "_addr";
    *traceOut << "\n";
}

void
SystolicAccel::writeTraceRow(uint64_t step, bool stalled)
{
    static const char *const causes[] = {"none", "read", "write", "dma",
                                         "drain"};
    *traceOut << traceCycle++ << ',' << step << ',' << int(stalled) << ','
              << causes[int(stalled || lastStall == StallCause::Drain
                            ? lastStall : StallCause::None)];
    for (auto &op : traceOps) {
        if (op.read && op.write) {
            *traceOut << ",RW," << op.readRow << '/' << op.writeRow;
        } else if (op.read) {
            *traceOut << ",R," << op.readRow;
        } else if (op.write) {
            *traceOut << ",W," << op.writeRow;
        } else {
            *traceOut << ",idle,";
        }
        op = TraceOp();
    }
    *traceOut << '\n';
}

bool
SystolicAccel::lanesEmpty() const
{
    for (auto &l : lanes)
        if (!l->queue.empty())
            return false;
    return true;
}

bool
SystolicAccel::drained() const
{
    return !outstandingWrites && !dramOutstanding && lanesEmpty();
}

// ---------------------------------------------------------------------
// Responses

bool
SystolicAccel::recvSpmResp(PacketPtr pkt)
{
    auto *st = dynamic_cast<AccessState *>(pkt->popSenderState());
    panic_if(!st, "SPM response without SystolicAccel sender state");

    switch (st->stream) {
      case Stream::Act:
      case Stream::Weight: {
        Slot &s = slots[st->id % lookahead];
        panic_if(s.cycle != st->id, "SPM data for cycle %d but slot holds "
                 "cycle %d", st->id, s.cycle);
        auto &dst = st->stream == Stream::Act ? s.act : s.wgt;
        std::memcpy(&dst[8 * st->word], pkt->getConstPtr<uint8_t>(), 8);
        assert(s.pending);
        s.pending--;
        const double lat = double(curTick() - st->created) / clockPeriod();
        stats.spmReads++;
        stats.spmReadLatencyTotal += lat;
        stats.spmServiceLatencyTotal +=
            double(curTick() - st->sent) / clockPeriod();
        stats.spmReadLatency.sample(lat);
        break;
      }
      case Stream::PsumRead: {
        Slot &s = slots[st->id % lookahead];
        panic_if(s.cycle != st->id, "Psum read for cycle %d but slot holds "
                 "cycle %d", st->id, s.cycle);
        assert(s.pending);
        s.pending--;
        break;
      }
      case Stream::Psum:
        assert(outstandingWrites);
        outstandingWrites--;
        break;
      case Stream::Dma: {
        const unsigned slot = st->id % weightSlots;
        assert(dmaRemaining[slot]);
        if (--dmaRemaining[slot] == 0)
            dmaTileLanded(st->id);
        break;
      }
      case Stream::ALoad:
        assert(aPending);
        aPending--;
        break;
      case Stream::CStoreRead: {
        auto it = cInFlight.find(st->id);
        panic_if(it == cInFlight.end(), "C chunk %d not in flight", st->id);
        CChunk &ch = it->second;
        std::memcpy(&ch.data[8 * st->word], pkt->getConstPtr<uint8_t>(),
                    8);
        if (--ch.pending == 0) {
            // Whole-line WriteLineReq: it invalidates any copy the host's
            // caches hold (the classic caches cannot merge a partial DMA
            // write into a dirty line).  A partial last line is padded
            // with zeros; the host's C buffer is 4 KiB aligned.
            ch.data.resize(lineSize, 0);
            auto *ws = new AccessState(Stream::CStoreWrite, st->id, 0,
                                       curTick());
            auto req = std::make_shared<Request>(
                cLineAddr(st->id), lineSize, 0, requestorId);
            PacketPtr wp = new Packet(req, MemCmd::WriteLineReq);
            wp->allocate();
            wp->setData(ch.data.data());
            wp->pushSenderState(ws);
            dramPort.send(wp);
            stats.cStoreBytes += ch.bytes;
            cDramPending++;
            cInFlight.erase(it);
        }
        break;
      }
      default:
        panic("unexpected stream on an SPM port");
    }
    delete st;
    delete pkt;
    return true;
}

bool
SystolicAccel::recvDramResp(PacketPtr pkt)
{
    auto *st = dynamic_cast<AccessState *>(pkt->popSenderState());
    panic_if(!st, "DRAM response without SystolicAccel sender state");

    if (st->stream == Stream::CStoreWrite) {
        assert(cDramPending);
        cDramPending--;
    } else if (st->stream == Stream::ALoadRead) {
        assert(dramOutstanding);
        dramOutstanding--;
        aLinesDone++;
        stats.dramBytes += pkt->getSize();
        // Scatter the row-major line into the K-blocked SPM layout.  K and
        // N are multiples of 8, so every 8-byte word stays in one row and
        // one K-block.
        const uint8_t *data = pkt->getConstPtr<uint8_t>();
        const uint64_t base = st->id * lineSize;
        for (unsigned w = 0; w < pkt->getSize() / 8; ++w) {
            const uint64_t o = base + 8 * w;
            const uint64_t m = o / K, k = o % K;
            const Addr a = aBase + ((k / N) * M + m) * N + k % N;
            auto *ws = new AccessState(Stream::ALoad, st->id, w, curTick());
            dmaQ(dmaLaneRR++, a).push(
                spmWrite(a, data + 8 * w, ws), false);
            aPending++;
            stats.aLoadBytes += 8;
        }
    } else if (st->stream == Stream::DramRead) {
        assert(dramOutstanding);
        dramOutstanding--;
        stats.dramBytes += lineSize;
        const uint64_t t = st->id;
        const unsigned slot = t % weightSlots;
        if (idealSpm) {
            if (--dmaLinesLeft[slot] == 0)
                dmaTileLanded(t);
        } else if (weightStore != WeightStore::Spm) {
            // The line reaches its weight slot without crossing the banks.
            const uint8_t *data = pkt->getConstPtr<uint8_t>();
            if (weightStore == WeightStore::Separate) {
                std::memcpy(&wStage[uint64_t(slot) * N * N +
                                    st->word * lineSize], data, lineSize);
            } else {
                const Addr base = wRowAddr(t, 0) + st->word * lineSize;
                for (unsigned w = 0; w < lineSize / 8; ++w)
                    spmFunctionalWrite(base + 8 * w, data + 8 * w);
            }
            if (--dmaLinesLeft[slot] == 0)
                dmaTileLanded(t);
        } else {
            // Split the 64-byte line into bank words.
            const Addr base = wRowAddr(t, 0) + st->word * lineSize;
            const uint8_t *data = pkt->getConstPtr<uint8_t>();
            for (unsigned w = 0; w < lineSize / 8; ++w) {
                auto *ws = new AccessState(Stream::Dma, t, w, curTick());
                dmaQ(dmaLaneRR, base + 8 * w).push(
                    spmWrite(base + 8 * w, data + 8 * w, ws), false);
                dmaLaneRR = (dmaLaneRR + 1) % dmaLanes;
                stats.dmaSpmBytes += 8;
            }
        }
    } else {
        panic("unexpected DRAM response");
    }
    delete st;
    delete pkt;
    return true;
}

Addr
SystolicAccel::cLineAddr(uint64_t chunk) const
{
    if (!nTile)
        return cDramBase + chunk * lineSize;
    // A slice of each row: C in DRAM stays row-major over all the columns.
    const uint64_t per_row = Nout * 4 / lineSize;
    return cDramBase + (chunk / per_row) * layerNout * 4 + nColsDone * 4 +
           (chunk % per_row) * lineSize;
}

void
SystolicAccel::dmaTileLanded(uint64_t tile)
{
    DPRINTF(SystolicAccelDma, "DMA tile %d landed in SPM\n", tile);
    dmaDone[tile] = true;
}

// ---------------------------------------------------------------------
// End of layer

void
SystolicAccel::layerDone(bool complete)
{
    stats.layerCycles += layerCycles;
    regs[RegLayerCycles / 8] = layerCycles;
    const bool checked = verify && complete;
    uint64_t bad = 0;
    if (checked) {
        doVerify();
        bad = stats.verifyMismatches.value() - verifyBaseline;
    }
    regs[RegVerify / 8] = bad;
    if (traceOut)
        traceOut->flush();
    phase = Phase::Done;
    status = 2 | (bad ? 4 : 0);     // done (+ error)

    const double util = double(M * nTiles) / double(cyclesElapsed);
    inform("SystolicAccel done: %d cycles (ideal %d), PE utilization "
           "%.4f, stalls spm=%d dma=%d, layer %d cycles (load %d, store "
           "%d), verify: %s",
           cyclesElapsed, schedEnd, complete ? util : 0.0,
           (uint64_t)stats.spmStallCycles.value(),
           (uint64_t)stats.dmaStallCycles.value(), layerCycles,
           (uint64_t)stats.loadCycles.value(),
           (uint64_t)stats.storeCycles.value(),
           !checked ? "skipped" : bad ? "FAIL" : "PASS");
    if (!hostDriven)
        exitSimLoop("systolic accelerator finished", 0);
}

void
SystolicAccel::doVerify()
{
    // Operands: the files (standalone) or what the host put in DRAM.  With
    // M tiling this checks the current M tile.
    std::vector<int8_t> a_mat, w_packed;
    if (hostDriven) {
        a_mat.resize(M * K);
        w_packed.resize(nTiles * N * N);
        // One cache line per functional access: a multi-line functional
        // packet does not merge dirty data held in the host's caches.
        const size_t chunk = lineSize;
        for (size_t o = 0; o < a_mat.size(); o += chunk)
            functionalRead(dramPort, aDramBase + o, &a_mat[o],
                           std::min(chunk, a_mat.size() - o));
        for (size_t o = 0; o < w_packed.size(); o += chunk)
            functionalRead(dramPort, wDramBase + o, &w_packed[o],
                           std::min(chunk, w_packed.size() - o));
    } else {
        a_mat = aRaw;
        w_packed = wPacked;
    }

    // Reference GEMM, INT32 wrapping, from row-major A and packed W.
    std::vector<int32_t> ref(M * Nout, 0);
    for (uint64_t t = 0; t < nTiles; ++t) {
        const uint64_t nb = t / nKb, kb = t % nKb;
        for (unsigned r = 0; r < N; ++r) {
            const uint64_t k = kb * N + r;
            if (k >= K)
                continue;
            const int8_t *wrow = &w_packed[(t * N + r) * N];
            for (uint64_t m = 0; m < M; ++m) {
                const int32_t a = a_mat[m * K + k];
                if (!a)
                    continue;
                int32_t *crow = &ref[m * Nout + nb * N];
                const unsigned cols = std::min<uint64_t>(N, Nout - nb * N);
                for (unsigned j = 0; j < cols; ++j)
                    crow[j] = int32_t(uint32_t(crow[j]) +
                                      uint32_t(a * wrow[j]));
            }
        }
    }

    uint64_t checked = 0, bad = 0;
    auto check = [&](const char *what, uint64_t m, uint64_t n, int32_t got,
                     int32_t exp) {
        checked++;
        if (got != exp && bad++ < 5)
            warn("verify %s: C[%d][%d] = %d, expected %d", what, m, n, got,
                 exp);
    };

    std::vector<int32_t> golden, dram_c;
    const auto &p = params();
    if (!hostDriven && !p.c_golden_file.empty()) {
        golden.resize(M * Nout);
        readFile(p.c_golden_file, golden.data(), golden.size() * 4);
    }
    if (hostDriven) {
        dram_c.resize(M * Nout);
        const size_t bytes = dram_c.size() * 4, chunk = lineSize;
        auto *dst = reinterpret_cast<uint8_t *>(dram_c.data());
        for (size_t o = 0; o < bytes; o += chunk) {
            const Addr a = nTile ? cLineAddr(o / chunk) : cDramBase + o;
            functionalRead(dramPort, a, dst + o,
                           std::min(chunk, bytes - o));
        }
    }

    std::vector<int32_t> spm_row(N);
    std::vector<int32_t> out(M * Nout);
    for (uint64_t nb = 0; nb < nNb; ++nb) {
        for (uint64_t m = 0; m < M; ++m) {
            const int32_t *acc_row = &acc[(nb * M + m) * N];
            if (!idealSpm) {
                Addr base = cBase + (nb * M + m) * N * 4;
                for (unsigned w = 0; w < N / 2; ++w)
                    spmFunctionalRead(base + 8 * w, &spm_row[2 * w]);
            }
            for (unsigned j = 0; j < N; ++j) {
                const uint64_t n = nb * N + j;
                if (n >= Nout)
                    continue;
                out[m * Nout + n] = acc_row[j];
                check("array", m, n, acc_row[j], ref[m * Nout + n]);
                if (!idealSpm)
                    check("spm", m, n, spm_row[j], acc_row[j]);
                if (!golden.empty())
                    check("golden", m, n, golden[m * Nout + n],
                          ref[m * Nout + n]);
                if (!dram_c.empty())
                    check("dram", m, n, dram_c[m * Nout + n], acc_row[j]);
            }
        }
    }
    stats.verifyChecked += checked;
    stats.verifyMismatches += bad;

    if (!p.c_out_file.empty()) {
        std::ofstream f(p.c_out_file, std::ios::binary);
        f.write(reinterpret_cast<const char *>(out.data()), out.size() * 4);
    }
}

// ---------------------------------------------------------------------
// Statistics

SystolicAccel::AccelStats::AccelStats(SystolicAccel &a)
    : statistics::Group(&a), accel(a),
      ADD_STAT(layers, statistics::units::Count::get(),
               "Layers started"),
      ADD_STAT(totalCycles, statistics::units::Cycle::get(),
               "Compute-phase cycles (first array cycle to last psum ack)"),
      ADD_STAT(scheduleCycles, statistics::units::Cycle::get(),
               "Stall-free schedule length (analytical cycle count)"),
      ADD_STAT(activeCycles, statistics::units::Cycle::get(),
               "Cycles in which the array advanced"),
      ADD_STAT(stallCycles, statistics::units::Cycle::get(),
               "Cycles the whole array was frozen"),
      ADD_STAT(spmStallCycles, statistics::units::Cycle::get(),
               "Stall cycles caused by the scratchpad (reads or writes)"),
      ADD_STAT(spmReadStallCycles, statistics::units::Cycle::get(),
               "Stalls waiting for activation/weight words"),
      ADD_STAT(spmWriteStallCycles, statistics::units::Cycle::get(),
               "Stalls because the partial-sum write buffer was full"),
      ADD_STAT(dmaStallCycles, statistics::units::Cycle::get(),
               "Stalls waiting for a weight tile to arrive from DRAM"),
      ADD_STAT(drainCycles, statistics::units::Cycle::get(),
               "Cycles after the schedule waiting for write acks"),
      ADD_STAT(fillDrainCycles, statistics::units::Cycle::get(),
               "Compute cycles outside the steady-state window (array "
               "filling, draining, write acks)"),
      ADD_STAT(steadyStateCycles, statistics::units::Cycle::get(),
               "Compute cycles from the first row's last partial sum to the "
               "last row entering the array"),
      ADD_STAT(steadyStateRows, statistics::units::Count::get(),
               "Activation rows that entered in the steady-state window"),
      ADD_STAT(steadyStateUtilization, statistics::units::Ratio::get(),
               "steadyStateRows / steadyStateCycles (NaN if a layer is too "
               "short to have a steady state)"),
      ADD_STAT(mTiles, statistics::units::Count::get(),
               "M tiles processed (one per layer without M tiling)"),
      ADD_STAT(nSlices, statistics::units::Count::get(),
               "M tile x N_out slice pairs processed (one without tiling)"),
      ADD_STAT(psumReadBytes, statistics::units::Byte::get(),
               "Old partial-sum bytes read back (psum_rmw)"),
      ADD_STAT(loadCycles, statistics::units::Cycle::get(),
               "Host-driven: cycles loading A from DRAM into the SPM"),
      ADD_STAT(storeCycles, statistics::units::Cycle::get(),
               "Host-driven: cycles storing C from the SPM to DRAM"),
      ADD_STAT(layerCycles, statistics::units::Cycle::get(),
               "START to DONE (load + compute + store)"),
      ADD_STAT(usefulRows, statistics::units::Count::get(),
               "Activation rows streamed (each = N*N MACs)"),
      ADD_STAT(realMacs, statistics::units::Count::get(),
               "MACs of the unpadded GEMM, M*K*Nout"),
      ADD_STAT(peUtilization, statistics::units::Ratio::get(),
               "usefulRows / totalCycles (MAC slots busy incl. padding)"),
      ADD_STAT(macUtilization, statistics::units::Ratio::get(),
               "realMacs / (N*N*totalCycles), excludes zero padding"),
      ADD_STAT(idealUtilization, statistics::units::Ratio::get(),
               "usefulRows / scheduleCycles (utilization with no stalls)"),
      ADD_STAT(layerUtilization, statistics::units::Ratio::get(),
               "usefulRows / layerCycles (includes A load and C store)"),
      ADD_STAT(actBytes, statistics::units::Byte::get(),
               "Activation bytes read from the SPM"),
      ADD_STAT(weightBytes, statistics::units::Byte::get(),
               "Weight bytes read from the SPM"),
      ADD_STAT(psumBytes, statistics::units::Byte::get(),
               "Partial-sum bytes written to the SPM"),
      ADD_STAT(dmaSpmBytes, statistics::units::Byte::get(),
               "Weight bytes written into the SPM by the DMA"),
      ADD_STAT(dramBytes, statistics::units::Byte::get(),
               "Bytes read from DRAM by the DMA (weights + A)"),
      ADD_STAT(aLoadBytes, statistics::units::Byte::get(),
               "Host-driven: A bytes written into the SPM"),
      ADD_STAT(cStoreBytes, statistics::units::Byte::get(),
               "Host-driven: C bytes written to DRAM"),
      ADD_STAT(arrayBytesPerCycle,
               statistics::units::Rate<statistics::units::Byte,
                                       statistics::units::Cycle>::get(),
               "Delivered array bandwidth, (act+weight+psum)/totalCycles"),
      ADD_STAT(demandBytesPerCycle,
               statistics::units::Rate<statistics::units::Byte,
                                       statistics::units::Cycle>::get(),
               "Measured demand D, (act+weight+psum)/scheduleCycles"),
      ADD_STAT(modelDemand,
               statistics::units::Rate<statistics::units::Byte,
                                       statistics::units::Cycle>::get(),
               "Eq. (1): (5NM + N^2) / max(M, N)"),
      ADD_STAT(spmReads, statistics::units::Count::get(),
               "8-byte SPM reads (activations + weights)"),
      ADD_STAT(spmReadLatencyTotal, statistics::units::Cycle::get(),
               "Sum of read latencies, generation to data return"),
      ADD_STAT(spmServiceLatencyTotal, statistics::units::Cycle::get(),
               "Sum of read latencies, crossbar accept to data return"),
      ADD_STAT(avgSpmLatency,
               statistics::units::Rate<statistics::units::Cycle,
                                       statistics::units::Count>::get(),
               "Scratchpad AMAT in cycles (includes bank-conflict waits)"),
      ADD_STAT(avgSpmServiceLatency,
               statistics::units::Rate<statistics::units::Cycle,
                                       statistics::units::Count>::get(),
               "Average read latency once the crossbar accepted the request"),
      ADD_STAT(spmReadLatency, statistics::units::Cycle::get(),
               "Distribution of SPM read latency in cycles"),
      ADD_STAT(laneRetries, statistics::units::Count::get(),
               "Sends refused by the crossbar/bank (conflicts)"),
      ADD_STAT(bankAccesses, statistics::units::Count::get(),
               "8-byte accesses sent to each SPM bank (bank balance)"),
      ADD_STAT(pioReads, statistics::units::Count::get(),
               "Host register reads"),
      ADD_STAT(pioWrites, statistics::units::Count::get(),
               "Host register writes"),
      ADD_STAT(verifyChecked, statistics::units::Count::get(),
               "Output elements compared"),
      ADD_STAT(verifyMismatches, statistics::units::Count::get(),
               "Output elements that did not match")
{
    const double pes = double(a.N) * a.N;
    peUtilization = usefulRows / totalCycles;
    steadyStateUtilization = steadyStateRows / steadyStateCycles;
    macUtilization = realMacs / (totalCycles * statistics::constant(pes));
    idealUtilization = usefulRows / scheduleCycles;
    layerUtilization = usefulRows / layerCycles;
    arrayBytesPerCycle = (actBytes + weightBytes + psumBytes) / totalCycles;
    demandBytesPerCycle =
        (actBytes + weightBytes + psumBytes) / scheduleCycles;
    avgSpmLatency = spmReadLatencyTotal / spmReads;
    avgSpmServiceLatency = spmServiceLatencyTotal / spmReads;
}

void
SystolicAccel::AccelStats::regStats()
{
    statistics::Group::regStats();
    spmReadLatency.init(0, 63, 1);
    bankAccesses.init(accel.spmBanks);
}

} // namespace gem5
