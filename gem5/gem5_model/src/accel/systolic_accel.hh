/*
 * EEL5764 (Fall 2026) Group 4 -- SystolicAccel.
 *
 * Timing + functional model of a weight-stationary INT8 N x N systolic
 * array fed by a banked scratchpad (SPM).  See SystolicAccel.py for the
 * port layout and gem5_model/README.txt for the schedule.
 *
 * Two ways to run a layer:
 *   host_driven = false  (standalone)  the GEMM shape and operands come
 *       from parameters; A is preloaded into the SPM and W into DRAM at
 *       tick 0; the layer starts at startup() and ends the simulation.
 *   host_driven = true   a RISC-V program writes A and packed W to DRAM,
 *       programs the memory-mapped registers (pio port) and sets START.
 *       The accelerator DMAs A into the SPM, runs the layer (weights are
 *       DMA'd tile by tile), DMAs C back to DRAM and raises DONE.
 */

#ifndef __ACCEL_SYSTOLIC_ACCEL_HH__
#define __ACCEL_SYSTOLIC_ACCEL_HH__

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/statistics.hh"
#include "mem/packet.hh"
#include "mem/port.hh"
#include "mem/tport.hh"
#include "params/SystolicAccel.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class SystolicAccel : public ClockedObject
{
  public:
    PARAMS(SystolicAccel);
    SystolicAccel(const Params &p);

    Port &getPort(const std::string &if_name,
                  PortID idx=InvalidPortID) override;
    void init() override;
    void startup() override;

    /** Control/status registers (64-bit, byte offsets from pio_addr). */
    enum Reg : Addr
    {
        RegCtrl = 0x00,         // W: 1 = start a layer
        RegStatus = 0x08,       // R: bit0 busy, bit1 done, bit2 error
        RegM = 0x10,            // RW: GEMM M
        RegK = 0x18,            // RW: GEMM K  (multiple of 8)
        RegNout = 0x20,         // RW: GEMM N_out (even)
        RegAAddr = 0x28,        // RW: DRAM address of A, row-major int8
        RegWAddr = 0x30,        // RW: DRAM address of packed W tiles
        RegCAddr = 0x38,        // RW: DRAM address for C, row-major int32
        RegLayerCycles = 0x40,  // R: cycles of the last layer (all phases)
        RegArrayDim = 0x48,     // R: N
        RegComputeCycles = 0x50,// R: cycles of the compute phase
        RegVerify = 0x58,       // R: internal check mismatches (0 = ok)
        RegEnd = 0x60
    };

  private:
    enum class Stream : uint8_t
    {
        Act, Weight, Psum,      // array traffic (SPM)
        Dma, DramRead,          // weight refill: SPM write / DRAM read
        ALoad, ALoadRead,       // A load:  SPM write / DRAM read
        CStoreRead, CStoreWrite // C store: SPM read / DRAM write
    };

    enum class Phase : uint8_t { Idle, LoadA, Compute, StoreC, Done };

    /** Carried on every timing packet so the response can be routed. */
    struct AccessState : public Packet::SenderState
    {
        Stream stream;
        uint64_t id;      // array cycle, tile, line or chunk
        unsigned word;    // 8-byte word index within the row / line
        Tick created;     // when the access was generated (AMAT start)
        Tick sent = 0;    // when the crossbar accepted it
        int bank = -1;    // SPM bank (for the per-bank statistics)
        AccessState(Stream s, uint64_t i, unsigned w, Tick c)
            : stream(s), id(i), word(w), created(c) {}
    };

    /** One scratchpad request queue (a bank or an edge lane). */
    class LanePort : public RequestPort
    {
      public:
        LanePort(const std::string &name, SystolicAccel &owner,
                 unsigned idx);
        std::deque<PacketPtr> queue;
        unsigned frontReads = 0;    // reads at the head (reads_first)
        bool waitRetry = false;
        Tick lastSend = MaxTick;
        /** Send the head packet if allowed; returns true if sent. */
        bool trySend();
        /** Queue a request (see the bank_policy parameter). */
        void push(PacketPtr pkt, bool is_read);

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;
        void recvRangeChange() override {}

      private:
        SystolicAccel &owner;
        unsigned idx;
    };

    class DramPort : public RequestPort
    {
      public:
        DramPort(const std::string &name, SystolicAccel &owner)
            : RequestPort(name), owner(owner) {}
        /** Packets refused by the memory bus, resent in order. */
        std::deque<PacketPtr> blocked;
        /** Send now, or queue behind earlier refused packets. */
        void send(PacketPtr pkt);
        bool busy() const { return !blocked.empty(); }

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;
        void recvRangeChange() override {}

      private:
        SystolicAccel &owner;
    };

    /** Memory-mapped control registers, reached from the host CPU. */
    class PioPort : public SimpleTimingPort
    {
      public:
        PioPort(const std::string &name, SystolicAccel &owner)
            : SimpleTimingPort(name, &owner), owner(owner) {}

      protected:
        Tick recvAtomic(PacketPtr pkt) override;
        AddrRangeList getAddrRanges() const override;

      private:
        SystolicAccel &owner;
    };

    /** What the static schedule does in one array cycle. */
    struct CycleInfo
    {
        int64_t actTile = -1;   // tile whose activation row m enters
        uint32_t m = 0;
        int64_t wTile = -1;     // tile whose weight row r is loaded
        uint32_t r = 0;
    };

    /** Read data staged for one array cycle (ring of `lookahead`). */
    struct Slot
    {
        uint64_t cycle = UINT64_MAX;
        unsigned pending = 0;
        std::vector<uint8_t> act;
        std::vector<uint8_t> wgt;
    };

    /** A computed output row waiting to leave the bottom edge. */
    struct PsumRow
    {
        uint64_t cycle = UINT64_MAX;    // array cycle its row entered
        Addr addr = 0;                  // SPM address of the C row
        std::vector<int32_t> vals;      // N accumulated INT32 values
    };

    /** A 64-byte piece of C being gathered from the SPM for DRAM. */
    struct CChunk
    {
        std::vector<uint8_t> data;
        unsigned pending = 0;
        unsigned bytes = 0;
    };

    // ---------------------------------------------------------------
    // Hardware parameters (fixed)
    const unsigned N;
    const bool doubleBuffer;
    const unsigned actLanes, wgtLanes, psumLanes, dmaLanes;
    const unsigned lookahead, psumQueueDepth, weightSlots;
    const unsigned dmaMaxOutstanding;
    const unsigned spmBanks;
    const bool perBank, moduloBanks, readsFirst;
    const bool idealSpm, idealDram, verify, hostDriven;
    const uint64_t maxCycles;
    const Addr spmBase, spmSize, bankBytes;
    const Addr pioAddr, pioSize;
    const Cycles pioLatency;
    static constexpr unsigned lineSize = 64;

    // Layer geometry (set by configure())
    uint64_t M = 0, K = 0, Nout = 0;
    uint64_t nKb = 0, nNb = 0, nTiles = 0, T = 0;
    Addr aBase = 0, wBase = 0, cBase = 0, spmEnd = 0;
    Addr wDramBase = 0, aDramBase = 0, cDramBase = 0;
    uint64_t schedEnd = 0;          // one past the last scheduled cycle

    RequestorID requestorId;
    std::vector<std::unique_ptr<LanePort>> lanes;
    DramPort dramPort;
    PioPort pioPort;
    EventFunctionWrapper tickEvent;
    EventFunctionWrapper startEvent;

    // Registers
    uint64_t regs[RegEnd / 8] = {};
    uint64_t status = 0;

    // ---------------------------------------------------------------
    // Functional data (standalone mode keeps host-side copies)
    std::vector<int8_t> aRaw, wRaw;     // row-major, unpadded
    std::vector<int8_t> wPacked;        // tile-major N x N tiles
    std::vector<int32_t> acc;           // [nb][m][N] accumulators
    std::vector<int8_t> wAct, wNext;    // active / shadow weight tile

    // ---------------------------------------------------------------
    // Run-time state
    Phase phase = Phase::Idle;
    uint64_t sched = 0;         // next array cycle to execute
    uint64_t genCycle = 0;      // next array cycle whose reads to issue
    std::vector<Slot> slots;
    std::vector<PsumRow> psumRing;
    uint64_t outstandingWrites = 0;     // psum writes queued or in flight
    uint64_t cyclesElapsed = 0;         // compute phase
    uint64_t layerCycles = 0;           // all phases
    uint64_t phaseCycles = 0;

    // Weight DMA state
    std::vector<bool> dmaDone;          // per tile: weights in SPM
    std::vector<unsigned> dmaRemaining; // per slot: SPM word acks left
    std::vector<unsigned> dmaLinesLeft; // per slot: DRAM lines left
    uint64_t dmaIssueTile = 0;
    unsigned dmaIssueLine = 0;
    unsigned dramOutstanding = 0;
    uint64_t weightTilesConsumed = 0;
    unsigned dmaLaneRR = 0;

    // A-load / C-store state
    uint64_t aLines = 0, aLinesIssued = 0, aLinesDone = 0, aPending = 0;
    uint64_t cChunks = 0, cChunksIssued = 0, cDramPending = 0;
    std::map<uint64_t, CChunk> cInFlight;

    // ---------------------------------------------------------------
    void configure(uint64_t m, uint64_t k, uint64_t n);
    void resetRunState();
    void startLayer();
    void layerDone(bool complete);
    Tick pioAccess(PacketPtr pkt);

    CycleInfo decode(uint64_t c) const;
    Addr aRowAddr(uint64_t tile, uint64_t m) const;
    Addr wRowAddr(uint64_t tile, uint64_t r) const;
    Addr cRowAddr(uint64_t tile, uint64_t m) const;

    /** Bank of a (logical) SPM address: 8-byte words, word mod BANKS. */
    unsigned bankOf(Addr a) const
    { return ((a - spmBase) >> 3) % spmBanks; }
    /**
     * Physical address of a logical SPM address.  With power-of-two
     * BANKS the crossbar's address interleaving does the banking and the
     * address is unchanged; with any other BANKS each bank is a contiguous
     * range and the accelerator's bank-select logic maps word w to bank
     * w mod BANKS, row w / BANKS.
     */
    Addr spmPhys(Addr a) const
    {
        if (!moduloBanks)
            return a;
        const uint64_t w = (a - spmBase) >> 3;
        return spmBase + (w % spmBanks) * bankBytes +
               (w / spmBanks) * 8 + (a & 7);
    }
    // Request queue for each stream.  In "bank" arbitration there is one
    // queue per bank and the address picks it; in "lane" arbitration each
    // edge lane has its own queue.
    LanePort &actQ(unsigned j, Addr a)
    { return perBank ? *lanes[bankOf(a)] : *lanes[j]; }
    LanePort &wgtQ(unsigned j, Addr a)
    { return perBank ? *lanes[bankOf(a)] : *lanes[actLanes + j]; }
    LanePort &psumQ(unsigned j, Addr a)
    { return perBank ? *lanes[bankOf(a)]
                     : *lanes[actLanes + wgtLanes + j]; }
    LanePort &dmaQ(unsigned j, Addr a)
    { return perBank ? *lanes[bankOf(a)]
                     : *lanes[actLanes + wgtLanes + psumLanes +
                              j % dmaLanes]; }

    PacketPtr makeRead(Addr addr, unsigned size, AccessState *st);
    PacketPtr makeWrite(Addr addr, unsigned size, const void *data,
                        AccessState *st);
    PacketPtr spmRead(Addr logical, AccessState *st)
    {
        st->bank = bankOf(logical);
        return makeRead(spmPhys(logical), 8, st);
    }
    PacketPtr spmWrite(Addr logical, const void *data, AccessState *st)
    {
        st->bank = bankOf(logical);
        return makeWrite(spmPhys(logical), 8, data, st);
    }

    void tick();
    void dmaAdvance();
    void aLoadAdvance();
    void cStoreAdvance();
    void generateReads();
    bool tryExecute(bool &dma_blocked);
    void execute(uint64_t c);
    bool lanesEmpty() const;
    bool drained() const;
    void doVerify();

    bool recvSpmResp(PacketPtr pkt);
    bool recvDramResp(PacketPtr pkt);
    void dmaTileLanded(uint64_t tile);

    void loadInputs();
    void functionalWrite(RequestPort &port, Addr addr, const void *data,
                         unsigned size);
    void functionalRead(RequestPort &port, Addr addr, void *data,
                        unsigned size);
    void spmFunctionalWrite(Addr logical, const void *data)
    { functionalWrite(*lanes[0], spmPhys(logical), data, 8); }
    void spmFunctionalRead(Addr logical, void *data)
    { functionalRead(*lanes[0], spmPhys(logical), data, 8); }

    // ---------------------------------------------------------------
    struct AccelStats : public statistics::Group
    {
        AccelStats(SystolicAccel &a);
        void regStats() override;

        SystolicAccel &accel;
        statistics::Scalar layers;
        statistics::Scalar totalCycles;
        statistics::Scalar scheduleCycles;
        statistics::Scalar activeCycles;
        statistics::Scalar stallCycles;
        statistics::Scalar spmStallCycles;
        statistics::Scalar spmReadStallCycles;
        statistics::Scalar spmWriteStallCycles;
        statistics::Scalar dmaStallCycles;
        statistics::Scalar drainCycles;
        statistics::Scalar loadCycles;
        statistics::Scalar storeCycles;
        statistics::Scalar layerCycles;
        statistics::Scalar usefulRows;
        statistics::Scalar realMacs;
        statistics::Formula peUtilization;
        statistics::Formula macUtilization;
        statistics::Formula idealUtilization;
        statistics::Formula layerUtilization;

        statistics::Scalar actBytes;
        statistics::Scalar weightBytes;
        statistics::Scalar psumBytes;
        statistics::Scalar dmaSpmBytes;
        statistics::Scalar dramBytes;
        statistics::Scalar aLoadBytes;
        statistics::Scalar cStoreBytes;
        statistics::Formula arrayBytesPerCycle;
        statistics::Formula demandBytesPerCycle;
        statistics::Scalar modelDemand;

        statistics::Scalar spmReads;
        statistics::Scalar spmReadLatencyTotal;
        statistics::Scalar spmServiceLatencyTotal;
        statistics::Formula avgSpmLatency;
        statistics::Formula avgSpmServiceLatency;
        statistics::Distribution spmReadLatency;
        statistics::Scalar laneRetries;
        statistics::Vector bankAccesses;

        statistics::Scalar pioReads;
        statistics::Scalar pioWrites;
        statistics::Scalar verifyChecked;
        statistics::Scalar verifyMismatches;
    } stats;
};

} // namespace gem5

#endif // __ACCEL_SYSTOLIC_ACCEL_HH__
