//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : systolic_host.c                                             //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //
// Group       : #4                                                          //
// Description : RISC-V host program for the SystolicAccel                   //
//---------------------------------------------------------------------------//
/*
 * Proposal phase 3 (host-driven mode).  Runs in gem5 SE mode on a
 * RiscvTimingSimpleCPU.
 *
 *   systolic_host MODE A.bin W.bin C_golden.bin M K N_OUT \
 *                 SHARED_VA SHARED_PA PIO_VA
 *
 *   MODE      accel  run the layer on the accelerator
 *             sw     run the same GEMM in software on this CPU (baseline
 *                    for the speedup metric)
 *   SHARED_*  a DRAM buffer the config maps into this process; the
 *             accelerator's DMA uses the physical address (SHARED_PA)
 *   PIO_VA    where the config mapped the accelerator's registers
 *
 * Statistics: the program calls m5_dump_reset_stats() right before
 * starting the layer and right after it completes, so stats.txt holds
 *   dump 0  setup (read files, pack weights)
 *   dump 1  the layer                  <- layer time, host CPI/IPC, ...
 *   dump 2  verification and exit
 *
 * Exit status 0 = the output matched C_golden.bin.
 *
 * Built with -DSYSTOLIC_HOST_EXT (systolic_host_ext) the program takes two
 * more optional arguments after PIO_VA:
 *   split-stats  dump statistics once more between programming the registers
 *                and polling, so dump 1 is the setup and dump 2 the wait
 *   prepacked    the packed weights are already in DRAM (the config loads
 *                W_packed_N<N>.bin): skip reading W and packing it
 * The default build has neither, so it stays byte-identical.
 */

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Register offsets: must match SystolicAccel::Reg in systolic_accel.hh */
enum {
    REG_CTRL = 0x00, REG_STATUS = 0x08, REG_M = 0x10, REG_K = 0x18,
    REG_NOUT = 0x20, REG_A = 0x28, REG_W = 0x30, REG_C = 0x38,
    REG_LAYER_CYCLES = 0x40, REG_ARRAY_DIM = 0x48,
    REG_COMPUTE_CYCLES = 0x50, REG_VERIFY = 0x58
};
#define STATUS_BUSY 1u
#define STATUS_DONE 2u
#define STATUS_ERROR 4u

static volatile uint64_t *regs;

static uint64_t rd(unsigned off) { return regs[off / 8]; }
static void wr(unsigned off, uint64_t v) { regs[off / 8] = v; }

/* gem5 pseudo-instruction: dump and reset statistics (m5op 0x42).
 * RISC-V encoding: opcode 0x7b with the function in bits 31:25. */
static void m5_dump_reset_stats(void)
{
    register uint64_t a0 __asm__("a0") = 0;   /* delay  */
    register uint64_t a1 __asm__("a1") = 0;   /* period */
    __asm__ volatile(".word 0x8400007b" : "+r"(a0) : "r"(a1) : "memory");
}

static void read_file(const char *path, void *dst, size_t bytes)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    size_t got = 0;
    while (got < bytes) {
        ssize_t r = read(fd, (char *)dst + got, bytes - got);
        if (r <= 0) {
            fprintf(stderr, "%s: short read (%zu of %zu bytes)\n", path,
                    got, bytes);
            exit(2);
        }
        got += (size_t)r;
    }
    close(fd);
}

static uint64_t align(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

/* Word-at-a-time copy and zero for tile rows (N bytes, a multiple of 8).
 * Tile rows are 1-4 words, so the loops are kept tight: no unrolling or
 * vectorizing (their set-up code costs more than the copy), and
 * build_host.sh passes -fno-builtin-memcpy/-memset so the compiler does not
 * turn them back into calls to the static libc's byte-loop routines. */
typedef uint64_t __attribute__((may_alias, aligned(1))) u64_unaligned;
#if defined(__clang__)
#define TIGHT_LOOP _Pragma("clang loop vectorize(disable) unroll(disable)")
#else
#define TIGHT_LOOP _Pragma("GCC unroll 1")
#endif

static inline void copy_row(int8_t *dst, const int8_t *src, uint64_t n)
{
    uint64_t i = 0;
    TIGHT_LOOP
    for (; i + 8 <= n; i += 8)
        *(u64_unaligned *)(dst + i) = *(const u64_unaligned *)(src + i);
    for (; i < n; ++i)
        dst[i] = src[i];
}

static inline void zero_row(int8_t *dst, uint64_t n)
{
    uint64_t i = 0;
    TIGHT_LOOP
    for (; i + 8 <= n; i += 8)
        *(u64_unaligned *)(dst + i) = 0;
    for (; i < n; ++i)
        dst[i] = 0;
}

int main(int argc, char **argv)
{
#ifdef SYSTOLIC_HOST_EXT
    int split_stats = 0, prepacked = 0;
    while (argc > 11) {
        --argc;
        if (strcmp(argv[argc], "split-stats") == 0) {
            split_stats = 1;
        } else if (strcmp(argv[argc], "prepacked") == 0) {
            prepacked = 1;
        } else {
            fprintf(stderr, "unknown option %s\n", argv[argc]);
            return 2;
        }
    }
#endif
    if (argc != 11) {
        fprintf(stderr, "usage: %s accel|sw A.bin W.bin C_golden.bin M K "
                "N_OUT SHARED_VA SHARED_PA PIO_VA\n", argv[0]);
        return 2;
    }
    const int sw = strcmp(argv[1], "sw") == 0;
    const uint64_t M = strtoull(argv[5], 0, 0);
    const uint64_t K = strtoull(argv[6], 0, 0);
    const uint64_t NOUT = strtoull(argv[7], 0, 0);
    uint8_t *shared = (uint8_t *)(uintptr_t)strtoull(argv[8], 0, 0);
    const uint64_t shared_pa = strtoull(argv[9], 0, 0);
    regs = (volatile uint64_t *)(uintptr_t)strtoull(argv[10], 0, 0);

    const uint64_t N = sw ? 8 : rd(REG_ARRAY_DIM);
    const uint64_t nkb = (K + N - 1) / N, nnb = (NOUT + N - 1) / N;

    /* Layout of the shared DRAM buffer */
    const uint64_t a_off = 0;
    const uint64_t w_off = align(a_off + M * K, 4096);
    const uint64_t c_off = align(w_off + nkb * nnb * N * N, 4096);
    int8_t *A = (int8_t *)(shared + a_off);
    int8_t *Wp = (int8_t *)(shared + w_off);
    int32_t *C = (int32_t *)(shared + c_off);

    int8_t *W = malloc(K * NOUT);
    int32_t *golden = malloc(M * NOUT * 4);
    if (!W || !golden) {
        fprintf(stderr, "out of memory\n");
        return 2;
    }
    read_file(argv[2], A, M * K);
#ifdef SYSTOLIC_HOST_EXT
    if (!prepacked)
#endif
    read_file(argv[3], W, K * NOUT);
    read_file(argv[4], golden, M * NOUT * 4);

#ifdef SYSTOLIC_HOST_EXT
    if (!sw && !prepacked) {
#else
    if (!sw) {
#endif
        /* Pack W into N x N tiles, tile t = nb * nkb + kb, zero padded:
         * tile row r holds W[kb*N + r][nb*N .. nb*N + N - 1].  Every byte
         * of every tile is written (data or zero padding). */
        for (uint64_t nb = 0; nb < nnb; ++nb) {
            const uint64_t cols = NOUT - nb * N < N ? NOUT - nb * N : N;
            for (uint64_t kb = 0; kb < nkb; ++kb) {
                int8_t *row = Wp + (nb * nkb + kb) * N * N;
                const int8_t *src = W + kb * N * NOUT + nb * N;
                for (uint64_t r = 0; r < N; ++r, row += N, src += NOUT) {
                    if (kb * N + r < K) {
                        copy_row(row, src, cols);
                        if (cols < N)
                            zero_row(row + cols, N - cols);
                    } else {
                        zero_row(row, N);
                    }
                }
            }
        }
    }
    if (sw)     /* the accelerator overwrites every line of C itself */
        memset(C, 0, M * NOUT * 4);

    /* Flush the host's caches.  Packing leaves the most recently written
     * weights dirty in L1/L2, and the accelerator's DMA would then be
     * served from the host's cache, not from DRAM.  In steady state a
     * layer's weights were prepared long before it runs, so touch one byte
     * per line of a scratch buffer several times the L2 size to push
     * everything out to DRAM. */
    {
        const uint64_t scratch_bytes = 4u << 20;
        volatile uint8_t *scratch = malloc(scratch_bytes);
        if (!scratch) {
            fprintf(stderr, "out of memory\n");
            return 2;
        }
        for (uint64_t i = 0; i < scratch_bytes; i += 64)
            scratch[i] = (uint8_t)i;
    }
    printf("HOST: %s layer M=%llu K=%llu N_out=%llu, array N=%llu\n",
           sw ? "software" : "accelerator", (unsigned long long)M,
           (unsigned long long)K, (unsigned long long)NOUT,
           (unsigned long long)N);

    /* ------------------------------------------------ layer region */
    uint64_t status = 0, polls = 0;
    m5_dump_reset_stats();
    if (sw) {
        for (uint64_t m = 0; m < M; ++m) {
            int32_t *crow = C + m * NOUT;
            for (uint64_t k = 0; k < K; ++k) {
                const int32_t a = A[m * K + k];
                const int8_t *wrow = W + k * NOUT;
                for (uint64_t n = 0; n < NOUT; ++n)
                    crow[n] += a * wrow[n];
            }
        }
    } else {
        wr(REG_M, M);
        wr(REG_K, K);
        wr(REG_NOUT, NOUT);
        wr(REG_A, shared_pa + a_off);
        wr(REG_W, shared_pa + w_off);
        wr(REG_C, shared_pa + c_off);
        wr(REG_CTRL, 1);
#ifdef SYSTOLIC_HOST_EXT
        if (split_stats)
            m5_dump_reset_stats();
#endif
        do {
            status = rd(REG_STATUS);
            ++polls;
        } while (!(status & STATUS_DONE));
    }
    m5_dump_reset_stats();
    /* ------------------------------------------------ end of layer */

    uint64_t bad = 0;
    for (uint64_t i = 0; i < M * NOUT; ++i) {
        if (C[i] != golden[i] && bad++ < 5)
            printf("HOST: C[%llu][%llu] = %d, expected %d\n",
                   (unsigned long long)(i / NOUT),
                   (unsigned long long)(i % NOUT), C[i], golden[i]);
    }
    if (sw) {
        printf("HOST: result mode=sw mismatches=%llu checked=%llu\n",
               (unsigned long long)bad, (unsigned long long)(M * NOUT));
    } else {
        printf("HOST: result mode=accel mismatches=%llu checked=%llu "
               "layer_cycles=%llu compute_cycles=%llu accel_verify=%llu "
               "status=%llu polls=%llu\n",
               (unsigned long long)bad, (unsigned long long)(M * NOUT),
               (unsigned long long)rd(REG_LAYER_CYCLES),
               (unsigned long long)rd(REG_COMPUTE_CYCLES),
               (unsigned long long)rd(REG_VERIFY),
               (unsigned long long)status, (unsigned long long)polls);
        if (status & STATUS_ERROR)
            bad++;
    }
    printf("HOST: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
