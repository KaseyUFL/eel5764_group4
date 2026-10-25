
# Parameterized SRAM Scratchpad Array — RTL Status

## 1. Module Overview

- **`mac.sv`** — Implements a clocked multiply-accumulate (MAC) stage. It resets its output and updates the result when enabled.
- **`sram_bank.sv`** — Implements a single-memory SRAM-style bank with write and synchronous read interfaces. `RD_VALID` indicates a read response.
- **`top.sv`** — Connects the SRAM banks, MAC instances, partial-sum chains, control signals, and stall/active-cycle counters.

## 2. Testbenches

- **`tb_mac.sv`** — Tests MAC reset behavior, enabled operation, and output hold behavior when disabled.
- **`tb_sram_bank.sv`** — Tests SRAM writes, synchronous reads, and read-valid behavior.
- **`tb_top.sv`** — Performs a top-level smoke test, checking basic integration and selected output/counter behavior. It is **not** a full systolic-array correctness test.

## 3. What Simulation Has Proven So Far

All three testbenches passed their current assertions in simulation. This provides initial evidence that:

- The MAC behaves as expected for the operations covered by its testbench.
- The SRAM bank's basic write/read behavior and valid signaling work for the tested cases.
- The top module can be simulated under the smoke-test conditions, and its limited checks pass.

These results apply only to the test cases currently implemented. They do **not** prove that the complete array calculates correct matrix multiplication or dot products, that stalls preserve data alignment, or that the design meets synthesis and timing requirements.

## 4. Remaining Work

1. **Full systolic-array correctness test:** Create a testbench that loads known weights and activations, runs a complete computation, and compares every output against a software/reference calculation. Start with small, hand-checkable inputs and then add randomized tests.

2. **Confirm array dataflow:** Define how activations, weights, and partial sums move through the MACs and how `N` and `BANKS` map to the compute dimensions. The current generated MAC connections form per-bank linear chains; they do not, by themselves, implement a conventional two-dimensional systolic array.

3. **Verify stall and SRAM timing:** Align synchronous SRAM read responses with the correct weights and activations. Check that addresses, valid signals, MAC state, and partial sums advance or hold correctly during stalls.

4. **Verify arithmetic:** Confirm signed INT8 multiplication and the intended accumulation width. If each 64-bit word contains eight INT8 values, verify that the MAC performs eight independent lane multiplications rather than one packed 64-bit multiplication.

5. **Expand corner-case tests:** Test reset, disabled cycles, consecutive reads and writes, address wraparound, stalls, and multiple legal values of `N` and `BANKS`.

6. **Run synthesis:** Check that the SRAMs infer the expected memory resources, inspect MAC and logic utilization, and record timing estimates for the selected FPGA target. Passing RTL simulation does not guarantee successful synthesis or timing closure.

## 5. Current Status

**Initial module-level tests pass, and the top-level smoke test passes. Full systolic-array correctness, arithmetic validation, and stall-safe dataflow remain to be verified.**
