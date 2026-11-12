`timescale 1ns/1ps

// Loads random signed weights and input vectors into SRAM, runs the array,
// and checks every column of every result against a reference model.
module tb_top;
    parameter integer N = 4;
    parameter integer BANKS = 4;
    parameter integer NUM_INPUTS = 6;
    localparam integer WPV = (N + 7) / 8;  // 64-bit words per row/vector
    localparam integer NUM_WORDS = ((N + NUM_INPUTS) * WPV + BANKS - 1) / BANKS;
    localparam integer ADDR_WIDTH = (NUM_WORDS > 1) ? $clog2(NUM_WORDS) : 1;
    localparam integer ACTIVE_CYCLES = NUM_INPUTS + 2*N - 1;
    // each vector takes R cycles to read; the array waits R-1 of them,
    // plus one more for the read latency of the first vector
    localparam integer R = (WPV + BANKS - 1) / BANKS;
    localparam integer STALL_CYCLES = R + (NUM_INPUTS - 1) * (R - 1);

    logic CLK = 1'b0;
    logic RST_N;
    logic EN;
    logic [N-1:0][7:0] ARRAY_OUT;
    logic [BANKS-1:0] WEIGHT_WR_EN;
    logic [BANKS-1:0][ADDR_WIDTH-1:0] WEIGHT_WR_ADDR;
    logic [BANKS-1:0][63:0] WEIGHT_WR_DATA;
    logic [31:0] stall_count;
    logic [31:0] active_count;

    logic [7:0] w [N][N];                 // w[j][i]: row j, column i
    logic [7:0] x [NUM_INPUTS][N];        // x[t][j]: vector t, row j
    logic [7:0] expected [NUM_INPUTS][N]; // expected[t][i]: vector t, column i
    bit         seen [NUM_INPUTS][N];
    int         errors = 0;

    always #5 CLK = ~CLK;

    top #(
        .N(N),
        .BANKS(BANKS),
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_WORDS(NUM_WORDS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .CLK(CLK),
        .RST_N(RST_N),
        .EN(EN),
        .ARRAY_OUT(ARRAY_OUT),
        .WEIGHT_WR_EN(WEIGHT_WR_EN),
        .WEIGHT_WR_ADDR(WEIGHT_WR_ADDR),
        .WEIGHT_WR_DATA(WEIGHT_WR_DATA),
        .stall_count(stall_count),
        .active_count(active_count)
    );

    // SRAM word k lives in bank k % BANKS at address k / BANKS
    task automatic write_word(input int k, input logic [63:0] data);
        @(negedge CLK);
        WEIGHT_WR_EN = '0;
        WEIGHT_WR_EN[k % BANKS] = 1'b1;
        WEIGHT_WR_ADDR[k % BANKS] = ADDR_WIDTH'(k / BANKS);
        WEIGHT_WR_DATA[k % BANKS] = data;
    endtask

    // Column i shows vector t = active_count - 1 - N - i: each vector takes
    // N cycles to reach the bottom row and i more to reach column i.
    always @(posedge CLK) begin
        #1;
        if (RST_N) begin
            for (int i = 0; i < N; i++) begin
                automatic int t = int'(active_count) - 1 - N - i;
                if (t >= 0 && t < NUM_INPUTS) begin
                    seen[t][i] = 1'b1;
                    if (ARRAY_OUT[i] !== expected[t][i]) begin
                        errors++;
                        $display("ERROR: vector %0d column %0d: expected %0d, got %0d",
                                 t, i, $signed(expected[t][i]), $signed(ARRAY_OUT[i]));
                    end
                end
            end
        end
    end

    initial begin
        logic [63:0] word;
        logic [7:0]  acc;

        RST_N = 1'b0;
        EN = 1'b0;
        WEIGHT_WR_EN = '0;
        WEIGHT_WR_ADDR = '0;
        WEIGHT_WR_DATA = '0;

        // random signed INT8 data; results wrap at 8 bits like the MACs
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                w[j][i] = 8'($urandom);
        for (int t = 0; t < NUM_INPUTS; t++)
            for (int j = 0; j < N; j++)
                x[t][j] = 8'($urandom);
        for (int t = 0; t < NUM_INPUTS; t++)
            for (int i = 0; i < N; i++) begin
                acc = '0;
                for (int j = 0; j < N; j++)
                    acc += 8'($signed(w[j][i]) * $signed(x[t][j]));
                expected[t][i] = acc;
            end

        repeat (2) @(negedge CLK);
        RST_N = 1'b1;

        // weight rows first, then the input vectors, WPV words each;
        // chunk c holds bytes 8c .. 8c+7
        for (int j = 0; j < N; j++)
            for (int c = 0; c < WPV; c++) begin
                word = '0;
                for (int b = 0; b < 8 && 8*c + b < N; b++)
                    word[8*b +: 8] = w[j][8*c + b];
                write_word(j*WPV + c, word);
            end
        for (int t = 0; t < NUM_INPUTS; t++)
            for (int c = 0; c < WPV; c++) begin
                word = '0;
                for (int b = 0; b < 8 && 8*c + b < N; b++)
                    word[8*b +: 8] = x[t][8*c + b];
                write_word((N + t)*WPV + c, word);
            end
        @(negedge CLK);
        WEIGHT_WR_EN = '0;

        EN = 1'b1;
        for (int c = 0; c < 1000 && !dut.controller_u.DONE; c++)
            @(negedge CLK);
        @(negedge CLK);

        if (!dut.controller_u.DONE)
            $fatal(1, "controller never reached DONE");
        if ((^stall_count) === 1'bx || (^active_count) === 1'bx)
            $fatal(1, "Top-level counters contain unknown values");
        if (active_count != ACTIVE_CYCLES)
            $fatal(1, "active_count=%0d, expected %0d", active_count, ACTIVE_CYCLES);
        if (stall_count != STALL_CYCLES)
            $fatal(1, "stall_count=%0d, expected %0d", stall_count, STALL_CYCLES);
        for (int t = 0; t < NUM_INPUTS; t++)
            for (int i = 0; i < N; i++)
                if (!seen[t][i])
                    $fatal(1, "vector %0d column %0d was never checked", t, i);
        if (errors != 0)
            $fatal(1, "%0d result mismatches", errors);

        $display("PASS: N=%0d BANKS=%0d, %0d vectors x %0d columns match the reference",
                 N, BANKS, NUM_INPUTS, N);
        $display("stall_count=%0d active_count=%0d utilization=%.2f",
                 stall_count, active_count,
                 real'(active_count) / real'(active_count + stall_count));
        $finish;
    end
endmodule
