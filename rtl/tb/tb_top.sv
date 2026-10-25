`timescale 1ns/1ps

module tb_top;
    localparam integer N = 4;
    localparam integer BANKS = 4;
    localparam integer ADDR_WIDTH = 2;

    logic CLK = 1'b0;
    logic RST_N;
    logic EN;
    logic [N-1:0][63:0] ARRAY_IN;
    logic [BANKS-1:0][63:0] ARRAY_OUT;
    logic [BANKS-1:0] WEIGHT_WR_EN;
    logic [BANKS-1:0][ADDR_WIDTH-1:0] WEIGHT_WR_ADDR;
    logic [BANKS-1:0][63:0] WEIGHT_WR_DATA;
    logic [31:0] stall_count;
    logic [31:0] active_count;

    always #5 CLK = ~CLK;

    top #(
        .N(N),
        .BANKS(BANKS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .CLK(CLK),
        .RST_N(RST_N),
        .EN(EN),
        .ARRAY_IN(ARRAY_IN),
        .ARRAY_OUT(ARRAY_OUT),
        .WEIGHT_WR_EN(WEIGHT_WR_EN),
        .WEIGHT_WR_ADDR(WEIGHT_WR_ADDR),
        .WEIGHT_WR_DATA(WEIGHT_WR_DATA),
        .stall_count(stall_count),
        .active_count(active_count)
    );

    initial begin
        RST_N = 1'b0;
        EN = 1'b0;
        ARRAY_IN = '0;
        WEIGHT_WR_EN = '0;
        WEIGHT_WR_ADDR = '0;
        WEIGHT_WR_DATA = '0;

        repeat (2) @(negedge CLK);
        RST_N = 1'b1;

        // Load every address of every SRAM bank while computation is disabled.
        for (int bank = 0; bank < BANKS; bank++) begin
            for (int addr = 0; addr < N; addr++) begin
                @(negedge CLK);
                WEIGHT_WR_EN = '0;
                WEIGHT_WR_EN[bank] = 1'b1;
                WEIGHT_WR_ADDR[bank] = addr[ADDR_WIDTH-1:0];
                WEIGHT_WR_DATA[bank] = 64'(bank + addr + 1);
            end
        end
        @(negedge CLK);
        WEIGHT_WR_EN = '0;

        // Drive simple nonzero input words and run for several cycles.
        for (int k = 0; k < N; k++)
            ARRAY_IN[k] = 64'(k + 1);

        EN = 1'b1;
        repeat (12) @(negedge CLK);
        EN = 1'b0;
        @(posedge CLK); #1;

        if ((^stall_count) === 1'bx || (^active_count) === 1'bx)
            $fatal(1, "Top-level counters contain unknown values");
        if (active_count == 0)
            $fatal(1, "active_count never incremented; check SRAM valid/stall timing");
        if ((^ARRAY_OUT) === 1'bx)
            $fatal(1, "ARRAY_OUT contains unknown values after the test");

        $display("PASS: top-level smoke test");
        $display("stall_count=%0d active_count=%0d", stall_count, active_count);
        $display("NOTE: this is a smoke test, not a full systolic-array correctness test.");
        $finish;
    end
endmodule
