`timescale 1ns/1ps

module tb_sram_bank;
    localparam integer NUM_WORDS = 4;
    localparam integer WORD_WIDTH = 64;
    localparam integer ADDR_WIDTH = 2;

    logic CLK = 1'b0;
    logic RST_N;
    logic WR_EN;
    logic [ADDR_WIDTH-1:0] WR_ADDR;
    logic [WORD_WIDTH-1:0] WR_DATA;
    logic RD_EN;
    logic [ADDR_WIDTH-1:0] RD_ADDR;
    logic [WORD_WIDTH-1:0] RD_DATA;
    logic RD_VALID;

    always #5 CLK = ~CLK;

    sram_bank #(
        .NUM_WORDS(NUM_WORDS),
        .WORD_WIDTH(WORD_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .CLK(CLK),
        .RST_N(RST_N),
        .WR_EN(WR_EN),
        .WR_ADDR(WR_ADDR),
        .WR_DATA(WR_DATA),
        .RD_EN(RD_EN),
        .RD_ADDR(RD_ADDR),
        .RD_DATA(RD_DATA),
        .RD_VALID(RD_VALID)
    );

    task automatic write_word(
        input logic [ADDR_WIDTH-1:0] addr,
        input logic [WORD_WIDTH-1:0] data
    );
        @(negedge CLK);
        WR_EN = 1'b1;
        WR_ADDR = addr;
        WR_DATA = data;
        @(negedge CLK);
        WR_EN = 1'b0;
    endtask

    task automatic read_and_check(
        input logic [ADDR_WIDTH-1:0] addr,
        input logic [WORD_WIDTH-1:0] expected
    );
        @(negedge CLK);
        RD_EN = 1'b1;
        RD_ADDR = addr;
        @(posedge CLK); #1;
        if (RD_VALID !== 1'b1)
            $fatal(1, "RD_VALID was not asserted for address %0d", addr);
        if (RD_DATA !== expected)
            $fatal(1, "Read address %0d failed: expected 0x%h, got 0x%h", addr, expected, RD_DATA);
        $display("PASS: read address %0d returned 0x%h", addr, RD_DATA);
        @(negedge CLK);
        RD_EN = 1'b0;
        @(posedge CLK); #1;
        if (RD_VALID !== 1'b0)
            $fatal(1, "RD_VALID should deassert when RD_EN is low");
    endtask

    initial begin
        RST_N = 1'b0;
        WR_EN = 1'b0;
        WR_ADDR = '0;
        WR_DATA = '0;
        RD_EN = 1'b0;
        RD_ADDR = '0;

        repeat (2) @(negedge CLK);
        RST_N = 1'b1;

        write_word(2'd0, 64'h0123_4567_89AB_CDEF);
        write_word(2'd1, 64'h1111_2222_3333_4444);
        write_word(2'd3, 64'hDEAD_BEEF_CAFE_F00D);

        read_and_check(2'd0, 64'h0123_4567_89AB_CDEF);
        read_and_check(2'd1, 64'h1111_2222_3333_4444);
        read_and_check(2'd3, 64'hDEAD_BEEF_CAFE_F00D);

        $display("tb_sram_bank: ALL TESTS PASSED");
        $finish;
    end
endmodule
