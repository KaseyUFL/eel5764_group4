`timescale 1ns/1ps

module tb_mac;

    logic CLK = 1'b0;
    logic RST_N;
    logic EN;
    logic [7:0] X_IN;
    logic [7:0] W_IN;
    logic [7:0] Y_IN;
    logic [7:0] Y_OUT;

    always #5 CLK = ~CLK;

    mac dut (
        .CLK(CLK),
        .RST_N(RST_N),
        .EN(EN),
        .X_IN(X_IN),
        .W_IN(W_IN),
        .Y_IN(Y_IN),
        .X_OUT(),
        .Y_OUT(Y_OUT)
    );

    task automatic check_output(
        input logic [7:0] expected,
        input string test_name
    );
        if (Y_OUT !== expected)
            $fatal(1, "%s failed: expected 0x%h, got 0x%h", test_name, expected, Y_OUT);
        else
            $display("PASS: %s", test_name);
    endtask

    initial begin
        RST_N = 1'b0;
        EN = 1'b0;
        X_IN = '0;
        W_IN = '0;
        Y_IN = '0;

        repeat (2) @(negedge CLK);
        check_output('0, "asynchronous reset");
        RST_N = 1'b1;

        // Basic arithmetic check: 2 * 3 + 4 = 10.
        @(negedge CLK);
        EN = 1'b1;
        X_IN = 8'd2;
        W_IN = 8'd3;
        Y_IN = 8'd4;
        @(posedge CLK); #1;
        check_output(8'd10, "basic multiply-accumulate");

        // Check that EN=0 holds the registered output.
        @(negedge CLK);
        EN = 1'b0;
        X_IN = 8'd100;
        W_IN = 8'd100;
        Y_IN = 8'd100;
        @(posedge CLK); #1;
        check_output(8'd10, "output holds when disabled");

        // Check another enabled operation with the same 64-bit arithmetic.
        @(negedge CLK);
        EN = 1'b1;
        X_IN = 8'd5;
        W_IN = 8'd6;
        Y_IN = 8'd7;
        @(posedge CLK); #1;
        check_output(8'd37, "second multiply-accumulate");

        $display("tb_mac: ALL TESTS PASSED");
        $finish;
    end
endmodule
