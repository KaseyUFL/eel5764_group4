//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : tb_mac_adv.sv                                               //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //                                                 //
// Group       : #4                                                          //
// Description : Advanced MAC Testbench                                      //
//---------------------------------------------------------------------------//
`timescale 1ns/1ps
//---------------------------------------------------------------------------//
// add_item class: randomizes x_in, w_in, and y_in
//---------------------------------------------------------------------------//
class add_item;
  rand bit [7:0] x_in;
  rand bit [7:0] w_in;
  rand bit [7:0] y_in; 
endclass

//---------------------------------------------------------------------------//
// testbench module
//---------------------------------------------------------------------------//
module tb_mac_adv;
  //-------------------------------------------------------------------------//
  // declare DUT I/O
  //-------------------------------------------------------------------------//
  logic              CLK = 1'b0;  // clock
  logic              RST_N;       // reset (active low)     
  logic              EN;          // enable  
  logic        [7:0] X_IN;        // input data
  logic        [7:0] W_IN;        // weight
  logic        [7:0] Y_IN;        // partial sum
  logic signed [7:0] Y_OUT;       // output data

  //-------------------------------------------------------------------------//
  // clock generator
  //-------------------------------------------------------------------------//
  initial begin : generate_clock
    forever #5 CLK <= ~CLK;
  end

  //-------------------------------------------------------------------------//
  // instantiate DUT
  //-------------------------------------------------------------------------//  
  mac dut (
    .CLK     (CLK),
    .RST_N   (RST_N),
    .EN      (EN),
    .X_IN    (X_IN),
    .W_IN    (W_IN),
    .Y_IN    (Y_IN),
    .Y_OUT   (Y_OUT)
  );

  //-------------------------------------------------------------------------//
  // check_output task: checks expected value against actual
  //-------------------------------------------------------------------------//   
  task automatic check_output (
    input logic signed [7:0] expected,
    input string test_name
  );
    if (Y_OUT !== expected) begin
      $fatal(1, "%s failed: expected %d, got %d", test_name, expected, Y_OUT);
    end else begin
      $display("%s passed: expected %d, got %d", test_name, expected, Y_OUT);
    end
  endtask

  //-------------------------------------------------------------------------//
  // create instance of add_item class
  //-------------------------------------------------------------------------//  
  add_item item;

  //-------------------------------------------------------------------------//
  // test procedure
  //-------------------------------------------------------------------------//    
  initial begin
    // create add_item item
    item  = new;

    // initialize inputs
    RST_N = 1'b0;
    EN    = 1'b0;
    X_IN  = '0;
    W_IN  = '0;
    Y_IN  = '0;

    // negate the reset
    repeat (2) @(negedge CLK);
    RST_N = 1'b1;

    // iterate the inputs
    for (int i = 0; i < 256; i++) begin
      // check for randomization failure
      assert (item.randomize())
      else $fatal(1, "ERROR: Randomization failed.");  

      // assign inputs
      @(negedge CLK);
      EN   <= 1'b1;
      X_IN <= item.x_in;
      W_IN <= item.w_in;
      Y_IN <= item.y_in;
      @(posedge CLK); #1;

      // signed arithmetic check: x_in * w_in + y_in
      check_output($signed(X_IN) * $signed(W_IN) + Y_IN, "signed multiply-accumulate");
    end

    // end simulation
    $display("tb_mac_adv: ALL TESTS PASSED!");
    disable generate_clock;
  end
endmodule : tb_mac_adv
