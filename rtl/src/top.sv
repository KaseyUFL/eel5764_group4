//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : top.sv                                                      //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //                                                 //
// Group       : #4                                                          //
// Description : Top-Level for the Parameterized SRAM Scratchpad Array       //
//---------------------------------------------------------------------------//
module top #(
  parameter N     = 8,          // port width
  parameter BANKS = 8           // bank count (number of SRAM blocks)
)(   
  input  logic       CLK,       // clock
  input  logic       RST_N,     // reset (active low)
  // add other signals below

);
  /* TODO:
  - decide on top-level i/o
  - investigate controls / stall counter
  - discuss stall / pipeline registers
  - wire up modules with generate loops
  */

  //-------------------------------------------------------------------------//
  // generate BANKS number of SRAM banks, N * BANKS number of MAC blocks
  //-------------------------------------------------------------------------//
  for (genvar i; i < BANKS; i++) begin 
    sram_bank #(
      .NUM_WORDS (N),    // address width
      .NUM_BYTES (8)     // data width (bytes)  
    ) sram_bank_u (   
      .CLK  (CLK),       // clock
      .EN   (/* TBD */), // enable
      .ADDR (/* TBD */), // read address
      .DATA (/* TBD */)  // read data
    );

    for (genvar j; j < N; j++) begin
      mac #(
        .NUM_BYTES (8)      // port width (bytes)
      ) mac_u (   
        .CLK   (CLK),       // clock
        .RST_N (RST_N),     // reset (active low)
        .EN    (/* TBD */), // enable
        .X_IN  (/* TBD */), // input data
        .W_IN  (/* TBD */), // weight
        .Y_IN  (/* TBD */), // partial sum
        .Y_OUT (/* TBD */)  // output data
      );
    end
  end

endmodule : top
