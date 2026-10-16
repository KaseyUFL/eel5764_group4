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
  parameter N     = 8, // input depth (number of MACs per SRAM bank)
  parameter BANKS = 8  // bank count (number of SRAM banks)
)(   
  input  logic                   CLK,      // clock
  input  logic                   RST_N,    // reset (active low)
  input  logic                   EN,       // enable
  input  logic [    N-1:0][63:0] ARRAY_IN, // array inputs (N * 8 bytes)
  output logic [BANKS-1:0][63:0] ARRAY_OUT // array outputs (BANKS * 8 bytes)
);
  /* TODO:
  - investigate controls / stall counter
  - discuss stall / pipeline registers, RAM cycle delay
  - wire up modules with generate loops
  */
  // internal signals
  logic [BANKS-1:0][63:0]        weights;
  logic [BANKS-1:0][N-1:0][63:0] array_wires;

  //-------------------------------------------------------------------------//
  // TBD controller
  //-------------------------------------------------------------------------//


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
      .DATA (weights[i])  // read data
    );

    // assign first row Y_IN to zero
    assign array_wires[i][0] = 64'h0;

    for (genvar j; j < N; j++) begin
      mac #(
        .NUM_BYTES (8)      // port width (bytes)
      ) mac_u (   
        .CLK   (CLK),       // clock
        .RST_N (RST_N),     // reset (active low)
        .EN    (/* TBD */), // enable
        .X_IN  (ARRAY_IN[j]), // input data
        .W_IN  (weights[i]), // weight
        .Y_IN  (array_wires[i][j]), // partial sum
        .Y_OUT (/* TBD */)  // output data
      );
    end

    // assign array output to last row
    assign ARRAY_OUT[i] = array_wires[i][N-1];
  end

endmodule : top
