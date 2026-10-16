//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : mac.sv                                                      //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //                                                 
// Group       : #4                                                          //
// Description : MAC (Multiply-Accumulate) unit                              //
//---------------------------------------------------------------------------//
module mac #(
  parameter NUM_BYTES = 8 // port width (bytes)
)(   
  input  logic                   CLK,   // clock
  input  logic                   RST_N, // reset (active low)
  input  logic                   EN,    // enable
  input  logic [8*NUM_BYTES-1:0] X_IN,  // input data
  input  logic [8*NUM_BYTES-1:0] W_IN,  // weight
  input  logic [8*NUM_BYTES-1:0] Y_IN,  // partial sum
  output logic [8*NUM_BYTES-1:0] Y_OUT  // output data
);
  // multiply-accumulate logic
  always_ff @(posedge CLK, negedge RST_N) begin
    if (!RST_N) begin
      Y_OUT <= 0;
    end else if (EN) begin
      Y_OUT <= Y_IN + (W_IN * X_IN);
    end
  end

endmodule : mac
