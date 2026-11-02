//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : mac.sv                                                      //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //                                                 
// Group       : #4                                                          //
// Description : MAC (Multiply-Accumulate) unit                              //
//---------------------------------------------------------------------------//
module mac (   
  input  logic       CLK,   // clock
  input  logic       RST_N, // reset (active low)
  input  logic       EN,    // enable
  input  logic [7:0] X_IN,  // input data
  input  logic [7:0] W_IN,  // weight
  input  logic [7:0] Y_IN,  // partial sum
  output logic [7:0] X_OUT, // output data
  output logic [7:0] Y_OUT  // output sum
);
  // multiply-accumulate logic
  always_ff @(posedge CLK, negedge RST_N) begin
    if (!RST_N) begin
      X_OUT <= 0;
      Y_OUT <= 0;
    end else if (EN) begin
      X_OUT <= X_IN;
      Y_OUT <= Y_IN + ($signed(W_IN) * $signed(X_IN));
    end
  end

endmodule : mac
