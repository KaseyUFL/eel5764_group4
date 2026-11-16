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
  input  logic signed [7:0] X_IN,  // input data

  //weight loading controls
  input  logic signed [7:0] W_IN,  // weight
  input logic W_LOAD,
  input logic W_SWAP,

  input  logic signed [31:0] Y_IN,  // partial sum
  output logic signed [7:0] X_OUT, // output data
  output logic signed [31:0] Y_OUT  // output sum
);

  //double-buffered weight registers
  logic signed [7:0] active_weight;
  logic signed [7:0] staging_weight;

  always_ff @(posedge CLK or negedge RST_N) begin
    if (!RST_N) begin
      X_OUT          <= 0;
      Y_OUT          <= 0;
      active_weight  <= 0;
      staging_weight <= 0;
    end else begin

      // load next weight into staging register
      if (W_LOAD) begin
        staging_weight <= W_IN; // weight loaded from SRAM
      end

      // swap staging and active weights
      if (W_SWAP) begin
        active_weight <= staging_weight;
      end

      // perform multiply-accumulate
      else if (EN) begin
        X_OUT <= X_IN;
        Y_OUT <= Y_IN + ($signed(active_weight) * $signed(X_IN));
      end

    end
  end


endmodule : mac
