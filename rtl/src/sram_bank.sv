//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : sram_bank.sv                                                //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //                                                 
// Group       : #4                                                          //
// Description : SRAM bank                                                   //
//---------------------------------------------------------------------------//
module sram_bank #(
  parameter NUM_WORDS = 8,  // address width
  parameter NUM_BYTES = 64, // data width (bytes)  
)(   
  input  logic                   CLK,  // clock
  input  logic                   EN,   // enable
  input  logic [  NUM_WORDS-1:0] ADDR, // read address
  output logic [8*NUM_BYTES-1:0] DATA  // read data
);
  // memory array
  (* ram_style = block *) logic [8*NUM_BYTES-1:0] memory [NUM_WORDS-1:0];

  // read channel
  always_ff @(posedge CLK) begin
    if (EN) begin
      DATA <= memory[ADDR];
    end
  end

endmodule : sram_bank
