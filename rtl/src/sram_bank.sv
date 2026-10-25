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
  parameter WORD_WIDTH = 64, // data width (bytes) 
  parameter integer ADDR_WIDTH = (NUM_WORDS > 1) ? $clog2(NUM_WORDS) : 1 
)(   
  input  logic                   CLK,  // clock
  input  logic                   RST_N,
  input  logic                   WR_EN,   // enable
  input  logic [ADDR_WIDTH-1:0] WR_ADDR,
  input  logic [WORD_WIDTH-1:0] WR_DATA,
  
  input  logic                  RD_EN,
  input  logic [ADDR_WIDTH-1:0] RD_ADDR,
  output logic [WORD_WIDTH-1:0] RD_DATA,
  output logic                  RD_VALID
);
  logic [WORD_WIDTH-1:0] memory [0:NUM_WORDS-1];

  // write channel
  always_ff @(posedge CLK) begin
        if (WR_EN)
            memory[WR_ADDR] <= WR_DATA;
    end

  // read channel
  always_ff @(posedge CLK or negedge RST_N) begin
    if (!RST_N) begin
      RD_DATA  <= '0;
      RD_VALID <= 1'b0;
    end else begin
      RD_VALID <= RD_EN;
      if (RD_EN) begin
        RD_DATA <= memory[RD_ADDR];
      end
    end
  end

endmodule : sram_bank
