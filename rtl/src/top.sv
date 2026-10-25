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
  parameter BANKS = 8,  // bank count (number of SRAM banks)
  parameter integer ADDR_WIDTH = (N > 1) ? $clog2(N) : 1
)(   
  input  logic                   CLK,      // clock
  input  logic                   RST_N,    // reset (active low)
  input  logic                   EN,       // enable
  input  logic [    N-1:0][63:0] ARRAY_IN, // array inputs (N * 8 bytes)
  output logic [BANKS-1:0][63:0] ARRAY_OUT, // array outputs (BANKS * 8 bytes)

  // SRAM write interface for each bank
  input  logic [BANKS-1:0] WEIGHT_WR_EN,
  input  logic [BANKS-1:0][ADDR_WIDTH-1:0] WEIGHT_WR_ADDR,
  input  logic [BANKS-1:0][63:0] WEIGHT_WR_DATA,

  // measurement counters
  output logic [31:0]            stall_count,
  output logic [31:0]            active_count

);

  // internal signals

  logic [ADDR_WIDTH-1:0] sram_addr;

  logic [BANKS-1:0][63:0]        weights;
  logic [BANKS-1:0]       weight_valid;

  logic all_weights_valid;
  logic stall;
  logic mac_en;
  logic sram_en;

  // N MAC stages need N+1 to include initial 0
  logic [BANKS-1:0][N:0][63:0] partial_sums; //renamed from array_wires to be more clear

  //-------------------------------------------------------------------------//
  // controller
  //-------------------------------------------------------------------------//

  assign sram_en = EN; 
  assign all_weights_valid = &weight_valid;

  assign stall  = EN && !all_weights_valid;
  assign mac_en = EN && all_weights_valid;

  // address advances for each SRAM read
  always_ff @(posedge CLK or negedge RST_N) begin
    if (!RST_N) begin
      sram_addr <= '0;
    end else if (sram_en) begin
      if (sram_addr == N-1)
        sram_addr <= '0;
      else
        sram_addr <= sram_addr + 1'b1;
    end
  end


  always_ff @(posedge CLK or negedge RST_N) begin
    if (!RST_N) begin
      stall_count  <= '0;
      active_count <= '0;
    end else if (EN) begin
      if (stall)
        stall_count <= stall_count + 1'b1; //count for stalled cycles
      else
        active_count <= active_count + 1'b1; //count for active cycles
    end
  end

  //-------------------------------------------------------------------------//
  // generate BANKS number of SRAM banks, N * BANKS number of MAC blocks
  //-------------------------------------------------------------------------//
  generate
  for (genvar i = 0; i < BANKS; i++) begin : gen_banks
    assign partial_sums[i][0] = 64'b0;

    sram_bank #(
      .NUM_WORDS(N),
      .WORD_WIDTH(64),
      .ADDR_WIDTH(ADDR_WIDTH)
    ) sram_bank_u (
      .CLK(CLK),
      .RST_N(RST_N),

      .WR_EN(WEIGHT_WR_EN[i]),
      .WR_ADDR(WEIGHT_WR_ADDR[i]),
      .WR_DATA(WEIGHT_WR_DATA[i]),

      .RD_EN(sram_en),
      .RD_ADDR(sram_addr),
      .RD_DATA(weights[i]),
      .RD_VALID(weight_valid[i])
    );

        for (genvar j = 0; j < N; j++) begin : gen_macs
          mac #(
            .NUM_BYTES (8)      // port width (bytes)
          ) mac_u (   
            .CLK   (CLK),       // clock
            .RST_N (RST_N),     // reset (active low)
            .EN    (mac_en), // enable
            .X_IN  (ARRAY_IN[j]), // input data
            .W_IN  (weights[i]), // weight
            .Y_IN  (partial_sums[i][j]), // partial sum
            .Y_OUT (partial_sums[i][j+1])  // output data
          );
        end

    // assign array output to last row
        assign ARRAY_OUT[i] = partial_sums[i][N];
    end
  endgenerate

endmodule : top
