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
  parameter N     = 8, // input depth (number of MACs squared)
  parameter BANKS = 8,  // bank count (number of SRAM banks)
  parameter NUM_INPUTS = N, // input vectors stored after the weights
  // words per bank: by default just enough for the weights and inputs
  parameter NUM_WORDS = ((N + NUM_INPUTS) * ((N + 7) / 8) + BANKS - 1) / BANKS,
  parameter integer ADDR_WIDTH = (NUM_WORDS > 1) ? $clog2(NUM_WORDS) : 1
)(   
  input  logic                   CLK,      // clock
  input  logic                   RST_N,    // reset (active low)
  input  logic                   EN,       // enable
  output logic [N-1:0][7:0] ARRAY_OUT, // array outputs (BANKS * 8 bytes)

  // SRAM write interface for each bank
  input  logic [BANKS-1:0] WEIGHT_WR_EN,
  input  logic [BANKS-1:0][ADDR_WIDTH-1:0] WEIGHT_WR_ADDR,
  input  logic [BANKS-1:0][63:0] WEIGHT_WR_DATA,

  // measurement counters
  output logic [31:0]            stall_count,
  output logic [31:0]            active_count

);

  localparam integer WPV         = (N + 7) / 8; // 64-bit words per row/vector
  localparam integer CHUNK_WIDTH = (WPV > 1) ? $clog2(WPV) : 1;
  localparam integer STEP_WIDTH  = $clog2(N + NUM_INPUTS + 1);

  // internal signals

  logic [BANKS-1:0][ADDR_WIDTH-1:0] sram_addr;

  logic [BANKS-1:0][63:0]  rd_data;   // SRAM read data
  logic [BANKS-1:0]        rd_valid;  // SRAM read valid

  logic stall;
  logic mac_en;
  logic [BANKS-1:0] sram_en;

  // controller outputs: returned words and their chunk of the row/vector
  logic                               ret_valid;
  logic                               ret_weight;
  logic [STEP_WIDTH-1:0]              ret_row;
  logic [BANKS-1:0]                   ret_en;
  logic [BANKS-1:0][CHUNK_WIDTH-1:0]  ret_chunk;
  logic [WPV-1:0][63:0]               ret_word;  // returned words, by chunk
  logic [WPV-1:0]                     ret_mask;
  logic                               x_valid;   // x_vec is a complete vector
  logic [64*WPV-1:0]                  x_vec;     // input vector being assembled
  logic [N-1:0][7:0]                  x_in;      // input vector to the input buffer

  // new stuff
  logic [N-1:0][7:0]         input_pipe;    // N * 8-bit input pipeline
  logic [N-1:0][64*WPV-1:0]  weights;       // N * 8-byte weight array   
  logic [N-1:0][N-1:0][7:0]  partial_sums;  // N * N * 8-bit partial sum array 
  logic [N-1:0][N-1:0][7:0]  x_pipeline;    // N * N * 8-bit x array    

  //-------------------------------------------------------------------------//
  // controller
  //-------------------------------------------------------------------------//
  controller #(
    .N          (N),
    .BANKS      (BANKS),
    .NUM_INPUTS (NUM_INPUTS),
    .NUM_WORDS  (NUM_WORDS),
    .ADDR_WIDTH (ADDR_WIDTH)
  ) controller_u (
    .CLK           (CLK),
    .RST_N         (RST_N),
    .EN            (EN),
    .SRAM_RD_EN    (sram_en),
    .SRAM_RD_ADDR  (sram_addr),
    .SRAM_RD_VALID (rd_valid),
    .RET_VALID     (ret_valid),
    .RET_WEIGHT    (ret_weight),
    .RET_ROW       (ret_row),
    .RET_EN        (ret_en),
    .RET_CHUNK     (ret_chunk),
    .X_VALID       (x_valid),
    .ARRAY_EN      (mac_en),
    .STALL         (stall),
    .DONE          (),
    .stall_count   (stall_count),
    .active_count  (active_count)
  );

  //-------------------------------------------------------------------------//
  // sort returned words by chunk of the row/vector
  //-------------------------------------------------------------------------//
  always_comb begin
    ret_word = '0;
    ret_mask = '0;
    for (int c = 0; c < WPV; c++)
      for (int b = 0; b < BANKS; b++)
        if (ret_en[b] && ret_chunk[b] == c) begin
          ret_word[c] = ret_word[c] | rd_data[b];
          ret_mask[c] = 1'b1;
        end
  end

  //-------------------------------------------------------------------------//
  // weight registers and input vector, filled one chunk (64 bits) at a time
  //-------------------------------------------------------------------------//
  for (genvar c = 0; c < WPV; c++) begin : gen_chunks
    for (genvar j = 0; j < N; j++) begin : gen_weights
      always_ff @(posedge CLK or negedge RST_N) begin
        if (!RST_N)
          weights[j][64*c +: 64] <= '0;
        else if (ret_valid && ret_weight && ret_row == j && ret_mask[c])
          weights[j][64*c +: 64] <= ret_word[c];
      end
    end

    always_ff @(posedge CLK or negedge RST_N) begin
      if (!RST_N)
        x_vec[64*c +: 64] <= '0;
      else if (ret_valid && !ret_weight && ret_mask[c])
        x_vec[64*c +: 64] <= ret_word[c];
    end
  end

  // zeros while draining the array
  assign x_in = x_valid ? x_vec[8*N-1:0] : '0;

  //-------------------------------------------------------------------------//
  // buffer inputs, row number * 1 clock periods
  //-------------------------------------------------------------------------//
  input_buffer #(
    .N     (N)
  ) input_buffer_u (
    .CLK   (CLK),   
    .RST_N (RST_N), 
    .EN    (mac_en),
    .X_IN  (x_in),
    .Y_OUT (input_pipe) 
  );

  //-------------------------------------------------------------------------//
  // TODO controller: (current SRAM control needs replacement)
  // - needs to load weights register after reset
  // - needs to load activation weights and stall if necessary
  //-------------------------------------------------------------------------//


  //-------------------------------------------------------------------------//
  // generate N * N MAC units
  //-------------------------------------------------------------------------//
  // j is the y-coordinate, this loop creates N rows of MACs  
  for (genvar j = 0; j < N; j++) begin 
    //-----------------------------------------------------------------------//
    // MACs at (0,j):
    // - EN    : mac_en
    // - X_IN  : input_pipe[j]
    // - W_IN  : weights[j][7:0]
    // - Y_IN  : if j = 0, 0x0, else, partial_sums[0][j-1]
    // - X_OUT : x_pipeline[0][j]
    // - Y_OUT : partial_sums[0][j]   
    //-----------------------------------------------------------------------//
    if (j == 0) begin 
      // MAC at (0,0)       
      mac mac_u (   
        .CLK   (CLK),            
        .RST_N (RST_N),               
        .EN    (mac_en),
        .X_IN  (input_pipe[j]),        
        .W_IN  (weights[j][7:0]),
        .Y_IN  (8'h0),
        .X_OUT (x_pipeline[0][j]),
        .Y_OUT (partial_sums[0][j]) 
      );
    end else begin
      // MAC at (0,j), where j > 0 
      mac mac_u (   
        .CLK   (CLK),            
        .RST_N (RST_N),               
        .EN    (mac_en),
        .X_IN  (input_pipe[j]),        
        .W_IN  (weights[j][7:0]),
        .Y_IN  (partial_sums[0][j-1]), 
        .X_OUT (x_pipeline[0][j]),
        .Y_OUT (partial_sums[0][j]) 
      );
    end
    //-----------------------------------------------------------------------//
    // i is the x-coordinate, this loop creates N-1 columns of MACs  
    for (genvar i = 1; i < N; i++) begin
      //---------------------------------------------------------------------//
      // MACs at (i,j), where i > 0:
      // - EN    : mac_en
      // - X_IN  : x_pipeline[i-1][j]
      // - W_IN  : weights[j][8*(i+1)-1:i*8]
      // - Y_IN  : if j = 0, 0x0, else, partial_sums[i][j-1]
      // - X_OUT : x_pipeline[i][j]
      // - Y_OUT : partial_sums[i][j]   
      //---------------------------------------------------------------------//      
      if (j == 0) begin
        // MACs at (i,0), where i > 0:
        mac mac_u (   
          .CLK   (CLK),                 
          .RST_N (RST_N),               
          .EN    (mac_en),
          .X_IN  (x_pipeline[i-1][j]),
          .W_IN  (weights[j][8*(i+1)-1:i*8]),          
          .Y_IN  (8'h0),
          .X_OUT (x_pipeline[i][j]),
          .Y_OUT (partial_sums[i][j]) 
        );
      end else begin
        // other macs
        mac mac_u (   
          .CLK   (CLK),                 
          .RST_N (RST_N),               
          .EN    (mac_en),
          .X_IN  (x_pipeline[i-1][j]),
          .W_IN  (weights[j][8*(i+1)-1:i*8]),          
          .Y_IN  (partial_sums[i][j-1]),  
          .X_OUT (x_pipeline[i][j]),
          .Y_OUT (partial_sums[i][j]) 
        );
      end
    end
  end

  //-------------------------------------------------------------------------// 
  // assign ARRAY_OUT to the final row MACs at (x,N-1)
  for (genvar i = 0; i < N; i++) begin
    assign ARRAY_OUT[i] = partial_sums[i][N-1];
  end

  //-------------------------------------------------------------------------//
  // generate BANKS SRAM banks
  //-------------------------------------------------------------------------//  
  for (genvar k = 0; k < BANKS; k++) begin
    sram_bank #(
      .NUM_WORDS  (NUM_WORDS),
      .WORD_WIDTH (64),
      .ADDR_WIDTH (ADDR_WIDTH)
    ) sram_bank_u (
      .CLK        (CLK),
      .RST_N      (RST_N),
      .WR_EN      (WEIGHT_WR_EN[k]),
      .WR_ADDR    (WEIGHT_WR_ADDR[k]),
      .WR_DATA    (WEIGHT_WR_DATA[k]),
      .RD_EN      (sram_en[k]),
      .RD_ADDR    (sram_addr[k]),
      .RD_DATA    (rd_data[k]),
      .RD_VALID   (rd_valid[k])
    );
  end

endmodule : top
