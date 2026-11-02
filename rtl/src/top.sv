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
  parameter integer ADDR_WIDTH = (N > 1) ? $clog2(N) : 1
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

  // internal signals

  logic [ADDR_WIDTH-1:0] sram_addr;

  logic [BANKS-1:0][63:0]        weights;
  logic [BANKS-1:0]       weight_valid;

  logic all_weights_valid;
  logic stall;
  logic mac_en;
  logic sram_en;

  // new stuff
  logic [N-1:0][7:0]         input_pipe;    // N * 8-bit input pipeline
  logic [N-1:0][63:0]        weights;       // N * 8-byte weight array   
  logic [N-1:0][N-1:0][7:0]  partial_sums;  // N * N * 8-bit partial sum array 
  logic [N-1:0][N-1:0][7:0]  x_pipeline;    // N * N * 8-bit x array    

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
  // buffer inputs, row number * 1 clock periods
  //-------------------------------------------------------------------------//
  input_buffer #(
    .N     (N)
  )(   
    .CLK   (CLK),   
    .RST_N (RST_N), 
    .EN    (/* TODO */),    
    .X_IN  (/* TODO */),  
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
    // - EN    : TBD
    // - X_IN  : input_pipe[j]
    // - W_IN  : weights[0][7:0]
    // - Y_IN  : if j = 0, 0x0, else, partial_sums[0][j-1]
    // - X_OUT : x_pipeline[0][j]
    // - Y_OUT : partial_sums[0][j]   
    //-----------------------------------------------------------------------//
    if (j == 0) begin 
      // MAC at (0,0)       
      mac mac_u (   
        .CLK   (CLK),            
        .RST_N (RST_N),               
        .EN    (/* TBD */),             
        .X_IN  (input_pipe[j]),        
        .W_IN  (weights[0][7:0]),          
        .Y_IN  (8'h0),
        .X_OUT (x_pipeline[0][j]),
        .Y_OUT (partial_sums[0][j]) 
      );
    end else begin
      // MAC at (0,j), where j > 0 
      mac mac_u (   
        .CLK   (CLK),                 
        .RST_N (RST_N),               
        .EN    (/* TBD */),              
        .X_IN  (input_pipe[j]),       
        .W_IN  (weights[0][7:0]),          
        .Y_IN  (partial_sums[0][j-1]), 
        .X_OUT (x_pipeline[0][j]),        
        .Y_OUT (partial_sums[0][j]) 
      );
    end
    //-----------------------------------------------------------------------//
    // i is the x-coordinate, this loop creates N-1 columns of MACs  
    for (genvar i = 1; i < N; i++) begin
      //---------------------------------------------------------------------//
      // MACs at (i,j), where j > 0:
      // - EN    : TBD
      // - X_IN  : x_pipeline[i][j]
      // - W_IN  : weights[j][8*(i+1)-1:i*8]
      // - Y_IN  : if j = 0, 0x0, else, partial_sums[i][j-1]
      // - X_OUT : x_pipeline[i-1][j]
      // - Y_OUT : partial_sums[i][j]   
      //---------------------------------------------------------------------//      
      if (j == 0) begin
        // MACs at (i,0), where i > 0:
        mac mac_u (   
          .CLK   (CLK),                 
          .RST_N (RST_N),               
          .EN    (/* TBD */),              
          .X_IN  (x_pipeline[i][j]),         
          .W_IN  (weights[j][8*(i+1)-1:i*8]),          
          .Y_IN  (8'h0),
          .X_OUT (x_pipeline[i-1][j]),
          .Y_OUT (partial_sums[i][j]) 
        );
      end else begin
        // other macs
        mac mac_u (   
          .CLK   (CLK),                 
          .RST_N (RST_N),               
          .EN    (/* TBD */),              
          .X_IN  (x_pipeline[i][j]),         
          .W_IN  (weights[j][8*(i+1)-1:i*8]),          
          .Y_IN  (partial_sums[i][j-1]),  
          .X_OUT (x_pipeline[i-1][j]),
          .Y_OUT (partial_sums[i][j]) 
        );
      end
    end
    //-----------------------------------------------------------------------// 
    // assign ARRAY_OUT to the final row MACs at (x,N-1)
    assign ARRAY_OUT[i] = partial_sums[i][N-1];
  end

  //-------------------------------------------------------------------------//
  // generate BANKS SRAM banks
  //-------------------------------------------------------------------------//  
  for (genvar k = 0; k < BANKS; k++) begin
    sram_bank #(
      .NUM_WORDS  (N),
      .WORD_WIDTH (64),
      .ADDR_WIDTH (ADDR_WIDTH)
    ) sram_bank_u (
      .CLK        (CLK),
      .RST_N      (RST_N),
      .WR_EN      (WEIGHT_WR_EN[i]),
      .WR_ADDR    (WEIGHT_WR_ADDR[i]),
      .WR_DATA    (WEIGHT_WR_DATA[i]),
      .RD_EN      (sram_en),
      .RD_ADDR    (sram_addr),
      .RD_DATA    (rd_data[BANKS]),
      .RD_VALID   (weight_valid[i])
    );
  end

endmodule : top
