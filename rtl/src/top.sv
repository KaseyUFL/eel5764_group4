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
  parameter M = 8,
  parameter BANKS = 8,  // bank count (number of SRAM banks)
  parameter DOUBLE_BUFFER = 1, // 1 = load the next tile's weights while the array is running
  parameter WORDS_PER_COLUMN = (N + 7) / 8,
  parameter COLUMNS_PER_BANK = (N + BANKS - 1) / BANKS,
  parameter SRAM_DEPTH = WORDS_PER_COLUMN * COLUMNS_PER_BANK,
  parameter integer ADDR_WIDTH = (SRAM_DEPTH > 1) ? $clog2(SRAM_DEPTH) : 1

)(   
  input  logic                   CLK,      // clock
  input  logic                   RST_N,    // reset (active low)
  input  logic                   EN,       // enable
  input  logic                   WEIGHTS_READY, // host has written a new tile into the SRAM
  output logic                   weights_taken, // controller is done reading the SRAM
  input logic [N-1:0][7:0] ACTIVATION_IN, // input_buffer activation
  output logic signed [N-1:0][31:0] ARRAY_OUT, // array outputs (BANKS * 8 bytes)

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
  logic [BANKS-1:0][63:0] rd_data;
  //logic [BANKS-1:0][63:0]        weights; //weight storage returned from SRAM
  logic [BANKS-1:0]       weight_valid;

  // controller signals
  //logic all_weights_valid;
  logic stall;
  logic mac_en;
  logic input_buffer_en;
  logic sram_en;

  // MAC weight loading controls
  logic [N-1:0][N-1:0] w_load;
  logic [N-1:0][N-1:0] w_swap;
  logic [N-1:0][N-1:0][7:0] w_data;

  // new stuff
  logic [N-1:0][7:0]         input_pipe;    // N * 8-bit input pipeline  
  logic signed [N-1:0][N-1:0][31:0]  partial_sums;  // N * N * 8-bit partial sum array 
  logic [N-1:0][N-1:0][7:0]  x_pipeline;    // N * N * 8-bit x array    

  logic [N-1:0][7:0] buffer_input;

  //-------------------------------------------------------------------------//
  // controller: loads weights from SRAM, swaps them in, then runs the array
  //-------------------------------------------------------------------------//
  controller #(
    .N     (N),
    .M     (M),
    .BANKS (BANKS),
    .DOUBLE_BUFFER (DOUBLE_BUFFER)
  ) controller_u (
    .CLK             (CLK),
    .RST_N           (RST_N),
    .EN              (EN),
    .WEIGHTS_READY   (WEIGHTS_READY),
    .weights_taken   (weights_taken),
    .ACTIVATION_IN   (ACTIVATION_IN),
    .weight_valid    (weight_valid),
    .rd_data         (rd_data),
    .sram_en         (sram_en),
    .sram_addr       (sram_addr),
    .mac_en          (mac_en),
    .w_load          (w_load),
    .w_swap          (w_swap),
    .w_data          (w_data),
    .input_buffer_en (input_buffer_en),
    .buffer_input    (buffer_input),
    .stall           (stall),
    .stall_count     (stall_count),
    .active_count    (active_count)
  );


  //-------------------------------------------------------------------------//
  // buffer inputs, row number * 1 clock periods
  //-------------------------------------------------------------------------//
  input_buffer #(
    .N     (N)
  ) input_buffer_u (
    .CLK   (CLK),   
    .RST_N (RST_N), 
    .EN    (input_buffer_en),    
    .X_IN  (buffer_input),  
    .Y_OUT (input_pipe) 
  );


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
        .EN    (mac_en),             
        .X_IN  (input_pipe[j]),        
        .W_IN(w_data[0][j]),   
        .W_LOAD(w_load[0][j]),
        .W_SWAP(w_swap[0][j]),      
        .Y_IN  (32'sd0),
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
        .W_IN(w_data[0][j]),
        .W_LOAD(w_load[0][j]),
        .W_SWAP(w_swap[0][j]),          
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
          .W_IN   (w_data[i][j]),
          .W_LOAD (w_load[i][j]),
          .W_SWAP (w_swap[i][j]),         
          .Y_IN  (32'sd0),
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
          .W_IN   (w_data[i][j]),
          .W_LOAD (w_load[i][j]),
          .W_SWAP (w_swap[i][j]),                 
          .Y_IN  (partial_sums[i][j-1]),  
          .X_OUT (x_pipeline[i][j]),
          .Y_OUT (partial_sums[i][j]) 
        );
      end

    end

  end

  //-------------------------------------------------------------------------//
  // partial sums from the bottom edge of the array
  //-------------------------------------------------------------------------//

  for (genvar i = 0; i < N; i++) begin : gen_outputs
    assign ARRAY_OUT[i] = partial_sums[i][N-1];
  end

  //-------------------------------------------------------------------------//
  // generate BANKS SRAM banks
  //-------------------------------------------------------------------------//  
  for (genvar k = 0; k < BANKS; k++) begin
    sram_bank #(
      .NUM_WORDS  (SRAM_DEPTH),
      .WORD_WIDTH (64),
      .ADDR_WIDTH (ADDR_WIDTH)
    ) sram_bank_u (
      .CLK        (CLK),
      .RST_N      (RST_N),
      .WR_EN      (WEIGHT_WR_EN[k]),
      .WR_ADDR    (WEIGHT_WR_ADDR[k]),
      .WR_DATA    (WEIGHT_WR_DATA[k]),
      .RD_EN      (sram_en),
      .RD_ADDR    (sram_addr),
      .RD_DATA    (rd_data[k]),
      .RD_VALID   (weight_valid[k])
    );
  end

endmodule : top
