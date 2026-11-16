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
  parameter WORDS_PER_COLUMN = (N + 7) / 8,
  parameter COLUMNS_PER_BANK = (N + BANKS - 1) / BANKS,
  parameter SRAM_DEPTH = WORDS_PER_COLUMN * COLUMNS_PER_BANK,
  parameter integer ADDR_WIDTH = (SRAM_DEPTH > 1) ? $clog2(SRAM_DEPTH) : 1

)(   
  input  logic                   CLK,      // clock
  input  logic                   RST_N,    // reset (active low)
  input  logic                   EN,       // enable
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

  logic all_banks_valid;

  logic [N-1:0][7:0] buffer_input;

  integer group_count;
  integer chunk_count;
  integer run_count;
  integer drain_count;

  // need new FSM: IDLE, LOAD_WEIGHTS, SWAP_WEIGHTS, LOAD_INPUTS, RUN, DONE
   //-------------------------------------------------------------------------//
  // TODO controller: (current SRAM control needs replacement)
  // - needs to load weights register after reset
  // - needs to load activation weights and stall if necessary
  //-------------------------------------------------------------------------//
  typedef enum logic [2:0] {
    IDLE,
    LOAD_WEIGHTS,
    SWAP_WEIGHTS,
    LOAD_INPUTS,
    RUN,
    DRAIN,
    DONE
  } state_t;

  state_t state, next_state;

  localparam integer DRAIN_CYCLES = N - 1;//(2 * N) - 2;
  assign all_banks_valid = &weight_valid;

  assign sram_en = (state == LOAD_WEIGHTS) && !all_banks_valid;

  assign sram_addr =(group_count * WORDS_PER_COLUMN) + chunk_count;

  always_ff @(posedge CLK or negedge RST_N) begin

    if (!RST_N) begin

      state <= IDLE;
      group_count <= 0;
      chunk_count <= 0;
      run_count   <= 0;
      drain_count <= 0;
      stall_count  <= '0;
      active_count <= '0;

    end else begin
      state <= next_state;
      // advance the weight-loading counters when SRAM data is valid
      if (EN) begin
        if (stall)
          stall_count <= stall_count + 1'b1;
        else if (state == RUN && mac_en)
          active_count <= active_count + 1'b1;
      end
      
      if ((state == LOAD_WEIGHTS) && all_banks_valid) begin
        if (chunk_count == WORDS_PER_COLUMN - 1) begin
          chunk_count <= 0;
          if (group_count == COLUMNS_PER_BANK - 1)
            group_count <= 0;
          else
            group_count <= group_count + 1;
        end else begin
          chunk_count <= chunk_count + 1;
        end
      end

      if (state == LOAD_INPUTS)
        run_count <= 0; // reset activation counter before starting a tile

      else if ((state == RUN) && EN)
        run_count <= run_count + 1;

      if ((state == RUN) && (next_state == DRAIN))
        drain_count <= 0; // drain after the final activation cycle

      else if ((state == DRAIN) && EN)
        drain_count <= drain_count + 1; // cycles to flush the pipeline
    end
  end

  always_comb begin
    next_state = state;
    mac_en = 1'b0;
    input_buffer_en = 1'b0;
    stall = 1'b0;
    w_load = '0;
    w_swap = '0;
    w_data = '0;
    buffer_input = '0;

    case (state)
      IDLE: begin
        if (EN)
          next_state = LOAD_WEIGHTS;
      end

      LOAD_WEIGHTS: begin
        stall = 1'b1;
        if (all_banks_valid) begin
          for (integer b = 0; b < BANKS; b = b + 1) begin
            // column assigned to this bank for the current group
            if ((group_count * BANKS + b) < N) begin
              for (integer lane = 0; lane < 8; lane = lane + 1) begin
                // row assigned to this byte lane
                if ((chunk_count * 8 + lane) < N) begin
                  w_data[group_count * BANKS + b]
                        [chunk_count * 8 + lane]
                      = rd_data[b][lane * 8 +: 8];
                  w_load[group_count * BANKS + b]
                        [chunk_count * 8 + lane] = 1'b1;
                end
              end
            end
          end
          // check whether all column groups and chunks have been loaded
          if ((group_count == COLUMNS_PER_BANK - 1) &&
              (chunk_count == WORDS_PER_COLUMN - 1))
            next_state = SWAP_WEIGHTS;
        end
      end

      SWAP_WEIGHTS: begin // move staging weights into active MAC weight registers
        w_swap = '1;
        next_state = LOAD_INPUTS;
      end

      LOAD_INPUTS: begin // prepare to feed activations into the skew buffer
        next_state = RUN;
      end

      RUN: begin // process M activation cycles

        mac_en = EN;
        input_buffer_en = EN;
        buffer_input = ACTIVATION_IN;
        if (EN && (run_count == M - 1))
          next_state = DRAIN;
      end

      DRAIN: begin //flush remaining activations and partial sums through the array

        mac_en = EN;
        input_buffer_en = EN;
        // Feed zeros so the remaining pipeline values can propagate
        buffer_input = '0;
        if (EN && (drain_count == DRAIN_CYCLES - 1))
          next_state = DONE;
      end

      DONE: begin
        if (!EN)
          next_state = IDLE;
      end

      default: begin
        next_state = IDLE;
      end
    endcase
  end


  //-------------------------------------------------------------------------//
  // buffer inputs, row number * 1 clock periods
  //-------------------------------------------------------------------------//
  input_buffer #(
    .N     (N)
  )(   
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
