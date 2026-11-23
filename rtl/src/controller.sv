//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : controller.sv                                               //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //
// Group       : #4                                                          //
// Description : Controller                                                  //
//---------------------------------------------------------------------------//

module controller #(
    parameter N = 8,
    parameter M = 8,
    parameter BANKS = 8,
    parameter DOUBLE_BUFFER = 1, // 1 = load the next tile's weights while the array is running
    parameter WORDS_PER_COLUMN = (N + 7) / 8,
    parameter COLUMNS_PER_BANK = (N + BANKS - 1) / BANKS,
    parameter SRAM_DEPTH = WORDS_PER_COLUMN * COLUMNS_PER_BANK,
    parameter integer ADDR_WIDTH = (SRAM_DEPTH > 1) ? $clog2(SRAM_DEPTH) : 1
)(
    input logic CLK,
    input logic RST_N,
    input logic EN,
    // weight handshake with the host
    input logic WEIGHTS_READY,  // host has written a new tile of weights into the SRAM
    output logic weights_taken, // pulses when the last SRAM word is loaded, host can write the next tile
    // activation input
    input logic [N-1:0][7:0] ACTIVATION_IN,
    // SRAM read interface
    input logic [BANKS-1:0] weight_valid,
    input logic [BANKS-1:0][63:0] rd_data,
    output logic sram_en,
    output logic [ADDR_WIDTH-1:0] sram_addr,
    // MAC array controls
    output logic mac_en,
    output logic [N-1:0][N-1:0] w_load,
    output logic [N-1:0][N-1:0] w_swap,
    output logic [N-1:0][N-1:0][7:0] w_data,
    // input buffer controls
    output logic input_buffer_en,
    output logic [N-1:0][7:0] buffer_input,
    // status and performance counters
    output logic stall,
    output logic [31:0] stall_count,
    output logic [31:0] active_count
);

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

  logic all_banks_valid;
  logic loading; // reading SRAM words into the MAC staging registers
  logic staged;  // staging registers hold a full tile that has not been swapped in yet
  // counters sized to their actual range instead of 32-bit integers
  localparam integer GW = (COLUMNS_PER_BANK > 1) ? $clog2(COLUMNS_PER_BANK) : 1;
  localparam integer CW = (WORDS_PER_COLUMN > 1) ? $clog2(WORDS_PER_COLUMN) : 1;
  logic [GW-1:0] group_count;
  logic [CW-1:0] chunk_count;
  integer run_count;
  integer drain_count;

  localparam integer DRAIN_CYCLES = (2 * N) - 1; // last output needs 2N-1 cycles to leave the array
  assign all_banks_valid = &weight_valid;// all SRAM banks must return valid data before loading the MACs

  // load while stalled in LOAD_WEIGHTS, or in the background while the array is running
  assign loading = WEIGHTS_READY && !staged &&
                   ((state == LOAD_WEIGHTS) || (DOUBLE_BUFFER && (state == RUN || state == DRAIN)));
  assign sram_en = loading && !all_banks_valid;
  assign weights_taken = loading && all_banks_valid &&
                         (group_count == COLUMNS_PER_BANK - 1) && (chunk_count == WORDS_PER_COLUMN - 1);
  integer address_value;
  always_comb begin
    address_value = (group_count * WORDS_PER_COLUMN) + chunk_count;
    sram_addr = address_value[ADDR_WIDTH-1:0];
  end

  always_ff @(posedge CLK or negedge RST_N) begin

    if (!RST_N) begin
      state <= IDLE;
      group_count <= 0;
      chunk_count <= 0;
      run_count   <= 0;
      drain_count <= 0;
      staged      <= 0;
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
      
      if (loading && all_banks_valid) begin
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

      // staging registers are full after the last word, and free again after the swap
      if (weights_taken)
        staged <= 1;
      else if (state == SWAP_WEIGHTS)
        staged <= 0;

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

  // weight data is statically wired: each MAC always reads the same bank/lane,
  // only w_load selects which MACs capture it
  for (genvar i = 0; i < N; i++) begin : gen_wdata_col
    for (genvar j = 0; j < N; j++) begin : gen_wdata_row
      assign w_data[i][j] = rd_data[i % BANKS][(j % 8) * 8 +: 8];
    end
  end

  always_comb begin
    next_state = state;
    mac_en = 1'b0;
    input_buffer_en = 1'b0;
    stall = 1'b0;
    w_load = '0;
    w_swap = '0;
    buffer_input = '0;

    // MAC column i is fed by bank (i % BANKS) during group (i / BANKS);
    // MAC row j is fed by byte lane (j % 8) during chunk (j / 8)
    if (loading && all_banks_valid) begin
      for (integer i = 0; i < N; i = i + 1)
        for (integer j = 0; j < N; j = j + 1)
          w_load[i][j] = (group_count == i / BANKS) && (chunk_count == j / 8);
    end

    case (state)
      IDLE: begin
        if (EN)
          next_state = LOAD_WEIGHTS;
      end

      LOAD_WEIGHTS: begin
        stall = 1'b1;
        // wait here until the whole tile is in the staging registers
        if (staged || weights_taken)
          next_state = SWAP_WEIGHTS;
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
        // feed zeros so the remaining pipeline values can propagate
        buffer_input = '0;
        if (EN && (drain_count == DRAIN_CYCLES - 1)) begin
          if (staged || weights_taken)
            next_state = SWAP_WEIGHTS; // next tile already loaded, swap it in
          else if (WEIGHTS_READY)
            next_state = LOAD_WEIGHTS; // next tile still loading, stall until it finishes
          else
            next_state = DONE;         // no more tiles
        end
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

endmodule : controller