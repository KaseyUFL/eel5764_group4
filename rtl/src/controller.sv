//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : controller.sv                                               //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //
// Group       : #4                                                          //
// Description : Array controller.                                           //
//---------------------------------------------------------------------------//
module controller #(
  parameter N          = 8, // array size (N x N MACs)
  parameter BANKS      = 8, // SRAM bank count
  parameter NUM_INPUTS = 8, // input vectors stored after the weights
  parameter NUM_WORDS  = 8, // words per bank
  parameter integer ADDR_WIDTH  = (NUM_WORDS > 1) ? $clog2(NUM_WORDS) : 1,
  parameter integer WPV         = (N + 7) / 8, // 64-bit words per row/vector
  parameter integer CHUNK_WIDTH = (WPV > 1) ? $clog2(WPV) : 1,
  parameter integer STEP_WIDTH  = $clog2(N + NUM_INPUTS + 1)
)(
  input  logic                                CLK,     // clock
  input  logic                                RST_N,   // reset (active low)
  input  logic                                EN,      // start; hold high to run once

  // SRAM read interface, one port per bank
  output logic [BANKS-1:0]                    SRAM_RD_EN,
  output logic [BANKS-1:0][ADDR_WIDTH-1:0]    SRAM_RD_ADDR,
  input  logic [BANKS-1:0]                    SRAM_RD_VALID,

  // returned words: bank b holds chunk RET_CHUNK[b] of row/vector RET_ROW
  output logic                                RET_VALID,
  output logic                                RET_WEIGHT,  // weight row (else input)
  output logic [STEP_WIDTH-1:0]               RET_ROW,     // weight row index
  output logic [BANKS-1:0]                    RET_EN,
  output logic [BANKS-1:0][CHUNK_WIDTH-1:0]   RET_CHUNK,

  // array control
  output logic                                X_VALID,  // input vector is complete
  output logic                                ARRAY_EN, // advance input buffer + MACs
  output logic                                STALL,
  output logic                                DONE,

  // measurement counters
  output logic [31:0]                         stall_count,
  output logic [31:0]                         active_count
);

  localparam integer STEPS   = N + NUM_INPUTS;  // weight rows + input vectors
  localparam integer DRAIN   = 2*N - 1;         // flush cycles
  localparam integer FED_W   = $clog2(NUM_INPUTS + 1);
  localparam integer DRAIN_W = $clog2(DRAIN + 1);
  localparam integer BANK_W  = (BANKS > 1) ? $clog2(BANKS) : 1;

  // weights and inputs must fit in BANKS banks of NUM_WORDS words
  if (NUM_INPUTS < 1 || STEPS * WPV > BANKS * NUM_WORDS) begin : g_check_mem
    $error("controller: %0d words of weights and inputs do not fit in %0d banks of %0d words",
           STEPS * WPV, BANKS, NUM_WORDS);
  end

  typedef enum logic [1:0] {S_IDLE, S_RUN, S_DRAIN, S_DONE} state_t;
  state_t state;

  // read pointer: next word is chunk c0 of row/vector step, in bank0 at addr0
  logic [STEP_WIDTH-1:0]  step;
  logic [CHUNK_WIDTH-1:0] c0;
  logic [BANK_W-1:0]      bank0;
  logic [ADDR_WIDTH-1:0]  addr0;

  logic                   issue;     // read a group of words this cycle
  logic                   grp_last;  // group finishes its row/vector
  logic [CHUNK_WIDTH:0]   grp_size;  // words in the group
  logic [BANK_W:0]        bank_sum;  // bank0 + grp_size
  logic [BANKS-1:0][CHUNK_WIDTH-1:0] rd_chunk;

  // 1-cycle SRAM read latency tracking
  logic                   ret_q;
  logic                   ret_last_q;
  logic [STEP_WIDTH-1:0]  ret_step_q;

  logic                   w_done;    // all weight rows loaded
  logic                   x_ready_q; // a complete input vector is in the buffer
  logic [FED_W-1:0]       x_fed;     // input vectors sent to the array
  logic [DRAIN_W-1:0]     drain_cnt;

  //-------------------------------------------------------------------------//
  // read issue: up to BANKS consecutive words, one per bank
  //-------------------------------------------------------------------------//
  assign issue = (state == S_RUN) && (step < STEPS);

  always_comb begin
    int rem, size, off;
    rem      = WPV - int'(c0);
    size     = (rem < BANKS) ? rem : BANKS;
    grp_last = (rem <= BANKS);
    grp_size = (CHUNK_WIDTH+1)'(size);
    bank_sum = (BANK_W+1)'(int'(bank0) + size);

    SRAM_RD_EN   = '0;
    SRAM_RD_ADDR = '0;
    rd_chunk     = '0;
    for (int b = 0; b < BANKS; b++) begin
      // word c0+off of the group lands in bank b
      if (b >= int'(bank0)) begin
        off             = b - int'(bank0);
        SRAM_RD_ADDR[b] = addr0;
      end else begin
        off             = b + BANKS - int'(bank0);
        SRAM_RD_ADDR[b] = addr0 + 1'b1;
      end
      rd_chunk[b] = CHUNK_WIDTH'(int'(c0) + off);
      if (issue && off < size)
        SRAM_RD_EN[b] = 1'b1;
    end
  end

  //-------------------------------------------------------------------------//
  // combinational outputs
  //-------------------------------------------------------------------------//
  assign RET_VALID  = ret_q && (&(SRAM_RD_VALID | ~RET_EN));
  assign RET_WEIGHT = (ret_step_q < N);
  assign RET_ROW    = ret_step_q;

  assign X_VALID    = x_ready_q;
  assign ARRAY_EN   = ((state == S_RUN) && x_ready_q) || (state == S_DRAIN);
  assign STALL      = (state == S_RUN) && w_done && !x_ready_q;
  assign DONE       = (state == S_DONE);

  //-------------------------------------------------------------------------//
  // state machine
  //-------------------------------------------------------------------------//
  always_ff @(posedge CLK or negedge RST_N) begin
    if (!RST_N) begin
      state      <= S_IDLE;
      step       <= '0;
      c0         <= '0;
      bank0      <= '0;
      addr0      <= '0;
      ret_q      <= 1'b0;
      ret_last_q <= 1'b0;
      ret_step_q <= '0;
      RET_EN     <= '0;
      RET_CHUNK  <= '0;
      w_done     <= 1'b0;
      x_ready_q  <= 1'b0;
      x_fed      <= '0;
      drain_cnt  <= '0;
    end else begin
      ret_q      <= issue;
      ret_last_q <= grp_last;
      ret_step_q <= step;
      RET_EN     <= SRAM_RD_EN;
      RET_CHUNK  <= rd_chunk;

      // a vector is ready the cycle after its last word returns
      x_ready_q  <= RET_VALID && !RET_WEIGHT && ret_last_q;
      if (RET_VALID && ret_last_q && ret_step_q == STEP_WIDTH'(N-1))
        w_done <= 1'b1;

      case (state)
        S_IDLE: begin
          if (EN) begin
            state     <= S_RUN;
            step      <= '0;
            c0        <= '0;
            bank0     <= '0;
            addr0     <= '0;
            w_done    <= 1'b0;
            x_fed     <= '0;
            drain_cnt <= '0;
          end
        end

        S_RUN: begin
          if (issue) begin
            if (grp_last) begin
              step <= step + 1'b1;
              c0   <= '0;
            end else begin
              c0   <= c0 + grp_size[CHUNK_WIDTH-1:0];
            end
            if (bank_sum >= BANKS) begin
              bank0 <= BANK_W'(bank_sum - BANKS);
              addr0 <= addr0 + 1'b1;
            end else begin
              bank0 <= bank_sum[BANK_W-1:0];
            end
          end
          if (x_ready_q) begin
            x_fed <= x_fed + 1'b1;
            if (x_fed == FED_W'(NUM_INPUTS-1))
              state <= S_DRAIN;
          end
        end

        S_DRAIN: begin
          drain_cnt <= drain_cnt + 1'b1;
          if (drain_cnt == DRAIN_W'(DRAIN-1))
            state <= S_DONE;
        end

        S_DONE: begin
          if (!EN)
            state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  //-------------------------------------------------------------------------//
  // measurement counters
  //-------------------------------------------------------------------------//
  always_ff @(posedge CLK or negedge RST_N) begin
    if (!RST_N) begin
      stall_count  <= '0;
      active_count <= '0;
    end else begin
      if (STALL)    stall_count  <= stall_count + 1'b1;  // waiting on SRAM
      if (ARRAY_EN) active_count <= active_count + 1'b1; // array advanced
    end
  end

endmodule : controller
