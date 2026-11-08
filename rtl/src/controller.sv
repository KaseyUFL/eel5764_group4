//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : controller.sv                                               //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //
// Group       : #4                                                          //
// Description : Array controller. Phase 1 fills the weight registers from   //
//               the SRAM banks, phase 2 streams the input buffer into the   //
//               MACs and stalls when no input is available.                 //
//---------------------------------------------------------------------------//
module controller #(
  parameter N     = 8, // rows (words per SRAM bank / MACs per bank)
  parameter BANKS = 8, // columns (SRAM banks)
  parameter integer ADDR_WIDTH = (N > 1) ? $clog2(N) : 1
)(
  input  logic                  CLK,          // clock
  input  logic                  RST_N,        // reset (active low)
  input  logic                  START,        // begin (re)loading weights

  // SRAM read interface (address is shared by all banks)
  output logic                  SRAM_RD_EN,
  output logic [ADDR_WIDTH-1:0] SRAM_RD_ADDR,
  input  logic [BANKS-1:0]      SRAM_RD_VALID,

  // weight register load: capture bank data into row W_LOAD_ROW
  output logic                  W_LOAD_EN,
  output logic [ADDR_WIDTH-1:0] W_LOAD_ROW,
  output logic                  WEIGHTS_READY,

  // input buffer handshake
  input  logic                  IN_VALID,     // input buffer has data
  output logic                  IN_READY,     // pop input buffer this cycle

  // array control
  output logic                  MAC_EN,
  output logic                  STALL,

  // measurement counters
  output logic [31:0]           stall_count,
  output logic [31:0]           active_count,
  output logic [31:0]           load_count
);

  typedef enum logic [1:0] {IDLE, LOAD_W, COMPUTE} state_t;
  state_t state;

  logic                  issue_done; // all N read requests issued
  logic [ADDR_WIDTH-1:0] rd_addr_q;  // address of the data now on RD_DATA
  logic                  all_valid;

  //-------------------------------------------------------------------------//
  // combinational outputs
  //-------------------------------------------------------------------------//
  // all banks read in lockstep, so BANKS only affects the width of this AND
  assign all_valid     = &SRAM_RD_VALID;

  assign SRAM_RD_EN    = (state == LOAD_W) && !issue_done;
  assign W_LOAD_EN     = (state == LOAD_W) && all_valid;
  assign W_LOAD_ROW    = rd_addr_q;
  assign WEIGHTS_READY = (state == COMPUTE);

  assign MAC_EN        = (state == COMPUTE) && IN_VALID;
  assign IN_READY      = MAC_EN;
  assign STALL         = (state == COMPUTE) && !IN_VALID;

  //-------------------------------------------------------------------------//
  // state machine
  //-------------------------------------------------------------------------//
  always_ff @(posedge CLK or negedge RST_N) begin
    if (!RST_N) begin
      state        <= IDLE;
      SRAM_RD_ADDR <= '0;
      rd_addr_q    <= '0;
      issue_done   <= 1'b0;
    end else begin
      case (state)
        IDLE: begin
          if (START) begin
            state        <= LOAD_W;
            SRAM_RD_ADDR <= '0;
            issue_done   <= 1'b0;
          end
        end

        LOAD_W: begin
          // issue one read per cycle, row 0 .. N-1
          if (SRAM_RD_EN) begin
            rd_addr_q <= SRAM_RD_ADDR; // tracks the 1-cycle SRAM latency
            if (SRAM_RD_ADDR == ADDR_WIDTH'(N-1))
              issue_done <= 1'b1;
            else
              SRAM_RD_ADDR <= SRAM_RD_ADDR + 1'b1;
          end
          // done once the last row has been captured
          if (W_LOAD_EN && rd_addr_q == ADDR_WIDTH'(N-1))
            state <= COMPUTE;
        end

        COMPUTE: begin
          if (START) begin
            state        <= LOAD_W;
            SRAM_RD_ADDR <= '0;
            issue_done   <= 1'b0;
          end
        end

        default: state <= IDLE;
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
      load_count   <= '0;
    end else begin
      if (state == LOAD_W) load_count   <= load_count + 1'b1;
      if (STALL)           stall_count  <= stall_count + 1'b1;
      if (MAC_EN)          active_count <= active_count + 1'b1;
    end
  end

endmodule : controller
