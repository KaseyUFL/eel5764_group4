`timescale 1ns/1ps

module tb_top;

  // change these to test different sizes
  // sizes we ran: N=8 BANKS=8, N=32 BANKS=1, N=32 BANKS=8, N=32 BANKS=32,
  //               N=16 BANKS=4, N=12 BANKS=5, N=4 BANKS=8
  parameter N         = 32;
  parameter M         = 8;
  parameter BANKS     = 1;
  parameter NUM_TILES = 2;

  localparam WORDS_PER_COLUMN = (N + 7) / 8;
  localparam COLUMNS_PER_BANK = (N + BANKS - 1) / BANKS;
  localparam SRAM_DEPTH       = WORDS_PER_COLUMN * COLUMNS_PER_BANK;
  localparam ADDR_WIDTH       = (SRAM_DEPTH > 1) ? $clog2(SRAM_DEPTH) : 1;

  // each SRAM word takes 2 cycles (1 to read, 1 to load into the MACs)
  localparam EXPECTED_STALLS  = 2 * SRAM_DEPTH;

  localparam IDLE         = 0;
  localparam LOAD_WEIGHTS = 1;
  localparam SWAP_WEIGHTS = 2;
  localparam LOAD_INPUTS  = 3;
  localparam RUN          = 4;
  localparam DRAIN        = 5;
  localparam DONE         = 6;

  logic                             CLK;
  logic                             RST_N;
  logic                             EN;
  logic [N-1:0][7:0]                ACTIVATION_IN;
  logic signed [N-1:0][31:0]        ARRAY_OUT;
  logic [BANKS-1:0]                 WEIGHT_WR_EN;
  logic [BANKS-1:0][ADDR_WIDTH-1:0] WEIGHT_WR_ADDR;
  logic [BANKS-1:0][63:0]           WEIGHT_WR_DATA;
  logic [31:0]                      stall_count;
  logic [31:0]                      active_count;

  top #(
    .N     (N),
    .M     (M),
    .BANKS (BANKS)
  ) dut (
    .CLK            (CLK),
    .RST_N          (RST_N),
    .EN             (EN),
    .ACTIVATION_IN  (ACTIVATION_IN),
    .ARRAY_OUT      (ARRAY_OUT),
    .WEIGHT_WR_EN   (WEIGHT_WR_EN),
    .WEIGHT_WR_ADDR (WEIGHT_WR_ADDR),
    .WEIGHT_WR_DATA (WEIGHT_WR_DATA),
    .stall_count    (stall_count),
    .active_count   (active_count)
  );

  initial CLK = 0;
  always #5 CLK = ~CLK;

  // signals inside the controller 
  wire [2:0] state  = dut.controller_u.state;
  wire       stall  = dut.controller_u.stall;
  wire       mac_en = dut.controller_u.mac_en;

  // test data
  logic signed [7:0] weights  [N][N];  // weights[i][j] goes to MAC(i,j)
  logic signed [7:0] acts     [M][N];  // acts[t][j] is activation vector t
  integer            expected [M][N];  // expected[t][i] = sum of weights[i][j] * acts[t][j]

  // bookkeeping for each tile
  integer            load_count  [N][N];
  logic signed [7:0] loaded_w    [N][N];
  logic              got_output  [M][N];
  integer            load_cycles;
  integer            swap_cycles;
  integer            mac_cycles;
  integer            outputs_seen;
  logic              check_outputs;

  integer errors = 0;
  integer tile;

  //-------------------------------------------------------------------------//
  // make random weights and activations, and compute the expected outputs
  //-------------------------------------------------------------------------//
  task make_data;
    begin
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
          weights[i][j] = $random;

      for (int t = 0; t < M; t++)
        for (int j = 0; j < N; j++)
          acts[t][j] = $random;

      for (int t = 0; t < M; t++)
        for (int i = 0; i < N; i++) begin
          expected[t][i] = 0;
          for (int j = 0; j < N; j++)
            expected[t][i] = expected[t][i] + weights[i][j] * acts[t][j];
        end
    end
  endtask

  //-------------------------------------------------------------------------//
  // write the weights into SRAM
  // column i goes to bank (i % BANKS), row j goes to byte lane (j % 8)
  // address = (i / BANKS) * WORDS_PER_COLUMN + (j / 8)
  //-------------------------------------------------------------------------//
  task write_sram;
    integer col, row;
    begin
      for (int a = 0; a < SRAM_DEPTH; a++) begin
        @(negedge CLK);
        for (int b = 0; b < BANKS; b++) begin
          col = (a / WORDS_PER_COLUMN) * BANKS + b;
          WEIGHT_WR_EN[b]   = 1;
          WEIGHT_WR_ADDR[b] = a;
          for (int lane = 0; lane < 8; lane++) begin
            row = (a % WORDS_PER_COLUMN) * 8 + lane;
            if (col < N && row < N)
              WEIGHT_WR_DATA[b][lane*8 +: 8] = weights[col][row];
            else
              WEIGHT_WR_DATA[b][lane*8 +: 8] = $random; // unused byte, should be ignored
          end
        end
      end
      @(negedge CLK);
      WEIGHT_WR_EN   = 0;
      WEIGHT_WR_ADDR = 0;
      WEIGHT_WR_DATA = 0;
    end
  endtask

  //-------------------------------------------------------------------------//
  // run one tile and check the results
  //-------------------------------------------------------------------------//
  task run_tile;
    integer stall_start, active_start, cycles, stalls, active;
    begin
      // clear bookkeeping
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
          load_count[i][j] = 0;
      for (int t = 0; t < M; t++)
        for (int i = 0; i < N; i++)
          got_output[t][i] = 0;
      load_cycles  = 0;
      swap_cycles  = 0;
      outputs_seen = 0;
      stall_start  = stall_count;
      active_start = active_count;
      cycles       = 0;

      $display("---- tile %0d start ----", tile);
      @(negedge CLK);
      EN = 1;

      while (state != DONE) begin
        @(negedge CLK);

        // drive the activation vector during RUN, junk otherwise (should be ignored)
        for (int j = 0; j < N; j++) begin
          if (state == RUN && dut.controller_u.run_count < M)
            ACTIVATION_IN[j] = acts[dut.controller_u.run_count][j];
          else
            ACTIVATION_IN[j] = $random;
        end

        // randomly drop EN during RUN and DRAIN
        if (state == RUN || state == DRAIN)
          EN = ($urandom % 4 != 0);
        else
          EN = 1;

        cycles = cycles + 1;
        if (cycles > 20000) begin
          $display("ERROR: timeout, stuck in state %0d", state);
          $finish;
        end
      end

      // go back to IDLE
      EN = 0;
      @(negedge CLK);
      if (state != IDLE) begin
        $display("ERROR: did not go back to IDLE after DONE (state %0d)", state);
        errors = errors + 1;
      end

      // end of tile checks
      stalls = stall_count - stall_start;
      active = active_count - active_start;
      $display("tile %0d: stalls = %0d (expected %0d), active = %0d (expected %0d), outputs = %0d / %0d",
               tile, stalls, EXPECTED_STALLS, active, M, outputs_seen, N * M);

      if (stalls == 0) begin
        $display("ERROR: no stalls were counted");
        errors = errors + 1;
      end
      if (stalls != load_cycles) begin
        $display("ERROR: stall_count went up by %0d but LOAD_WEIGHTS lasted %0d cycles", stalls, load_cycles);
        errors = errors + 1;
      end
      if (load_cycles != EXPECTED_STALLS) begin
        $display("ERROR: LOAD_WEIGHTS lasted %0d cycles, expected %0d", load_cycles, EXPECTED_STALLS);
        errors = errors + 1;
      end
      if (active != M) begin
        $display("ERROR: active_count went up by %0d, expected %0d", active, M);
        errors = errors + 1;
      end
      if (swap_cycles != 1) begin
        $display("ERROR: SWAP_WEIGHTS lasted %0d cycles, expected 1", swap_cycles);
        errors = errors + 1;
      end
      if (outputs_seen != N * M) begin
        $display("ERROR: only %0d of %0d outputs came out before DONE (last output needs %0d drain cycles)",
                 outputs_seen, N * M, 2 * N - 1);
        errors = errors + 1;
      end
    end
  endtask

  //-------------------------------------------------------------------------//
  // monitor: checks on every clock edge
  //-------------------------------------------------------------------------//
  always @(posedge CLK) begin
    if (RST_N) begin

      // weight loading: remember what each MAC was loaded with
      if (state == LOAD_WEIGHTS) begin
        load_cycles = load_cycles + 1;
        for (int i = 0; i < N; i++)
          for (int j = 0; j < N; j++)
            if (dut.controller_u.w_load[i][j]) begin
              load_count[i][j] = load_count[i][j] + 1;
              loaded_w[i][j]   = dut.controller_u.w_data[i][j];
            end
      end
      else if (dut.controller_u.w_load != 0) begin
        $display("ERROR: w_load high outside LOAD_WEIGHTS (state %0d)", state);
        errors = errors + 1;
      end

      // swap: every MAC should be loaded once with the right weight
      if (state == SWAP_WEIGHTS) begin
        swap_cycles = swap_cycles + 1;
        if (dut.controller_u.w_swap !== '1) begin
          $display("ERROR: w_swap not sent to every MAC");
          errors = errors + 1;
        end
        for (int i = 0; i < N; i++)
          for (int j = 0; j < N; j++) begin
            if (load_count[i][j] != 1) begin
              $display("ERROR: MAC(%0d,%0d) was loaded %0d times", i, j, load_count[i][j]);
              errors = errors + 1;
            end
            else if (loaded_w[i][j] !== weights[i][j]) begin
              $display("ERROR: MAC(%0d,%0d) got weight %0d, expected %0d", i, j, loaded_w[i][j], weights[i][j]);
              errors = errors + 1;
            end
          end
      end
      else if (dut.controller_u.w_swap != 0) begin
        $display("ERROR: w_swap high outside SWAP_WEIGHTS (state %0d)", state);
        errors = errors + 1;
      end

      // stall checks
      if (stall && state != LOAD_WEIGHTS) begin
        $display("ERROR: stall high outside LOAD_WEIGHTS (state %0d)", state);
        errors = errors + 1;
      end
      if (stall && mac_en) begin
        $display("ERROR: stall and mac_en high at the same time");
        errors = errors + 1;
      end
      if (mac_en && state != RUN && state != DRAIN) begin
        $display("ERROR: mac_en high outside RUN/DRAIN (state %0d)", state);
        errors = errors + 1;
      end

      // count MAC clocks since RUN started so we know when outputs come out
      if (state == LOAD_INPUTS)
        mac_cycles = 0;
      if (mac_en) begin
        mac_cycles    = mac_cycles + 1;
        check_outputs = 1;
      end
    end
  end

  //-------------------------------------------------------------------------//
  // output checker
  // activation vector t enters on MAC clock t, and column i's result comes
  // out of the bottom of the array N + i clocks later
  //-------------------------------------------------------------------------//
  integer t_out;
  always @(negedge CLK) begin
    if (check_outputs) begin
      check_outputs = 0;
      for (int i = 0; i < N; i++) begin
        t_out = (mac_cycles - 1) - N - i;
        if (t_out >= 0 && t_out < M) begin
          if (ARRAY_OUT[i] !== expected[t_out][i]) begin
            $display("ERROR: ARRAY_OUT[%0d] for vector %0d = %0d, expected %0d",
                     i, t_out, $signed(ARRAY_OUT[i]), expected[t_out][i]);
            errors = errors + 1;
          end
          if (!got_output[t_out][i]) begin
            got_output[t_out][i] = 1;
            outputs_seen = outputs_seen + 1;
          end
        end
      end
    end
  end

  //-------------------------------------------------------------------------//
  // main test
  //-------------------------------------------------------------------------//
  initial begin
    $display("N = %0d, M = %0d, BANKS = %0d, SRAM_DEPTH = %0d", N, M, BANKS, SRAM_DEPTH);

    RST_N          = 0;
    EN             = 0;
    ACTIVATION_IN  = 0;
    WEIGHT_WR_EN   = 0;
    WEIGHT_WR_ADDR = 0;
    WEIGHT_WR_DATA = 0;
    check_outputs  = 0;
    mac_cycles     = 0;
    load_cycles    = 0;
    swap_cycles    = 0;

    // reset
    repeat (3) @(negedge CLK);
    RST_N = 1;
    @(negedge CLK);
    if (state != IDLE || stall_count != 0 || active_count != 0) begin
      $display("ERROR: bad values after reset");
      errors = errors + 1;
    end

    // with EN low the controller should stay in IDLE
    repeat (3) @(negedge CLK);
    if (state != IDLE) begin
      $display("ERROR: left IDLE while EN was low");
      errors = errors + 1;
    end

    // run the tiles, each with new weights
    for (tile = 0; tile < NUM_TILES; tile++) begin
      make_data;
      write_sram;
      run_tile;
    end

    $display("==============================================");
    $display("final stall_count = %0d, active_count = %0d", stall_count, active_count);
    if (errors == 0)
      $display("TEST PASSED");
    else
      $display("TEST FAILED with %0d errors", errors);
    $display("==============================================");
    $finish;
  end

endmodule : tb_top