//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : tb_top.sv                                                   //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //
// Group       : #4                                                          //
// Description : Testbench for the top level                                 //
//                                                                           //
// Runs NUM_TILES tiles back to back, each with new weights.                 //
//                                                                           //
// The host side (host_loop) writes a tile into the SRAM, raises             //
// WEIGHTS_READY, waits for weights_taken, then writes the next tile.        //
// With DOUBLE_BUFFER = 1 the controller loads the next tile's weights into  //
// the MAC staging registers while the array is still running the current   //
// tile, so after the first tile most of the load is hidden.                 //
//                                                                           //
// The testbench checks:                                                     //
//  - every MAC gets the right weight exactly once before each swap          //
//  - w_load only happens in LOAD_WEIGHTS (or RUN/DRAIN when double          //
//    buffering), and the swap goes to all MACs for one cycle                //
//  - stalls only happen while waiting in LOAD_WEIGHTS and get counted      //
//  - every ARRAY_OUT value matches the expected matrix product              //
//  - stall_count and active_count at the end                                //
// RANDOM_EN = 1 randomly drops EN during RUN/DRAIN to check the array       //
// pauses correctly. Use RANDOM_EN = 0 for cycle count measurements.         //
//---------------------------------------------------------------------------//
`timescale 1ns/1ps

module tb_top;

  // change these to test different sizes
  parameter N             = 32;
  parameter M             = 8;
  parameter BANKS         = 4;
  parameter NUM_TILES     = 4;
  parameter DOUBLE_BUFFER = 1; // 1 = load next tile during RUN/DRAIN, 0 = only in LOAD_WEIGHTS
  parameter RANDOM_EN     = 1; // 1 = randomly drop EN during RUN/DRAIN

  localparam WORDS_PER_COLUMN = (N + 7) / 8;
  localparam COLUMNS_PER_BANK = (N + BANKS - 1) / BANKS;
  localparam SRAM_DEPTH       = WORDS_PER_COLUMN * COLUMNS_PER_BANK;
  localparam ADDR_WIDTH       = (SRAM_DEPTH > 1) ? $clog2(SRAM_DEPTH) : 1;

  // loading a whole tile while stalled takes 2 cycles per SRAM word
  // (1 to read, 1 to load into the MACs)
  localparam FULL_LOAD_STALLS = 2 * SRAM_DEPTH;

  // state encoding from controller.sv
  localparam IDLE         = 0;
  localparam LOAD_WEIGHTS = 1;
  localparam SWAP_WEIGHTS = 2;
  localparam LOAD_INPUTS  = 3;
  localparam RUN          = 4;
  localparam DRAIN        = 5;
  localparam DONE         = 6;

  // dut signals
  logic                             CLK;
  logic                             RST_N;
  logic                             EN;
  logic                             WEIGHTS_READY;
  logic                             weights_taken;
  logic [N-1:0][7:0]                ACTIVATION_IN;
  logic signed [N-1:0][31:0]        ARRAY_OUT;
  logic [BANKS-1:0]                 WEIGHT_WR_EN;
  logic [BANKS-1:0][ADDR_WIDTH-1:0] WEIGHT_WR_ADDR;
  logic [BANKS-1:0][63:0]           WEIGHT_WR_DATA;
  logic [31:0]                      stall_count;
  logic [31:0]                      active_count;

  top #(
    .N             (N),
    .M             (M),
    .BANKS         (BANKS),
    .DOUBLE_BUFFER (DOUBLE_BUFFER)
  ) dut (
    .CLK            (CLK),
    .RST_N          (RST_N),
    .EN             (EN),
    .WEIGHTS_READY  (WEIGHTS_READY),
    .weights_taken  (weights_taken),
    .ACTIVATION_IN  (ACTIVATION_IN),
    .ARRAY_OUT      (ARRAY_OUT),
    .WEIGHT_WR_EN   (WEIGHT_WR_EN),
    .WEIGHT_WR_ADDR (WEIGHT_WR_ADDR),
    .WEIGHT_WR_DATA (WEIGHT_WR_DATA),
    .stall_count    (stall_count),
    .active_count   (active_count)
  );

  // 100 MHz clock
  initial CLK = 0;
  always #5 CLK = ~CLK;

  // signals inside the controller (dut.controller_u)
  wire [2:0] state  = dut.controller_u.state;
  wire       stall  = dut.controller_u.stall;
  wire       mac_en = dut.controller_u.mac_en;

  // test data for every tile
  logic signed [7:0] weights  [NUM_TILES][N][N]; // weights[t][i][j] goes to MAC(i,j)
  logic signed [7:0] acts     [NUM_TILES][M][N]; // acts[t][v][j] is activation vector v
  integer            expected [NUM_TILES][M][N]; // expected[t][v][i] = sum of weights[t][i][j] * acts[t][v][j]

  // bookkeeping
  integer            load_count  [N][N];  // times each MAC was loaded since the last swap
  logic signed [7:0] loaded_w    [N][N];  // last weight each MAC was loaded with
  logic              got_output  [M][N];
  integer            outputs_seen;
  integer            swap_count;          // tile running in the array = swap_count - 1
  integer            tile_stalls [NUM_TILES];
  integer            stalls_at_last_swap;
  integer            load_cycles;
  integer            bg_load_cycles;      // cycles w_load was high during RUN/DRAIN
  integer            mac_cycles;
  integer            cycles;
  logic              check_outputs;

  integer errors = 0;

  //-------------------------------------------------------------------------//
  // make random weights and activations for every tile, and compute the
  // expected outputs
  //-------------------------------------------------------------------------//
  task make_data;
    begin
      for (int t = 0; t < NUM_TILES; t++) begin
        for (int i = 0; i < N; i++)
          for (int j = 0; j < N; j++)
            weights[t][i][j] = $random;

        for (int v = 0; v < M; v++)
          for (int j = 0; j < N; j++)
            acts[t][v][j] = $random;

        for (int v = 0; v < M; v++)
          for (int i = 0; i < N; i++) begin
            expected[t][v][i] = 0;
            for (int j = 0; j < N; j++)
              expected[t][v][i] = expected[t][v][i] + weights[t][i][j] * acts[t][v][j];
          end
      end
    end
  endtask

  //-------------------------------------------------------------------------//
  // write tile t's weights into SRAM
  // column i goes to bank (i % BANKS), row j goes to byte lane (j % 8)
  // address = (i / BANKS) * WORDS_PER_COLUMN + (j / 8)
  //-------------------------------------------------------------------------//
  task write_sram(input integer t);
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
              WEIGHT_WR_DATA[b][lane*8 +: 8] = weights[t][col][row];
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
  // host: after the controller takes a tile, write the next one
  // (the SRAM only holds one tile, so it can't be written any earlier)
  //-------------------------------------------------------------------------//
  task host_loop;
    begin
      for (int t = 1; t <= NUM_TILES; t++) begin
        @(negedge CLK);
        while (!weights_taken) @(negedge CLK);
        @(negedge CLK);          // let the controller see the last word
        WEIGHTS_READY = 0;
        if (t < NUM_TILES) begin
          write_sram(t);
          WEIGHTS_READY = 1;
        end
      end
    end
  endtask

  //-------------------------------------------------------------------------//
  // check that the tile that just finished sent out all of its outputs
  //-------------------------------------------------------------------------//
  task check_tile_outputs;
    begin
      if (outputs_seen != N * M) begin
        $display("ERROR: tile %0d only sent %0d of %0d outputs", swap_count - 1, outputs_seen, N * M);
        errors = errors + 1;
      end
      outputs_seen = 0;
      for (int v = 0; v < M; v++)
        for (int i = 0; i < N; i++)
          got_output[v][i] = 0;
    end
  endtask

  //-------------------------------------------------------------------------//
  // monitor: checks on every clock edge
  //-------------------------------------------------------------------------//
  always @(posedge CLK) begin
    if (RST_N) begin

      if (state == LOAD_WEIGHTS)
        load_cycles = load_cycles + 1;

      // weight loading: remember what each MAC was loaded with
      if (dut.controller_u.w_load != 0) begin
        if (state == RUN || state == DRAIN) begin
          bg_load_cycles = bg_load_cycles + 1;
          if (!DOUBLE_BUFFER) begin
            $display("ERROR: w_load high during RUN/DRAIN with DOUBLE_BUFFER = 0");
            errors = errors + 1;
          end
        end
        else if (state != LOAD_WEIGHTS) begin
          $display("ERROR: w_load high in state %0d", state);
          errors = errors + 1;
        end
        for (int i = 0; i < N; i++)
          for (int j = 0; j < N; j++)
            if (dut.controller_u.w_load[i][j]) begin
              load_count[i][j] = load_count[i][j] + 1;
              loaded_w[i][j]   = dut.controller_u.w_data[i][j];
            end
      end

      // swap: every MAC should have been loaded once with the next tile's weight
      if (state == SWAP_WEIGHTS) begin
        if (dut.controller_u.w_swap !== '1) begin
          $display("ERROR: w_swap not sent to every MAC");
          errors = errors + 1;
        end
        for (int i = 0; i < N; i++)
          for (int j = 0; j < N; j++) begin
            if (load_count[i][j] != 1) begin
              $display("ERROR: tile %0d MAC(%0d,%0d) was loaded %0d times", swap_count, i, j, load_count[i][j]);
              errors = errors + 1;
            end
            else if (loaded_w[i][j] !== weights[swap_count][i][j]) begin
              $display("ERROR: tile %0d MAC(%0d,%0d) got weight %0d, expected %0d",
                       swap_count, i, j, loaded_w[i][j], weights[swap_count][i][j]);
              errors = errors + 1;
            end
            load_count[i][j] = 0;
          end

        // the previous tile should be finished
        if (swap_count > 0)
          check_tile_outputs;

        // stalls this tile waited for its weights
        tile_stalls[swap_count] = stall_count - stalls_at_last_swap;
        stalls_at_last_swap     = stall_count;
        swap_count              = swap_count + 1;
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
  // activation vector v enters on MAC clock v, and column i's result comes
  // out of the bottom of the array N + i clocks later
  //-------------------------------------------------------------------------//
  integer v_out;
  always @(negedge CLK) begin
    if (check_outputs) begin
      check_outputs = 0;
      for (int i = 0; i < N; i++) begin
        v_out = (mac_cycles - 1) - N - i;
        if (v_out >= 0 && v_out < M) begin
          if (ARRAY_OUT[i] !== expected[swap_count - 1][v_out][i]) begin
            $display("ERROR: tile %0d ARRAY_OUT[%0d] for vector %0d = %0d, expected %0d",
                     swap_count - 1, i, v_out, $signed(ARRAY_OUT[i]), expected[swap_count - 1][v_out][i]);
            errors = errors + 1;
          end
          if (!got_output[v_out][i]) begin
            got_output[v_out][i] = 1;
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
    $display("N = %0d, M = %0d, BANKS = %0d, SRAM_DEPTH = %0d, DOUBLE_BUFFER = %0d, RANDOM_EN = %0d",
             N, M, BANKS, SRAM_DEPTH, DOUBLE_BUFFER, RANDOM_EN);

    RST_N               = 0;
    EN                  = 0;
    WEIGHTS_READY       = 0;
    ACTIVATION_IN       = 0;
    WEIGHT_WR_EN        = 0;
    WEIGHT_WR_ADDR      = 0;
    WEIGHT_WR_DATA      = 0;
    check_outputs       = 0;
    mac_cycles          = 0;
    load_cycles         = 0;
    bg_load_cycles      = 0;
    swap_count          = 0;
    outputs_seen        = 0;
    stalls_at_last_swap = 0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        load_count[i][j] = 0;
    for (int v = 0; v < M; v++)
      for (int i = 0; i < N; i++)
        got_output[v][i] = 0;

    // reset
    repeat (3) @(negedge CLK);
    RST_N = 1;
    @(negedge CLK);
    if (state != IDLE || stall_count != 0 || active_count != 0 || weights_taken) begin
      $display("ERROR: bad values after reset");
      errors = errors + 1;
    end

    // with EN low the controller should stay in IDLE
    repeat (3) @(negedge CLK);
    if (state != IDLE) begin
      $display("ERROR: left IDLE while EN was low");
      errors = errors + 1;
    end

    // first tile goes into the SRAM before we start
    make_data;
    write_sram(0);
    WEIGHTS_READY = 1;
    fork
      host_loop;
    join_none

    // run until the controller says there are no more tiles
    @(negedge CLK);
    EN     = 1;
    cycles = 0;
    while (state != DONE) begin
      @(negedge CLK);

      // drive the activation vector during RUN, junk otherwise (should be ignored)
      for (int j = 0; j < N; j++) begin
        if (state == RUN && dut.controller_u.run_count < M)
          ACTIVATION_IN[j] = acts[swap_count - 1][dut.controller_u.run_count][j];
        else
          ACTIVATION_IN[j] = $random;
      end

      // randomly drop EN during RUN and DRAIN
      if (RANDOM_EN && (state == RUN || state == DRAIN))
        EN = ($urandom % 4 != 0);
      else
        EN = 1;

      cycles = cycles + 1;
      if (cycles > 100000) begin
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
    check_tile_outputs;

    // end of test checks
    if (swap_count != NUM_TILES) begin
      $display("ERROR: only %0d of %0d tiles were run", swap_count, NUM_TILES);
      errors = errors + 1;
    end
    if (active_count != NUM_TILES * M) begin
      $display("ERROR: active_count = %0d, expected %0d", active_count, NUM_TILES * M);
      errors = errors + 1;
    end
    if (stall_count != load_cycles) begin
      $display("ERROR: stall_count = %0d but LOAD_WEIGHTS lasted %0d cycles", stall_count, load_cycles);
      errors = errors + 1;
    end
    for (int t = 0; t < swap_count; t++) begin
      $display("tile %0d: stalls = %0d (full load = %0d)", t, tile_stalls[t], FULL_LOAD_STALLS);
      // the first tile (and every tile without double buffering) does a full load
      if ((t == 0 || !DOUBLE_BUFFER) && tile_stalls[t] != FULL_LOAD_STALLS) begin
        $display("ERROR: tile %0d stalled %0d cycles, expected %0d", t, tile_stalls[t], FULL_LOAD_STALLS);
        errors = errors + 1;
      end
      // with double buffering a tile should never stall more than a full load
      if (tile_stalls[t] > FULL_LOAD_STALLS) begin
        $display("ERROR: tile %0d stalled %0d cycles, more than a full load", t, tile_stalls[t]);
        errors = errors + 1;
      end
    end
    if (DOUBLE_BUFFER && NUM_TILES > 1 && bg_load_cycles == 0) begin
      $display("ERROR: double buffering is on but no weights were loaded during RUN/DRAIN");
      errors = errors + 1;
    end

    $display("final stall_count = %0d, active_count = %0d, total cycles = %0d, background loads = %0d",
             stall_count, active_count, cycles, bg_load_cycles);
    $display("utilization = %0.1f%% (active / (active + stall))",
             100.0 * active_count / (active_count + stall_count));
    $display("RESULT N=%0d BANKS=%0d DB=%0d DEPTH=%0d STALLS=%0d ACTIVE=%0d CYCLES=%0d T0=%0d TLAST=%0d",
             N, BANKS, DOUBLE_BUFFER, SRAM_DEPTH, stall_count, active_count, cycles,
             tile_stalls[0], tile_stalls[NUM_TILES - 1]);
    if (errors == 0)
      $display("TEST PASSED");
    else
      $display("TEST FAILED with %0d errors", errors);
    $finish;
  end

endmodule : tb_top
