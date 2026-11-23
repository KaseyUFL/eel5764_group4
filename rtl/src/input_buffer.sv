//---------------------------------------------------------------------------//
// EEL5764 Computer Architecture                                             //
//---------------------------------------------------------------------------//
// Project     : Parameterized SRAM Scratchpad Array                         //
// Filename    : input_buffer.sv                                             //
// Engineers   : Evan Espino-DeJesus, Tyla Hart, Kasey Hill and James Woelke //                                                 
// Group       : #4                                                          //
// Description : Input Buffer                                                //
//---------------------------------------------------------------------------//
module input_buffer #(
  parameter N = 8                  // buffer rows 
)(   
  input  logic              CLK,   // clock
  input  logic              RST_N, // reset (active low)
  input  logic              EN,    // enable
  input  logic [N-1:0][7:0] X_IN,  // input data
  output logic [N-1:0][7:0] Y_OUT  // output data
);
  // internal signals
  logic [N-1:0][N-1:0][7:0] pipeline; // N * N * 8-bit pipeline

  //-------------------------------------------------------------------------//
  // generate N * N MAC units, i = x-coordinate, j = y-coordinate
  //-------------------------------------------------------------------------//
  for (genvar j = 0; j < N; j++) begin
    // first column of registers
    always_ff @(posedge CLK, negedge RST_N) begin
      if (!RST_N) begin
        pipeline[0][j] <= 0;
      end else if (EN) begin
        pipeline[0][j] <= X_IN[j];
      end
    end

    for (genvar i = 1; i < N; i++) begin
      always_ff @(posedge CLK, negedge RST_N) begin
        if (!RST_N) begin
          pipeline[i][j] <= 0;
        end else if (EN) begin
          pipeline[i][j] <= pipeline[i-1][j];
        end
      end
    end

    // output is input delayed by j+1 clocks
    assign Y_OUT[j] = pipeline[j][j];    
  end

endmodule : input_buffer
