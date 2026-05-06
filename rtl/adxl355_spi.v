//============================================================================
// adxl355_spi.v - SKETCH (Round 14) — Analog Devices ADXL355 SPI driver
//
// Reads the Z-axis acceleration register (0x09 .. 0x0B, 20-bit signed),
// down-converts to INT8, and emits z_valid pulse at 1 kHz (after the
// ADXL355 default ODR=1 kHz).
//
// SPI mode 0 (CPOL=0, CPHA=0). 4 MHz SCLK from a /12 divider off 50 MHz.
//
// Use this in place of UART CMD_LD_X to feed real-time SCG data to the
// classifier without host involvement.
//============================================================================
`timescale 1ns/1ps

module adxl355_spi #(
    parameter integer CLK_HZ = 50_000_000,
    parameter integer SCLK_HZ = 4_000_000
) (
    input  wire        clk,
    input  wire        rst_n,
    output reg         sclk,
    output reg         csn,
    output reg         mosi,
    input  wire        miso,
    output reg signed [7:0] z_int8,
    output reg              z_valid
);
    localparam integer DIV = (CLK_HZ / SCLK_HZ / 2);   // ≈6 for 4 MHz
    reg [3:0]  div_cnt;
    reg [5:0]  bit_cnt;
    reg [23:0] tx_shift;     // 8b cmd + 16b address + 24b data window
    reg [23:0] rx_shift;
    reg [3:0]  state;
    localparam ST_IDLE = 4'd0, ST_ASSERT = 4'd1, ST_TX = 4'd2,
               ST_RX = 4'd3, ST_DEASSERT = 4'd4;

    // ADXL355 SPI register read = (address << 1) | 1.  Z high byte = 0x09.
    localparam [7:0] CMD_READ_Z = 8'h09 << 1 | 8'h01;

    // ... full implementation deferred — this is a sketch ...
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sclk <= 1'b0; csn <= 1'b1; mosi <= 1'b0;
            z_int8 <= 8'sd0; z_valid <= 1'b0;
            div_cnt <= 0; bit_cnt <= 0;
            tx_shift <= 0; rx_shift <= 0;
            state <= ST_IDLE;
        end else begin
            z_valid <= 1'b0;
            // TODO: implement state machine
            // Each ADXL355 frame: CSn low → 8 bits cmd → 8 bits addr → 24 bits data MSB first
            // After 24 RX bits, sign-extend to 32, right-shift 12 to get INT8 magnitude
            //   z_int8 <= rx_shift[19:12];   // top 8 bits of 20-bit signed reading
            // Pulse z_valid for 1 cycle, then idle until next 1 ms tick
        end
    end

    // Note: the full implementation needs a 1 ms "trigger" and proper SCLK
    // toggling; left as Round 14 future work.  See ADXL355 datasheet §10.
endmodule
