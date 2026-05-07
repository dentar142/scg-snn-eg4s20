// scg_adc_stream.v - ADC-direct streaming front-end for real-time SCG SNN.
//
// Replaces the UART RX path of scg_top_snn.v with a 5-channel parallel ADC
// sampler. Targets: 1 kHz/channel sample rate, 256-sample sliding window,
// 256-ms inference cadence (one inference per window-shift). FPGA inference
// is 3.6 ms (T=32) or 1.8 ms (T=16), so 99% of each ms is idle - room for
// power gating in the real ADC-direct deployment.
//
// Hardware target: HX4S20C board GPIO pins + external ADC chip (e.g.,
// AD7606C-16 8-channel, parallel SPI, up to 200 kSPS). Five channels of
// the 8-ch ADC carry: PVDF / PZT / ACC / PCG / ERB.
//
// PIN MAP (proposed; route via constraints/scg_top_adc.adc):
//   adc_clk_o     - 16-MHz SPI clock to ADC          (output, drive  8 mA)
//   adc_cs_n_o    - chip select, active-low          (output, drive  8 mA)
//   adc_convst_o  - conversion start pulse           (output, drive  8 mA)
//   adc_busy_i    - ADC asserts during conversion    (input, pullup)
//   adc_db_i[15:0]- 16-bit parallel data bus         (input, pullup)
//   adc_rd_n_o    - read strobe                      (output, drive  4 mA)
//
// LATENCY BUDGET (1 kHz target, 1000 us / sample budget):
//   ADC convert (8 ch)         150 us  (AD7606C-16 typ 75 us per channel,
//                                       parallel readout)
//   FIFO write to ring buffer    1 us
//   Window shift + LIF inference 1800 us (T=16, fits in 2 cycles of 1 kHz)
//   Margin                       ~50% slack for jitter / re-tries
//
// MODULE INTERFACE:
//   clk_i            : 50 MHz (board crystal, same as scg_top_snn)
//   rst_n_i          : active-low synchronous reset
//   adc_*            : pins above
//   sample_o[15:0]   : current 16-bit ADC sample (signed)
//   sample_chan_o[2:0]: channel index 0..4 of sample_o
//   sample_valid_o   : 1-cycle pulse when sample_o is fresh
//   window_ready_o   : asserts when 256-sample window per channel complete
//                      and ready for inference (one pulse per 256 ms)
//
// SCAFFOLD STATUS:
//   - Skeleton FSM only; physical ADC driver is NOT verified on real
//     hardware (no AD7606C available during SRTP).
//   - Replaces uart_rx in scg_top_snn.v; engine connection unchanged.
//   - For functional sim, see sim/tb_adc_stream.sv (testbench drives
//     synthetic 1-kHz signal through this module into the existing engine).

`default_nettype none

module scg_adc_stream #(
    parameter integer CLK_HZ      = 50_000_000,
    parameter integer SAMPLE_RATE = 1_000,        // Hz/channel
    parameter integer N_CHAN      = 5,
    parameter integer WIN_LEN     = 256,
    parameter integer ADC_BITS    = 16,
    parameter integer SPI_DIV     = 3             // CLK_HZ / 16 MHz ≈ 3
) (
    input  wire                    clk_i,
    input  wire                    rst_n_i,
    // ADC pins
    output reg                     adc_clk_o,
    output reg                     adc_cs_n_o,
    output reg                     adc_convst_o,
    input  wire                    adc_busy_i,
    input  wire [ADC_BITS-1:0]     adc_db_i,
    output reg                     adc_rd_n_o,
    // Outputs to engine
    output reg  signed [ADC_BITS-1:0] sample_o,
    output reg  [$clog2(N_CHAN):0]    sample_chan_o,
    output reg                        sample_valid_o,
    output reg                        window_ready_o
);

    // ---------------- 1 kHz sample tick -----------------
    localparam integer TICK_PERIOD = CLK_HZ / SAMPLE_RATE;   // 50_000 cycles
    reg [$clog2(TICK_PERIOD)-1:0] tick_cnt;
    wire sample_tick = (tick_cnt == TICK_PERIOD - 1);

    always @(posedge clk_i) begin
        if (!rst_n_i) tick_cnt <= 0;
        else if (sample_tick) tick_cnt <= 0;
        else tick_cnt <= tick_cnt + 1;
    end

    // ---------------- ADC capture FSM (per sample_tick) ----------------
    localparam S_IDLE  = 3'd0;
    localparam S_CONV  = 3'd1;   // assert CONVST, wait for BUSY high
    localparam S_WAIT  = 3'd2;   // wait for BUSY low (conversion done)
    localparam S_READ  = 3'd3;   // sequentially read N_CHAN words
    localparam S_DONE  = 3'd4;
    reg [2:0] state;
    reg [$clog2(N_CHAN):0] read_idx;

    // ---------------- Window completion counter ----------------
    reg [$clog2(WIN_LEN):0] win_cnt;

    always @(posedge clk_i) begin
        if (!rst_n_i) begin
            state <= S_IDLE; read_idx <= 0;
            sample_valid_o <= 0; window_ready_o <= 0;
            adc_convst_o <= 0; adc_cs_n_o <= 1; adc_rd_n_o <= 1; adc_clk_o <= 0;
            win_cnt <= 0;
        end else begin
            sample_valid_o <= 0;
            window_ready_o <= 0;
            case (state)
                S_IDLE: if (sample_tick) begin
                    adc_convst_o <= 1;
                    state <= S_CONV;
                end
                S_CONV: if (adc_busy_i) begin
                    adc_convst_o <= 0;
                    state <= S_WAIT;
                end
                S_WAIT: if (!adc_busy_i) begin
                    adc_cs_n_o <= 0;
                    adc_rd_n_o <= 0;
                    read_idx <= 0;
                    state <= S_READ;
                end
                S_READ: begin
                    sample_o       <= $signed(adc_db_i);
                    sample_chan_o  <= read_idx;
                    sample_valid_o <= 1;
                    if (read_idx == N_CHAN - 1) begin
                        adc_cs_n_o <= 1;
                        adc_rd_n_o <= 1;
                        state      <= S_DONE;
                    end else begin
                        read_idx <= read_idx + 1;
                    end
                end
                S_DONE: begin
                    if (win_cnt == WIN_LEN - 1) begin
                        win_cnt        <= 0;
                        window_ready_o <= 1;   // tell engine to run
                    end else begin
                        win_cnt <= win_cnt + 1;
                    end
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // SPI clock: divide CLK_HZ by SPI_DIV*2.  Held low when CS deasserted.
    reg [$clog2(SPI_DIV):0] spi_div_cnt;
    always @(posedge clk_i) begin
        if (!rst_n_i) begin spi_div_cnt <= 0; adc_clk_o <= 0; end
        else if (adc_cs_n_o) begin spi_div_cnt <= 0; adc_clk_o <= 0; end
        else if (spi_div_cnt == SPI_DIV - 1) begin
            adc_clk_o   <= ~adc_clk_o;
            spi_div_cnt <= 0;
        end else spi_div_cnt <= spi_div_cnt + 1;
    end

endmodule

`default_nettype wire
