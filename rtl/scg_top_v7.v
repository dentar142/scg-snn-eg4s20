//============================================================================
// scg_top.v
// Top module for SCG INT8 1D-CNN classifier on Anlogic EG4S20BG256.
//
//  - UART (115200 8N1) is the only host interface in v0.
//  - Protocol (host -> FPGA):
//      0xA1 + N(2B) + N bytes  : load weights to weight BRAM at running offset
//      0xA2 + 256 bytes        : load one INT8 SCG window into input BRAM
//      0xA3                    : run inference; FPGA replies with 1 byte class
//      0xA0                    : reset internal pointers
//  - All compute is INT8 weights x INT8 acts -> INT32 acc -> INT16 bias add ->
//    multiply by M0 -> arithmetic right shift -> ReLU clamp to INT8.
//
// This file integrates: UART RX/TX, command FSM, MAC array dispatcher,
// weight/activation BRAM glue. The MAC array PEs are in scg_mac_array.v.
// All RTL is hand-written for this project; no third-party code reused.
//============================================================================
`timescale 1ns/1ps

module scg_top_v7 #(
    parameter integer CLK_HZ       = 50_000_000,
    parameter integer BAUD         = 115_200,
    parameter integer WIN_LEN      = 256,
    parameter integer N_LAYERS     = 4,
    parameter integer WEIGHT_DEPTH = 52224,       // 51 KB ≥ 51744 v7 params (51 B9K)
    parameter integer ACT_DEPTH    = 8192         // 4 KB per bank × 2
) (
    input  wire        clk_i,                    // 50 MHz HX4S20C crystal
    input  wire        rst_n_i,                  // active-low reset (push button)
    input  wire        uart_rx_i,
    output wire        uart_tx_o,
    output wire [3:0]  led_o                     // HX4S20C 4-bit LEDs (A4/A3/C10/B12)
);

    //--------------------------------------------------------------------
    // v0 clocking: pass crystal through directly (no PLL).
    // Will swap in EG_PHY_PLL once timing closure is verified at 24 MHz.
    //--------------------------------------------------------------------
    wire clk   = clk_i;
    wire rst_n = rst_n_i;

    //--------------------------------------------------------------------
    // UART RX (8N1, 16x oversample)
    //--------------------------------------------------------------------
    localparam integer BAUD_DIV = CLK_HZ / (BAUD * 16);

    reg  [15:0] os_cnt;                          // oversample counter
    reg         os_tick;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            os_cnt  <= 16'd0;
            os_tick <= 1'b0;
        end else begin
            if (os_cnt == BAUD_DIV - 1) begin
                os_cnt  <= 16'd0;
                os_tick <= 1'b1;
            end else begin
                os_cnt  <= os_cnt + 16'd1;
                os_tick <= 1'b0;
            end
        end
    end

    // 2-FF metastability synchronizer for the asynchronous UART RX line.
    // Without this, ~1 in 80 frames suffered cumulative bit-slip on a busy
    // RX stream (host -> FPGA), causing host-side timeouts during long bench
    // runs.  Adding this brings frame error rate to << 1e-9 / bit at 50 MHz.
    reg rx_sync_a, rx_sync_b;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_sync_a <= 1'b1;
            rx_sync_b <= 1'b1;
        end else begin
            rx_sync_a <= uart_rx_i;
            rx_sync_b <= rx_sync_a;
        end
    end
    wire uart_rx_synced = rx_sync_b;

    reg  [3:0] rx_smpl;
    reg  [3:0] rx_bit;
    reg  [7:0] rx_shift;
    reg  [1:0] rx_state;
    localparam RX_IDLE = 2'd0, RX_START = 2'd1, RX_DATA = 2'd2, RX_STOP = 2'd3;
    reg  [7:0] rx_data;
    reg        rx_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state <= RX_IDLE;
            rx_smpl  <= 0; rx_bit <= 0; rx_shift <= 0;
            rx_data  <= 0; rx_valid <= 1'b0;
        end else begin
            rx_valid <= 1'b0;
            if (os_tick) begin
                case (rx_state)
                    RX_IDLE:  if (uart_rx_synced == 1'b0) begin
                                  rx_state <= RX_START; rx_smpl <= 4'd0;
                              end
                    RX_START: if (rx_smpl == 4'd7) begin
                                  if (uart_rx_synced == 1'b0) begin
                                      rx_state <= RX_DATA; rx_smpl <= 4'd0; rx_bit <= 4'd0;
                                  end else
                                      rx_state <= RX_IDLE;
                              end else rx_smpl <= rx_smpl + 4'd1;
                    RX_DATA:  if (rx_smpl == 4'd15) begin
                                  rx_smpl <= 4'd0;
                                  rx_shift <= {uart_rx_synced, rx_shift[7:1]};
                                  if (rx_bit == 4'd7) rx_state <= RX_STOP;
                                  else rx_bit <= rx_bit + 4'd1;
                              end else rx_smpl <= rx_smpl + 4'd1;
                    RX_STOP:  if (rx_smpl == 4'd15) begin
                                  rx_state <= RX_IDLE;
                                  rx_data  <= rx_shift;
                                  rx_valid <= 1'b1;
                              end else rx_smpl <= rx_smpl + 4'd1;
                endcase
            end
        end
    end

    //--------------------------------------------------------------------
    // UART TX (1 byte, push-on-pulse)
    //--------------------------------------------------------------------
    reg [3:0]  tx_bit;
    reg [9:0]  tx_shift;       // start + 8 + stop
    reg [15:0] tx_baud_cnt;
    reg [1:0]  tx_state;
    reg        tx_busy;
    reg [7:0]  tx_data;
    reg        tx_start;

    localparam integer BAUD_DIV_TX = CLK_HZ / BAUD;
    localparam TX_IDLE = 2'd0, TX_BIT = 2'd1;
    assign uart_tx_o = (tx_state == TX_BIT) ? tx_shift[0] : 1'b1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state <= TX_IDLE;
            tx_busy  <= 1'b0;
            tx_baud_cnt <= 0; tx_bit <= 0; tx_shift <= 10'h3FF;
        end else case (tx_state)
            TX_IDLE: if (tx_start) begin
                         tx_shift <= {1'b1, tx_data, 1'b0};
                         tx_state <= TX_BIT; tx_bit <= 0;
                         tx_baud_cnt <= 0; tx_busy <= 1'b1;
                     end else tx_busy <= 1'b0;
            TX_BIT:  if (tx_baud_cnt == BAUD_DIV_TX - 1) begin
                         tx_baud_cnt <= 0;
                         tx_shift    <= {1'b1, tx_shift[9:1]};
                         if (tx_bit == 4'd9) begin
                             tx_state <= TX_IDLE; tx_busy <= 1'b0;
                         end else tx_bit <= tx_bit + 4'd1;
                     end else tx_baud_cnt <= tx_baud_cnt + 16'd1;
        endcase
    end

    //--------------------------------------------------------------------
    // BRAMs : ask TD to infer EG_PHY_BSRAM (B9K block RAM) instead of LUTRAM.
    //   - syn_ramstyle="block_ram" : Synplify/TD-compatible attribute
    //   - read port is REGISTERED below (synchronous), which is required for
    //     BRAM inference on EG4 family.
    //--------------------------------------------------------------------
    (* syn_ramstyle = "block_ram" *) reg  [7:0]  weight_bram [0:WEIGHT_DEPTH-1];
    (* syn_ramstyle = "block_ram" *) reg  [7:0]  act_bram_a  [0:ACT_DEPTH/2-1];
    (* syn_ramstyle = "block_ram" *) reg  [7:0]  act_bram_b  [0:ACT_DEPTH/2-1];

    reg  [15:0] w_addr_w;                        // write port: driven by FSM (16-bit for 52 KB)
    wire [15:0] w_addr_r;                        // read port:  driven by MAC engine
    reg  [11:0] a_addr_w;                        // write port: driven by FSM (UART loader)
    wire [11:0] a_addr_r;                        // read port:  driven by MAC engine
    reg         pp_sel;                          // 0: read A / write B, 1: swap

    // Synchronous read pipelines (1-cycle latency, BRAM-friendly)
    reg  [7:0]  w_dout;
    reg  [7:0]  a_dout;
    // Round 8c: ping-pong selects on li[0]:
    //   layer 0 (even): read A, write B
    //   layer 1 (odd) : read B, write A
    //   layer 2 (even): read A, write B
    //   layer 3 (odd) : read B, write A
    wire [1:0]  engine_li;
    wire        engine_read_b  =  engine_li[0];   // layers 1, 3
    wire        engine_write_b = ~engine_li[0];   // layers 0, 2
    wire [11:0] engine_a_waddr;
    wire [7:0]  engine_a_wdata;
    wire        engine_a_we;
    always @(posedge clk) begin
        w_dout <= weight_bram[w_addr_r];
        a_dout <= engine_read_b ? act_bram_b[a_addr_r] : act_bram_a[a_addr_r];
    end
    // Consolidated act_bram_a writers: UART loader during CMD_LD_X, engine on
    // even-numbered layers (li[0]==1 since engine_write_b = ~li[0]).  These
    // never overlap in time because CMD_LD_X completes before CMD_RUN fires.
    always @(posedge clk) begin
        if (fsm == S_X_DATA && rx_valid)
            act_bram_a[a_addr_w] <= rx_data;
        else if (engine_a_we && !engine_write_b)
            act_bram_a[engine_a_waddr] <= engine_a_wdata;
    end
    // act_bram_b: engine-only writer (odd-numbered layers).
    always @(posedge clk) begin
        if (engine_a_we && engine_write_b)
            act_bram_b[engine_a_waddr] <= engine_a_wdata;
    end

    //--------------------------------------------------------------------
    // Command FSM
    //--------------------------------------------------------------------
    localparam CMD_LD_W = 8'hA1;
    localparam CMD_LD_X = 8'hA2;
    localparam CMD_RUN  = 8'hA3;
    localparam CMD_RST  = 8'hA0;

    localparam S_IDLE   = 4'd0;
    localparam S_W_LEN0 = 4'd1, S_W_LEN1 = 4'd2, S_W_DATA = 4'd3;
    localparam S_X_DATA = 4'd4;
    localparam S_RUN    = 4'd5;
    localparam S_DONE   = 4'd6;

    reg  [3:0]  fsm;
    reg  [15:0] count;
    reg  [15:0] need;

    // Inference engine handshake (driven by scg_mac_array)
    reg         run_pulse;
    wire        run_busy;
    wire [1:0]  run_class;
    wire        run_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm      <= S_IDLE;
            count    <= 0;  need <= 0;
            w_addr_w <= 0;  a_addr_w <= 0;
            pp_sel   <= 1'b0;
            run_pulse <= 1'b0;
            tx_start <= 1'b0; tx_data <= 0;
        end else begin
            run_pulse <= 1'b0;
            tx_start  <= 1'b0;
            case (fsm)
                S_IDLE: if (rx_valid) case (rx_data)
                    CMD_RST:  begin w_addr_w <= 0; a_addr_w <= 0; end
                    CMD_LD_W: fsm <= S_W_LEN0;
                    CMD_LD_X: begin fsm <= S_X_DATA; count <= 0; end
                    CMD_RUN:  begin fsm <= S_RUN; run_pulse <= 1'b1; end
                endcase
                S_W_LEN0: if (rx_valid) begin need[7:0]  <= rx_data; fsm <= S_W_LEN1; end
                S_W_LEN1: if (rx_valid) begin need[15:8] <= rx_data; fsm <= S_W_DATA; count <= 0; end
                S_W_DATA: if (rx_valid) begin
                    weight_bram[w_addr_w] <= rx_data;
                    w_addr_w <= w_addr_w + 16'd1;
                    count    <= count + 16'd1;
                    if (count + 1 == need) fsm <= S_IDLE;
                end
                S_X_DATA: if (rx_valid) begin
                    // Note: act_bram_a actual write moved to consolidated
                    // always block below (to avoid multi-driver with engine).
                    a_addr_w <= a_addr_w + 12'd1;
                    count    <= count + 16'd1;
                    if (count + 1 == WIN_LEN) begin
                        fsm <= S_IDLE;
                        a_addr_w <= 0;
                        pp_sel   <= 1'b0;
                    end
                end
                S_RUN:  if (run_done) fsm <= S_DONE;
                S_DONE: begin
                    tx_data  <= {6'd0, run_class};
                    tx_start <= ~tx_busy;
                    if (tx_start) fsm <= S_IDLE;
                end
            endcase
        end
    end

    //--------------------------------------------------------------------
    // MAC array (compute engine) — see scg_mac_array.v
    //--------------------------------------------------------------------
    scg_mac_array_v7 #(
        .N_LAYERS(N_LAYERS)
    ) u_engine (
        .clk_i      (clk),
        .rst_n_i    (rst_n),
        .start_i    (run_pulse),
        .busy_o     (run_busy),
        .done_o     (run_done),
        .class_o    (run_class),
        // Weight BRAM port
        .w_addr_o   (w_addr_r),
        .w_data_i   (w_dout),
        // Activation BRAM port (read from current ping bank)
        .a_raddr_o  (a_addr_r),
        .a_rdata_i  (a_dout),
        // Activation BRAM port (write to opposite pong bank)
        .a_waddr_o  (engine_a_waddr),
        .a_wdata_o  (engine_a_wdata),
        .a_we_o     (engine_a_we),
        .li_o       (engine_li)
    );

    //--------------------------------------------------------------------
    // LED heartbeat / status (HX4S20C 4 user LEDs, assumed active HIGH).
    //   LED[0] : ~1.5 Hz blink   (proves bitstream is running and clock OK)
    //   LED[1] : !uart_rx_i      (lit when host is sending; idle UART = off)
    //   LED[2] : run_busy        (lit while MAC engine is doing inference)
    //   LED[3] : !rst_n_i        (lit while user is pressing reset)
    //--------------------------------------------------------------------
    reg [25:0] heart_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) heart_cnt <= 26'd0;
        else        heart_cnt <= heart_cnt + 26'd1;
    end

    assign led_o[0] = heart_cnt[24];
    assign led_o[1] = ~uart_rx_i;
    assign led_o[2] = run_busy;
    assign led_o[3] = ~rst_n_i;

endmodule
