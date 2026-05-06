//============================================================================
// scg_top_snn.v
// Top module for SCG INT8 SNN classifier (256→64→3 LIF) on Anlogic EG4S20BG256.
//
//  - UART (115200 8N1) is the only host interface.
//  - Protocol (host -> FPGA):
//      0xA2 + 256 bytes  : load one INT8 SCG window into input BRAM
//      0xA3              : run inference; FPGA replies with 1 byte class
//      0xA0              : reset internal pointers
//  - W1 / W2 ROMs are baked into the bitstream via $readmemh
//      (W1 = 16384 B in BRAM32K, W2 = 192 B in distributed/B9K).
//  - All compute is multiplication-free for FC2 (binary spike fan-in);
//    FC1 uses a single INT8×INT8 MAC.
//
// All RTL is hand-written for this project; no third-party code reused.
//============================================================================
`timescale 1ns/1ps

module scg_top_snn #(
    parameter integer CLK_HZ      = 50_000_000,
    parameter integer BAUD        = 115_200,
    parameter integer WIN_LEN     = 256,
    parameter integer H           = 64,
    parameter integer N_CLASSES   = 3,
    parameter integer T           = 32,
    parameter integer LEAK_SHIFT  = 4,
    parameter signed [23:0] THETA1 = 24'sd44012,    // from rtl/weights_snn/meta.json (holdout-trained)
    parameter signed [23:0] THETA2 = 24'sd660
) (
    input  wire        clk_i,
    input  wire        rst_n_i,
    input  wire        uart_rx_i,
    output wire        uart_tx_o,
    output wire [3:0]  led_o
);

    wire clk   = clk_i;
    wire rst_n = rst_n_i;

    //--------------------------------------------------------------------
    // UART RX (8N1, 16x oversample) — copied verbatim from scg_top_v7
    //--------------------------------------------------------------------
    localparam integer BAUD_DIV = CLK_HZ / (BAUD * 16);
    reg  [15:0] os_cnt;
    reg         os_tick;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin os_cnt <= 0; os_tick <= 0; end
        else if (os_cnt == BAUD_DIV - 1) begin os_cnt <= 0; os_tick <= 1; end
        else begin os_cnt <= os_cnt + 1; os_tick <= 0; end
    end

    reg rx_sync_a, rx_sync_b;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin rx_sync_a <= 1'b1; rx_sync_b <= 1'b1; end
        else begin rx_sync_a <= uart_rx_i; rx_sync_b <= rx_sync_a; end
    end
    wire uart_rx_synced = rx_sync_b;

    reg  [3:0] rx_smpl, rx_bit;
    reg  [7:0] rx_shift;
    reg  [1:0] rx_state;
    localparam RX_IDLE=2'd0, RX_START=2'd1, RX_DATA=2'd2, RX_STOP=2'd3;
    reg  [7:0] rx_data;
    reg        rx_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state <= RX_IDLE; rx_smpl <= 0; rx_bit <= 0;
            rx_shift <= 0; rx_data <= 0; rx_valid <= 0;
        end else begin
            rx_valid <= 1'b0;
            if (os_tick) case (rx_state)
                RX_IDLE:  if (uart_rx_synced == 1'b0) begin rx_state <= RX_START; rx_smpl <= 0; end
                RX_START: if (rx_smpl == 4'd7) begin
                              if (uart_rx_synced == 1'b0) begin rx_state <= RX_DATA; rx_smpl <= 0; rx_bit <= 0; end
                              else rx_state <= RX_IDLE;
                          end else rx_smpl <= rx_smpl + 1;
                RX_DATA:  if (rx_smpl == 4'd15) begin
                              rx_smpl <= 0;
                              rx_shift <= {uart_rx_synced, rx_shift[7:1]};
                              if (rx_bit == 4'd7) rx_state <= RX_STOP;
                              else rx_bit <= rx_bit + 1;
                          end else rx_smpl <= rx_smpl + 1;
                RX_STOP:  if (rx_smpl == 4'd15) begin
                              rx_state <= RX_IDLE; rx_data <= rx_shift; rx_valid <= 1'b1;
                          end else rx_smpl <= rx_smpl + 1;
            endcase
        end
    end

    //--------------------------------------------------------------------
    // UART TX
    //--------------------------------------------------------------------
    reg [3:0]  tx_bit;
    reg [9:0]  tx_shift;
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
            tx_state <= TX_IDLE; tx_busy <= 0; tx_baud_cnt <= 0;
            tx_bit <= 0; tx_shift <= 10'h3FF;
        end else case (tx_state)
            TX_IDLE: if (tx_start) begin
                         tx_shift <= {1'b1, tx_data, 1'b0};
                         tx_state <= TX_BIT; tx_bit <= 0;
                         tx_baud_cnt <= 0; tx_busy <= 1;
                     end else tx_busy <= 0;
            TX_BIT:  if (tx_baud_cnt == BAUD_DIV_TX - 1) begin
                         tx_baud_cnt <= 0;
                         tx_shift <= {1'b1, tx_shift[9:1]};
                         if (tx_bit == 4'd9) begin tx_state <= TX_IDLE; tx_busy <= 0; end
                         else tx_bit <= tx_bit + 1;
                     end else tx_baud_cnt <= tx_baud_cnt + 1;
        endcase
    end

    //--------------------------------------------------------------------
    // BRAMs
    //   * x_bram   :  256 bytes input (registered read; loaded by UART)
    //   * w1_rom   : 16384 bytes (H × N_IN), $readmemh from rtl/weights_snn/W1.hex
    //   * w2_rom   :   192 bytes (N_CLASSES × H), $readmemh from W2.hex
    //--------------------------------------------------------------------
    (* syn_ramstyle = "block_ram" *) reg  [7:0] x_bram   [0:WIN_LEN-1];
    (* syn_ramstyle = "block_ram" *) reg  [7:0] w1_rom   [0:H*WIN_LEN-1];
    (* syn_ramstyle = "block_ram" *) reg  [7:0] w2_rom   [0:N_CLASSES*H-1];

    // Synthesis runs from build_snn/, so the hex files are one dir up.
    initial begin
        $readmemh("../rtl/weights_snn/W1.hex", w1_rom);
        $readmemh("../rtl/weights_snn/W2.hex", w2_rom);
    end

    // SNN engine read addresses and buses
    wire [$clog2(WIN_LEN)-1:0]   eng_x_addr;
    wire [$clog2(H*WIN_LEN)-1:0] eng_w1_addr;
    wire [$clog2(N_CLASSES*H)-1:0] eng_w2_addr;

    reg  signed [7:0]  x_dout, w1_dout, w2_dout;
    always @(posedge clk) begin
        x_dout  <= x_bram[eng_x_addr];
        w1_dout <= w1_rom[eng_w1_addr];
        w2_dout <= w2_rom[eng_w2_addr];
    end

    //--------------------------------------------------------------------
    // Command FSM
    //--------------------------------------------------------------------
    localparam CMD_LD_X = 8'hA2, CMD_RUN = 8'hA3, CMD_RST = 8'hA0;
    localparam S_IDLE=3'd0, S_X_DATA=3'd1, S_RUN=3'd2, S_DONE=3'd3;

    reg  [2:0]  fsm;
    reg  [8:0]  count;        // 0..256
    reg  [7:0]  x_waddr;
    reg         run_pulse;
    wire        run_done;
    wire [1:0]  run_class;
    wire [7:0]  sc0, sc1, sc2;

    // x_bram write port (UART loader)
    always @(posedge clk) begin
        if (fsm == S_X_DATA && rx_valid) x_bram[x_waddr] <= rx_data;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm <= S_IDLE; count <= 0; x_waddr <= 0;
            run_pulse <= 0; tx_start <= 0; tx_data <= 0;
        end else begin
            run_pulse <= 1'b0;
            tx_start  <= 1'b0;
            case (fsm)
                S_IDLE: if (rx_valid) case (rx_data)
                    CMD_RST:  begin x_waddr <= 0; count <= 0; end
                    CMD_LD_X: begin fsm <= S_X_DATA; count <= 0; x_waddr <= 0; end
                    CMD_RUN:  begin fsm <= S_RUN;   run_pulse <= 1'b1; end
                endcase
                S_X_DATA: if (rx_valid) begin
                    x_waddr <= x_waddr + 1;
                    count   <= count + 1;
                    if (count + 1 == WIN_LEN) begin
                        fsm <= S_IDLE; x_waddr <= 0;
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
    // SNN engine
    //--------------------------------------------------------------------
    scg_snn_engine #(
        .N_IN(WIN_LEN), .H(H), .N_CLASSES(N_CLASSES),
        .T(T), .LEAK_SHIFT(LEAK_SHIFT)
    ) u_engine (
        .clk      (clk),
        .rst_n    (rst_n),
        .start_i  (run_pulse),
        .done_o   (run_done),

        .x_addr_o (eng_x_addr),
        .x_data_i (x_dout),

        .w1_addr_o(eng_w1_addr),
        .w1_data_i(w1_dout),

        .w2_addr_o(eng_w2_addr),
        .w2_data_i(w2_dout),

        .theta1_i (THETA1),
        .theta2_i (THETA2),

        .pred_o   (run_class),
        .sc0_o    (sc0),
        .sc1_o    (sc1),
        .sc2_o    (sc2)
    );

    //--------------------------------------------------------------------
    // LEDs
    //--------------------------------------------------------------------
    reg [25:0] heart_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) heart_cnt <= 0;
        else        heart_cnt <= heart_cnt + 1;
    end
    assign led_o[0] = heart_cnt[24];
    assign led_o[1] = ~uart_rx_i;
    assign led_o[2] = (fsm == S_RUN);
    assign led_o[3] = ~rst_n_i;

endmodule
