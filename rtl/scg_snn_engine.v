// =============================================================================
// scg_snn_engine.v - INT8 LIF SNN inference for 256->H->K SCG classification
// =============================================================================
// Architecture (matches tools/sim_snn.py exactly):
//   1) FC1 precompute: I1[i] = sum_{j=0..N_IN-1} x[j] * W1[i,j]   (INT24 accum)
//   2) for t in 0..T-1:
//        v1[i]  <- (v1[i] - (v1[i] >>> LEAK_SHIFT)) + I1[i]      (i in 0..H-1)
//        s1[i]  <- (v1[i] >= theta1)
//        v1[i]  <- v1[i] - s1[i]*theta1                          (soft reset)
//        I2[c]  = sum_{i: s1[i]=1} W2[c,i]                       (binary fan-in)
//        v2[c]  <- (v2[c] - (v2[c] >>> LEAK_SHIFT)) + I2[c]
//        s2[c]  <- (v2[c] >= theta2)
//        v2[c]  <- v2[c] - s2[c]*theta2
//        sc[c]  <- sc[c] + s2[c]
//   3) pred = argmax(sc[0..N_CLASSES-1])  (generic loop over K classes)
//
// Resource budget for EG4S20: 1 INT8xINT8 DSP MAC + ~1500 LUT + 1xBRAM32K (W1)
// Latency estimate @ 50 MHz, T=32, H=64, K=5, LEAK_SHIFT=4:
//   FC1 precompute = H * N_IN = 16384 cycles  (one-time)
//   per timestep   = H (LIF1) + H*K (FC2) + K (LIF2) ~ 64+320+5 = 389 cycles
//   total ~ 16384 + 32*389 = 28832 cycles ~ 0.58 ms / inference
// =============================================================================

`timescale 1ns / 1ps

module scg_snn_engine #(
    parameter integer N_IN       = 256,
    parameter integer H          = 64,
    parameter integer N_CLASSES  = 3,
    parameter integer T          = 32,
    parameter integer LEAK_SHIFT = 4
) (
    input  wire                clk,
    input  wire                rst_n,

    // Control
    input  wire                start_i,
    output reg                 done_o,

    // Input activation BRAM (single read port, shared with CNN engine)
    output reg  [$clog2(N_IN)-1:0]    x_addr_o,
    input  wire signed [7:0]          x_data_i,

    // W1 ROM (H * N_IN bytes; address = i * N_IN + j)
    output reg  [$clog2(H*N_IN)-1:0]  w1_addr_o,
    input  wire signed [7:0]          w1_data_i,

    // W2 ROM (N_CLASSES * H bytes; address = c * H + i)
    output reg  [$clog2(N_CLASSES*H)-1:0] w2_addr_o,
    input  wire signed [7:0]          w2_data_i,

    // Thresholds (loaded by host or hardcoded at top-level)
    input  wire signed [23:0]  theta1_i,
    input  wire signed [23:0]  theta2_i,

    // Outputs (pred_o width = ceil(log2(N_CLASSES)); 2 bits for K=3, 3 bits for K=5)
    output reg  [$clog2(N_CLASSES)-1:0]   pred_o
);

    // For internal use (same as port width)
    localparam integer PRED_W = $clog2(N_CLASSES);

    // -------------------------------------------------------------------------
    // Storage
    // -------------------------------------------------------------------------
    reg signed [23:0] I1   [0:H-1];          // pre-computed FC1 input current
    reg signed [23:0] v1   [0:H-1];
    reg signed [23:0] v2   [0:N_CLASSES-1];
    reg               s1   [0:H-1];          // current spike vector
    reg        [7:0]  sc   [0:N_CLASSES-1];  // spike counters

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    localparam S_IDLE      = 4'd0,
               S_INIT      = 4'd1,
               S_FC1_FETCH = 4'd2,
               S_FC1_MAC   = 4'd3,
               S_FC1_NEXT  = 4'd4,
               S_LIF1      = 4'd5,
               S_FC2_FETCH = 4'd6,
               S_FC2_ACC   = 4'd7,
               S_FC2_NEXT  = 4'd8,
               S_LIF2      = 4'd9,
               S_TS_NEXT   = 4'd10,
               S_ARGMAX    = 4'd11,
               S_DONE      = 4'd12,
               S_AM_STEP   = 4'd13;

    reg [3:0] state;

    // FC1 loop counters
    reg [$clog2(H+1)-1:0]    fc1_i;
    reg [$clog2(N_IN+1)-1:0] fc1_j;
    reg signed [23:0]        fc1_acc;

    // Time-step counter
    reg [$clog2(T+1)-1:0]    t_idx;

    // LIF1 counter
    reg [$clog2(H+1)-1:0]    lif1_i;

    // FC2 counters (per class, sweep i over hidden neurons)
    reg [$clog2(N_CLASSES+1)-1:0] fc2_c;
    reg [$clog2(H+1)-1:0]    fc2_i;
    reg signed [23:0]        fc2_acc;

    // LIF2 counter
    reg [$clog2(N_CLASSES+1)-1:0] lif2_c;

    // Argmax loop variables
    reg [$clog2(N_CLASSES+1)-1:0] am_idx;       // sweep over classes 1..K-1
    reg [7:0]                     am_best_val;
    reg [PRED_W-1:0]              am_best_idx;

    // -------------------------------------------------------------------------
    // Main FSM
    // -------------------------------------------------------------------------
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done_o   <= 1'b0;
            pred_o   <= {PRED_W{1'b0}};
            x_addr_o <= 0; w1_addr_o <= 0; w2_addr_o <= 0;
            fc1_i <= 0; fc1_j <= 0; fc1_acc <= 0;
            t_idx <= 0; lif1_i <= 0;
            fc2_c <= 0; fc2_i <= 0; fc2_acc <= 0;
            lif2_c <= 0;
            am_idx <= 0; am_best_val <= 8'd0; am_best_idx <= {PRED_W{1'b0}};
            for (k = 0; k < H; k = k + 1) begin
                I1[k] <= 0; v1[k] <= 0; s1[k] <= 1'b0;
            end
            for (k = 0; k < N_CLASSES; k = k + 1) begin
                v2[k] <= 0; sc[k] <= 8'd0;
            end
        end else begin
            case (state)
            // ---------------------------------------------------------------
            S_IDLE: begin
                done_o <= 1'b0;
                if (start_i) begin
                    fc1_i <= 0; fc1_j <= 0; fc1_acc <= 0;
                    t_idx <= 0;
                    state <= S_INIT;
                end
            end

            S_INIT: begin
                for (k = 0; k < H; k = k + 1) begin
                    v1[k] <= 0;
                end
                for (k = 0; k < N_CLASSES; k = k + 1) begin
                    v2[k] <= 0; sc[k] <= 8'd0;
                end
                // Issue first FC1 fetch (i=0, j=0)
                x_addr_o  <= 0;
                w1_addr_o <= 0;
                state <= S_FC1_FETCH;
            end

            // ---------------------------------------------------------------
            // FC1 precompute  I1[i] = ∑_j x[j] * W1[i,j]
            //   FETCH cycle: addresses asserted last cycle, BRAMs respond now
            //   MAC cycle  : multiply-accumulate, advance j (or finish)
            // ---------------------------------------------------------------
            S_FC1_FETCH: begin
                // BRAM read latency = 1 cycle; data is valid this cycle
                state <= S_FC1_MAC;
            end

            S_FC1_MAC: begin
                fc1_acc <= fc1_acc + $signed(x_data_i) * $signed(w1_data_i);
                if (fc1_j == N_IN - 1) begin
                    state <= S_FC1_NEXT;
                end else begin
                    fc1_j     <= fc1_j + 1;
                    x_addr_o  <= fc1_j + 1;
                    w1_addr_o <= w1_addr_o + 1;
                    state     <= S_FC1_FETCH;
                end
            end

            S_FC1_NEXT: begin
                // Store this neuron's accumulated current
                I1[fc1_i] <= fc1_acc;
                fc1_acc   <= 0;
                if (fc1_i == H - 1) begin
                    // FC1 done; start time-step loop
                    lif1_i <= 0;
                    state  <= S_LIF1;
                end else begin
                    fc1_i     <= fc1_i + 1;
                    fc1_j     <= 0;
                    x_addr_o  <= 0;
                    w1_addr_o <= w1_addr_o + 1;  // start of next neuron's row
                    state     <= S_FC1_FETCH;
                end
            end

            // ---------------------------------------------------------------
            // LIF1: for each i, v1 ← (v1 - v1>>>k) + I1; spike + reset
            // ---------------------------------------------------------------
            S_LIF1: begin: lif1_blk
                reg signed [23:0] v_leak;
                reg signed [23:0] v_new;
                v_leak = v1[lif1_i] - (v1[lif1_i] >>> LEAK_SHIFT);
                v_new  = v_leak + I1[lif1_i];
                if (v_new >= theta1_i) begin
                    s1[lif1_i] <= 1'b1;
                    v1[lif1_i] <= v_new - theta1_i;
                end else begin
                    s1[lif1_i] <= 1'b0;
                    v1[lif1_i] <= v_new;
                end
                if (lif1_i == H - 1) begin
                    fc2_c     <= 0;
                    fc2_i     <= 0;
                    fc2_acc   <= 0;
                    w2_addr_o <= 0;
                    state     <= S_FC2_FETCH;
                end else begin
                    lif1_i <= lif1_i + 1;
                end
            end

            // ---------------------------------------------------------------
            // FC2: I2[c] = ∑_{i: s1[i]=1} W2[c,i]   (binary-spike fan-in)
            // ---------------------------------------------------------------
            S_FC2_FETCH: begin
                state <= S_FC2_ACC;
            end

            S_FC2_ACC: begin
                if (s1[fc2_i]) begin
                    fc2_acc <= fc2_acc + $signed(w2_data_i);
                end
                if (fc2_i == H - 1) begin
                    state <= S_FC2_NEXT;
                end else begin
                    fc2_i     <= fc2_i + 1;
                    w2_addr_o <= w2_addr_o + 1;
                    state     <= S_FC2_FETCH;
                end
            end

            S_FC2_NEXT: begin: fc2_next_blk
                reg signed [23:0] v_leak2;
                reg signed [23:0] v_new2;
                v_leak2 = v2[fc2_c] - (v2[fc2_c] >>> LEAK_SHIFT);
                v_new2  = v_leak2 + fc2_acc;
                if (v_new2 >= theta2_i) begin
                    if (sc[fc2_c] != 8'hFF) sc[fc2_c] <= sc[fc2_c] + 8'd1;
                    v2[fc2_c] <= v_new2 - theta2_i;
                end else begin
                    v2[fc2_c] <= v_new2;
                end
                if (fc2_c == N_CLASSES - 1) begin
                    state <= S_TS_NEXT;
                end else begin
                    fc2_c     <= fc2_c + 1;
                    fc2_i     <= 0;
                    fc2_acc   <= 0;
                    w2_addr_o <= w2_addr_o + 1;
                    state     <= S_FC2_FETCH;
                end
            end

            S_LIF2: begin
                // Fused into S_FC2_NEXT above
                state <= S_TS_NEXT;
            end

            // ---------------------------------------------------------------
            S_TS_NEXT: begin
                if (t_idx == T - 1) begin
                    state <= S_ARGMAX;
                end else begin
                    t_idx     <= t_idx + 1;
                    lif1_i    <= 0;
                    state     <= S_LIF1;
                end
            end

            // Generic argmax across N_CLASSES counters.
            //   S_ARGMAX  : seed best with sc[0], idx 1 -> S_AM_STEP
            //   S_AM_STEP : compare sc[am_idx] with current best; advance idx;
            //               when am_idx == N_CLASSES-1 done -> S_DONE
            S_ARGMAX: begin
                am_best_val <= sc[0];
                am_best_idx <= {PRED_W{1'b0}};
                if (N_CLASSES <= 1) begin
                    pred_o <= {PRED_W{1'b0}};
                    done_o <= 1'b1;
                    state  <= S_DONE;
                end else begin
                    am_idx <= 1;
                    state  <= S_AM_STEP;
                end
            end

            S_AM_STEP: begin
                if (sc[am_idx] > am_best_val) begin
                    am_best_val <= sc[am_idx];
                    am_best_idx <= am_idx[PRED_W-1:0];
                end
                if (am_idx == N_CLASSES - 1) begin
                    // commit final argmax
                    pred_o <= (sc[am_idx] > am_best_val) ?
                              am_idx[PRED_W-1:0] : am_best_idx;
                    done_o <= 1'b1;
                    state  <= S_DONE;
                end else begin
                    am_idx <= am_idx + 1;
                end
            end

            S_DONE: begin
                if (!start_i) state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
