//============================================================================
// scg_mac_array.v
// 4-PE INT8 MAC array + layer sequencer for the SCG-CNN.
// Original RTL written for this project — uses Anlogic EG_PHY_MULT primitive.
//
// Pipeline per output element:
//   for k = 0..K-1
//     for ci = 0..C_in-1
//       acc[co, x] += a[ci, x*S + k] * w[co, ci, k]   (4 co's in parallel)
//   acc += bias[co]
//   y = sat_int8( (acc * M0) >>> shift )                 (ReLU folded)
//
// For v0 we keep the per-layer constants embedded as ROM (LUT) instead of
// loading them via UART. Adjust LAYER_PARAMS_* below to match export_weights.py
//============================================================================
`timescale 1ns/1ps

module scg_mac_array #(
    parameter integer WIN_LEN  = 256,
    parameter integer N_LAYERS = 4,
    parameter integer NUM_PE   = 4
) (
    input  wire        clk_i,
    input  wire        rst_n_i,
    input  wire        start_i,
    output reg         busy_o,
    output reg         done_o,
    output reg [1:0]   class_o,

    // Weight BRAM port
    output reg  [10:0] w_addr_o,
    input  wire [7:0]  w_data_i,

    // Activation BRAM read port (current input)
    output reg  [10:0] a_raddr_o,
    input  wire [7:0]  a_rdata_i,

    // Activation BRAM write port (next-layer output)
    output reg  [10:0] a_waddr_o,
    output reg  [7:0]  a_wdata_o,
    output reg         a_we_o,
    // Round 8c: expose current layer index so scg_top can ping-pong banks.
    // li[0] selects read/write bank: read = (li[0]==0 ? a : b),
    // write = (li[0]==0 ? b : a).
    output wire [1:0]  li_o
);
    assign li_o = li;
    //--------------------------------------------------------------------
    // Per-layer hyper-parameters (replace with values from export_weights.py)
    //--------------------------------------------------------------------
    // L0: 1->8,  K=5,  L=256 (no pool inside conv; pool happens after)
    // L1: 8->16, K=5,  L=128
    // L2: 16->16,K=5,  L=64
    // L3: 16->3, K=1,  L=32
    function automatic [7:0] L_CIN  (input [1:0] li);
        case (li) 2'd0: L_CIN  = 8'd1;
                  2'd1: L_CIN  = 8'd8;
                  2'd2: L_CIN  = 8'd16;
                  2'd3: L_CIN  = 8'd16;
                  default: L_CIN = 8'd0;
        endcase
    endfunction
    function automatic [7:0] L_COUT (input [1:0] li);
        case (li) 2'd0: L_COUT = 8'd8;
                  2'd1: L_COUT = 8'd16;
                  2'd2: L_COUT = 8'd16;
                  2'd3: L_COUT = 8'd3;
                  default: L_COUT = 8'd0;
        endcase
    endfunction
    function automatic [7:0] L_K (input [1:0] li);
        case (li) 2'd0: L_K = 8'd5;
                  2'd1: L_K = 8'd5;
                  2'd2: L_K = 8'd5;
                  2'd3: L_K = 8'd1;
                  default: L_K = 8'd0;
        endcase
    endfunction
    function automatic [9:0] L_LEN (input [1:0] li);
        case (li) 2'd0: L_LEN = 10'd256;
                  2'd1: L_LEN = 10'd128;
                  2'd2: L_LEN = 10'd64;
                  2'd3: L_LEN = 10'd32;
                  default: L_LEN = 10'd0;
        endcase
    endfunction

    // Same-padding offset: pad = K // 2 = 2 for K=5 layers, 0 for K=1 layer.
    // Round 8a fix: a_addr_calc must use (x + k - pad), masking out invalid
    // positions (< 0 or >= L_in) by gating the activation byte to 0.
    function automatic [3:0] L_PAD (input [1:0] li);
        case (li) 2'd0, 2'd1, 2'd2: L_PAD = 4'd2;  // K=5
                  2'd3:             L_PAD = 4'd0;  // K=1
                  default:          L_PAD = 4'd0;
        endcase
    endfunction

    // Next layer's input length = L_LEN(li+1).  We write the conv output of
    // li at stride L_LEN_NEXT so the next layer's reads (which use its own
    // L_LEN as stride) align.  L3 has no "next" so we keep its stride =
    // L_LEN(3) = 32 — those writes feed GAP via gap_acc, not BRAM.
    function automatic [9:0] L_LEN_NEXT (input [1:0] li);
        case (li) 2'd0: L_LEN_NEXT = 10'd128;   // L1 input length
                  2'd1: L_LEN_NEXT = 10'd64;
                  2'd2: L_LEN_NEXT = 10'd32;
                  2'd3: L_LEN_NEXT = 10'd32;
                  default: L_LEN_NEXT = 10'd0;
        endcase
    endfunction
    // Per-layer requantization shift & M0 — filled from export_weights.py
    // (full 20-record CEBS run; PyTorch FP=84.50%, INT8 golden=84.60%,
    //  91.9% agreement, essentially zero quantization loss).
    function automatic [4:0] L_SHIFT (input [1:0] li);
        case (li) 2'd0: L_SHIFT = 5'd16;
                  2'd1: L_SHIFT = 5'd15;
                  2'd2: L_SHIFT = 5'd15;
                  2'd3: L_SHIFT = 5'd13;
                  default: L_SHIFT = 5'd0;
        endcase
    endfunction
    function automatic signed [15:0] L_M0 (input [1:0] li);
        case (li) 2'd0: L_M0 = 16'sd390;
                  2'd1: L_M0 = 16'sd414;
                  2'd2: L_M0 = 16'sd269;
                  2'd3: L_M0 = 16'sd319;
                  default: L_M0 = 16'sd0;
        endcase
    endfunction

    // Per-output-channel bias ROM (INT16, signed) — values come straight from
    // export_weights.py L*_b.mem files. Bias is added to acc *before* the
    // (acc * M0) >>> shift requantization step, exactly matching golden_model.py.
    //
    // Implementation: FLAT 6-bit index across all 43 outputs (8+16+16+3) to
    // avoid nested-case synthesis issues observed with TD 6.2.x where only
    // the largest-magnitude bias was honored — the flat single-case form
    // synthesizes deterministically as a single LUT mux tree.
    //   L0 (8):  flat_idx 0..7
    //   L1 (16): flat_idx 8..23
    //   L2 (16): flat_idx 24..39
    //   L3 (3):  flat_idx 40..42
    function automatic signed [15:0] L_BIAS (input [1:0] li, input [4:0] co);
        reg [5:0] idx;
        begin
            case (li)
                2'd0: idx = 6'd0  + {1'b0, co};
                2'd1: idx = 6'd8  + {1'b0, co};
                2'd2: idx = 6'd24 + {1'b0, co};
                2'd3: idx = 6'd40 + {1'b0, co};
                default: idx = 6'd63;
            endcase
            case (idx)
                // L0
                6'd0:  L_BIAS = 16'shFD27;  6'd1:  L_BIAS = 16'shFE49;
                6'd2:  L_BIAS = 16'sh004B;  6'd3:  L_BIAS = 16'sh005F;
                6'd4:  L_BIAS = 16'sh01FA;  6'd5:  L_BIAS = 16'sh03F0;
                6'd6:  L_BIAS = 16'shFF76;  6'd7:  L_BIAS = 16'sh053A;
                // L1
                6'd8:  L_BIAS = 16'sh0284;  6'd9:  L_BIAS = 16'sh0316;
                6'd10: L_BIAS = 16'shFA69;  6'd11: L_BIAS = 16'shFC83;
                6'd12: L_BIAS = 16'sh0536;  6'd13: L_BIAS = 16'shFDB9;
                6'd14: L_BIAS = 16'sh017D;  6'd15: L_BIAS = 16'sh027C;
                6'd16: L_BIAS = 16'shFF7A;  6'd17: L_BIAS = 16'shFFE0;
                6'd18: L_BIAS = 16'shFEA2;  6'd19: L_BIAS = 16'sh04A8;
                6'd20: L_BIAS = 16'sh0916;  6'd21: L_BIAS = 16'shFFE2;
                6'd22: L_BIAS = 16'sh039F;  6'd23: L_BIAS = 16'shFEC7;
                // L2
                6'd24: L_BIAS = 16'sh0191;  6'd25: L_BIAS = 16'shFF84;
                6'd26: L_BIAS = 16'shFD80;  6'd27: L_BIAS = 16'sh00B8;
                6'd28: L_BIAS = 16'sh0734;  6'd29: L_BIAS = 16'shFCF1;
                6'd30: L_BIAS = 16'sh062B;  6'd31: L_BIAS = 16'sh0474;
                6'd32: L_BIAS = 16'sh01A9;  6'd33: L_BIAS = 16'sh01A1;
                6'd34: L_BIAS = 16'sh018B;  6'd35: L_BIAS = 16'sh00A6;
                6'd36: L_BIAS = 16'sh024A;  6'd37: L_BIAS = 16'sh004A;
                6'd38: L_BIAS = 16'shFBCD;  6'd39: L_BIAS = 16'shFF8D;
                // L3
                6'd40: L_BIAS = 16'sh00FD;  6'd41: L_BIAS = 16'sh00AE;
                6'd42: L_BIAS = 16'shFF61;
                default: L_BIAS = 16'sd0;
            endcase
        end
    endfunction

    //--------------------------------------------------------------------
    // FSM
    //--------------------------------------------------------------------
    localparam S_IDLE      = 4'd0,
               S_INIT      = 4'd1,
               S_FETCH     = 4'd2,
               S_FETCH_DLY = 4'd9,   // 1-cycle wait for synchronous BRAM read
               S_MAC       = 4'd3,
               S_REQ       = 4'd4,
               S_BIAS      = 4'd10,  // 1-cycle pipeline stage: acc[i] += bias
               S_WRITE     = 4'd5,
               S_NEXT      = 4'd6,
               S_GAP       = 4'd7,
               S_DONE      = 4'd8;

    reg  [3:0] state;
    reg  [1:0] li;                  // layer index (0..3)
    reg  [9:0] x_idx;               // spatial index within current layer
    reg  [7:0] co_base;             // group of NUM_PE output channels
    reg  [7:0] ci_idx;
    reg  [3:0] k_idx;
    reg  [2:0] write_idx;           // 0..NUM_PE-1: which PE to write next

    // Accumulators for NUM_PE parallel output channels (signed INT32)
    reg signed [31:0] acc [0:NUM_PE-1];

    // GAP accumulators for the final 3 classes
    reg signed [31:0] gap_acc [0:2];

    // Pre-computed addresses
    wire [10:0] w_addr_calc;
    wire [10:0] a_addr_calc;

    //--------------------------------------------------------------------
    // Address arithmetic (kept simple; weight layout = layer-major,
    //   then [co, ci, k] row-major)
    //--------------------------------------------------------------------
    reg [10:0] layer_w_base;
    reg [10:0] layer_a_in_base;     // 0 if first layer reads from input bank
    reg [10:0] layer_a_out_base;

    always @(*) begin
        case (li)
            2'd0: layer_w_base = 11'd0;
            2'd1: layer_w_base = 11'd40;       // 1*8*5
            2'd2: layer_w_base = 11'd680;      // 40 + 8*16*5
            2'd3: layer_w_base = 11'd1960;     // 680 + 16*16*5
            default: layer_w_base = 11'd0;
        endcase
    end

    assign w_addr_calc = layer_w_base
                       + (co_base + 8'd0) * L_CIN(li) * L_K(li)
                       + ci_idx * L_K(li)
                       + k_idx;

    // Same-padding: read position pos = x_idx + k_idx - pad.  When pos is
    // out of [0, L_in-1] we still issue the read but gate the activation byte
    // to 0 in the MAC step (`act_byte_eff` below).
    wire signed [10:0] pos_signed = $signed({1'b0, x_idx})
                                  + $signed({7'd0, k_idx})
                                  - $signed({7'd0, L_PAD(li)});
    wire act_valid = (pos_signed >= 11'sd0)
                  && (pos_signed < $signed({1'b0, L_LEN(li)}));
    wire [9:0] pos_eff = act_valid ? pos_signed[9:0] : 10'd0;
    assign a_addr_calc = (li == 2'd0)
                       ? (ci_idx * 11'd256 + {1'b0, pos_eff})
                       : (ci_idx * L_LEN(li) + {1'b0, pos_eff});

    //--------------------------------------------------------------------
    // PE compute (combinational multiply, registered accumulate)
    //--------------------------------------------------------------------
    // Round 8a: latch act_valid by 1 cycle to align with BRAM read latency.
    // S_FETCH samples a_addr_calc and act_valid; S_FETCH_DLY waits 1 cycle;
    // S_MAC then reads a_rdata_i (now valid) AND act_valid_d (delayed).
    reg act_valid_d;
    always @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)            act_valid_d <= 1'b0;
        else if (state == S_FETCH) act_valid_d <= act_valid;
    end
    wire signed [7:0] act_byte = act_valid_d ? a_rdata_i : 8'sd0;
    wire signed [7:0] w_byte   = w_data_i;

    // Instantiate NUM_PE EG_PHY_MULT primitives.
    // For TD 6.2.2 we can also use a `*` operator and let synthesis infer
    // the multiplier — the implementation below uses the `*` form for
    // portability. Replace with explicit EG_PHY_MULT instantiation if you
    // want guaranteed DSP18 mapping.
    wire signed [15:0] mul [0:NUM_PE-1];
    genvar gi;
    generate
        for (gi = 0; gi < NUM_PE; gi = gi + 1) begin : g_pe
            assign mul[gi] = act_byte * w_byte;     // synth -> EG_PHY_MULT
        end
    endgenerate

    integer i;
    always @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            state    <= S_IDLE;
            busy_o   <= 1'b0; done_o <= 1'b0; class_o <= 2'd0;
            li <= 0; x_idx <= 0; co_base <= 0; ci_idx <= 0; k_idx <= 0;
            write_idx <= 3'd0;
            for (i = 0; i < NUM_PE; i = i + 1) acc[i] <= 0;
            for (i = 0; i < 3;     i = i + 1) gap_acc[i] <= 0;
            w_addr_o <= 0; a_raddr_o <= 0; a_waddr_o <= 0; a_wdata_o <= 0; a_we_o <= 1'b0;
        end else begin
            done_o  <= 1'b0;
            a_we_o  <= 1'b0;
            case (state)
                S_IDLE: if (start_i) begin
                    busy_o  <= 1'b1;
                    li <= 0; x_idx <= 0; co_base <= 0; ci_idx <= 0; k_idx <= 0;
                    for (i = 0; i < 3; i = i + 1) gap_acc[i] <= 0;
                    state <= S_INIT;
                end
                S_INIT: begin
                    for (i = 0; i < NUM_PE; i = i + 1) acc[i] <= 0;
                    state <= S_FETCH;
                end
                S_FETCH: begin
                    w_addr_o  <= w_addr_calc;
                    a_raddr_o <= a_addr_calc;
                    state     <= S_FETCH_DLY;
                end
                S_FETCH_DLY: begin
                    // Wait one cycle for the synchronous BRAM read pipeline
                    state <= S_MAC;
                end
                S_MAC: begin
                    // Accumulate one tap into all NUM_PE accumulators
                    for (i = 0; i < NUM_PE; i = i + 1)
                        acc[i] <= acc[i] + {{16{mul[i][15]}}, mul[i]};
                    // Advance k, ci, co_base
                    if (k_idx + 1 < L_K(li)) begin
                        k_idx <= k_idx + 4'd1;
                        state <= S_FETCH;
                    end else begin
                        k_idx <= 0;
                        if (ci_idx + 1 < L_CIN(li)) begin
                            ci_idx <= ci_idx + 8'd1;
                            state  <= S_FETCH;
                        end else begin
                            ci_idx <= 0;
                            state  <= S_REQ;
                        end
                    end
                end
                S_REQ: begin
                    // Pipeline stage: add per-output-channel bias to all 4
                    // PE accumulators on its OWN cycle so the long
                    // (acc + bias) -> (* M0) -> (>>> shift) -> sat_int8 -> reg
                    // combinational chain in S_WRITE is split.  Both operands
                    // are signed — Verilog auto-extends without {} concat.
                    acc[0] <= acc[0] + L_BIAS(li, co_base[4:0] + 5'd0);
                    acc[1] <= acc[1] + L_BIAS(li, co_base[4:0] + 5'd1);
                    acc[2] <= acc[2] + L_BIAS(li, co_base[4:0] + 5'd2);
                    acc[3] <= acc[3] + L_BIAS(li, co_base[4:0] + 5'd3);
                    write_idx <= 3'd0;
                    state     <= S_BIAS;
                end
                S_BIAS: begin
                    // bias is now committed in the previous cycle; this state
                    // simply waits one extra cycle to ensure the new acc
                    // values are available to S_WRITE (no logic needed; the NB
                    // assignment in S_REQ takes effect at the start of S_BIAS).
                    state <= S_WRITE;
                end
                S_WRITE: begin
                    // Diagnostic: write all x with ORIGINAL stride to verify
                    // ping-pong + bias path alone (without pool changes).
                    if (co_base + {5'd0, write_idx} < L_COUT(li)) begin
                        a_waddr_o <= (co_base + {5'd0, write_idx}) * L_LEN(li) + x_idx;
                        a_wdata_o <= (li == 2'd3)
                            ? sat_int8_signed(($signed(acc[write_idx]) * $signed(L_M0(li))) >>> L_SHIFT(li))
                            : sat_int8_relu  (($signed(acc[write_idx]) * $signed(L_M0(li))) >>> L_SHIFT(li));
                        a_we_o    <= 1'b1;
                        if (li == 2'd3) begin
                            gap_acc[co_base + {5'd0, write_idx}] <=
                                gap_acc[co_base + {5'd0, write_idx}]
                                + sat_int8_signed(($signed(acc[write_idx]) * $signed(L_M0(li))) >>> L_SHIFT(li));
                        end
                    end
                    if (write_idx == NUM_PE - 1) begin
                        state <= S_NEXT;
                    end else begin
                        write_idx <= write_idx + 3'd1;
                    end
                end
                S_NEXT: begin
                    if (co_base + NUM_PE < L_COUT(li)) begin
                        co_base <= co_base + NUM_PE;
                        state   <= S_INIT;
                    end else begin
                        co_base <= 0;
                        if (x_idx + 1 < L_LEN(li)) begin
                            x_idx <= x_idx + 10'd1;
                            state <= S_INIT;
                        end else begin
                            x_idx <= 0;
                            if (li == 2'd3) state <= S_GAP;
                            else begin
                                li <= li + 2'd1;
                                state <= S_INIT;
                            end
                        end
                    end
                end
                S_GAP: begin
                    // Argmax over the 3 GAP accumulators
                    if (gap_acc[1] > gap_acc[0] && gap_acc[1] >= gap_acc[2])
                        class_o <= 2'd1;
                    else if (gap_acc[2] > gap_acc[0] && gap_acc[2] > gap_acc[1])
                        class_o <= 2'd2;
                    else
                        class_o <= 2'd0;
                    state <= S_DONE;
                end
                S_DONE: begin
                    busy_o <= 1'b0; done_o <= 1'b1;
                    state  <= S_IDLE;
                end
            endcase
        end
    end

    //--------------------------------------------------------------------
    // Saturation + ReLU helper
    //--------------------------------------------------------------------
    function automatic signed [7:0] sat_int8_relu(input signed [47:0] v);
        if (v <= 0)         sat_int8_relu = 8'sd0;
        else if (v > 127)   sat_int8_relu = 8'sd127;
        else                sat_int8_relu = v[7:0];
    endfunction

    // Symmetric INT8 saturation, no ReLU (used at the final logit layer L3)
    function automatic signed [7:0] sat_int8_signed(input signed [47:0] v);
        if (v < -128)       sat_int8_signed = -8'sd128;
        else if (v > 127)   sat_int8_signed = 8'sd127;
        else                sat_int8_signed = v[7:0];
    endfunction

endmodule
