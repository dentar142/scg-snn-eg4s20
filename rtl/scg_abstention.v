// =============================================================================
// scg_abstention.v - Per-sample OOD abstention for the SCG SNN
// =============================================================================
// Reads the 3 output-neuron spike counters (sc0, sc1, sc2) plus the existing
// argmax-based pred from scg_snn_engine, and decides whether to emit the
// predicted class or abstain (UNK). The decision is pure combinational + 1
// register and adds no DSP, no BRAM.
//
//   margin = max(sc0,sc1,sc2) - second_max(sc0,sc1,sc2)
//   pred_o = (margin >= tau_i) ? pred_i : 2'd3       // 3 = UNK class
//
// Recommended tau_i: 8'd3 (see doc/calibration_report.md).
// At tau=3 on the 9,660-sample hold-out: cov=72.26%, sel_acc=91.20%
// (vs 77.72% baseline). Per-subject: b015 cov=90.58%/sel_acc=99.83%,
// b007 cov=67.03%/sel_acc=84.14%, b002 cov=58.81%/sel_acc=85.82%.
//
// Estimated cost on Anlogic EG4S20: ~60 LUT4, 0 DSP, 0 BRAM.
//
// NOTE: this module is OPTIONAL. The same computation can be done host-side
// from the 3 spike counters that scg_top_snn already emits over UART, with
// zero changes to the gold bitstream. Use this only if a future revision
// wants on-device abstention (e.g. battery deployment without a host).
// =============================================================================

`timescale 1ns / 1ps

module scg_abstention #(
    parameter [7:0] DEFAULT_TAU = 8'd3
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        sc_valid_i,    // pulse when sc0/sc1/sc2 are stable
    input  wire [7:0]  sc0_i,
    input  wire [7:0]  sc1_i,
    input  wire [7:0]  sc2_i,
    input  wire [1:0]  pred_i,        // existing argmax from scg_snn_engine
    input  wire [7:0]  tau_i,         // configurable threshold (default 3)

    output reg  [1:0]  pred_o,        // 0/1/2 = class, 3 = UNK
    output reg  [7:0]  margin_o,      // for telemetry / UART debug
    output reg         abstained_o,   // 1 = was abstained (pred_o==3)
    output reg         valid_o
);

    // -------------------------------------------------------------------------
    // Combinational top1 / top2 of {sc0, sc1, sc2}
    //   Stage 1: sort the first pair (a, b)
    //   Stage 2: insert c relative to the sorted pair
    // -------------------------------------------------------------------------
    wire [7:0] s01_max = (sc0_i >= sc1_i) ? sc0_i : sc1_i;
    wire [7:0] s01_min = (sc0_i >= sc1_i) ? sc1_i : sc0_i;

    wire [7:0] top1_w = (s01_max >= sc2_i) ? s01_max : sc2_i;
    // top2 = max of (s01_min, min(s01_max, sc2))
    wire [7:0] mid_w  = (s01_max >= sc2_i) ? sc2_i   : s01_max;
    wire [7:0] top2_w = (s01_min >= mid_w) ? s01_min : mid_w;

    wire [7:0] margin_w = top1_w - top2_w;          // unsigned 8-bit subtract
    wire       trust_w  = (margin_w >= tau_i);

    // -------------------------------------------------------------------------
    // Output register
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pred_o      <= 2'd0;
            margin_o    <= 8'd0;
            abstained_o <= 1'b0;
            valid_o     <= 1'b0;
        end else begin
            valid_o <= sc_valid_i;
            if (sc_valid_i) begin
                margin_o    <= margin_w;
                abstained_o <= ~trust_w;
                pred_o      <= trust_w ? pred_i : 2'd3;
            end
        end
    end

endmodule
