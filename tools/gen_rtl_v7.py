"""gen_rtl_v7.py — generate scg_mac_array_v7.v with v5/v7-architecture constants.

Reads rtl/weights_v5/ (or v7/) and emits a fully-instantiated MAC array RTL
with hardcoded per-channel bias / M0 / shift ROMs.  Then prints a `cat <<EOF`-
style template.  The output goes to rtl/scg_mac_array_v7.v.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_int(path: Path, width_bits: int) -> list:
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        v = int(line, 16)
        if width_bits == 16 and v >= 0x8000: v -= 0x10000
        elif width_bits == 32 and v >= 0x80000000: v -= 0x100000000
        elif width_bits == 5: v &= 0x1F
        vals.append(v)
    return vals


def emit_case(name: str, vals: list, width: int, idx_width: int) -> str:
    lines = [f"    function automatic signed [{width-1}:0] {name} (input [{idx_width-1}:0] idx);"]
    lines.append(f"        case (idx)")
    for i, v in enumerate(vals):
        lines.append(f"            {idx_width}'d{i}: {name} = {width}'sh{v & ((1<<width)-1):0{(width+3)//4}X};")
    lines.append(f"            default: {name} = {width}'sd0;")
    lines.append(f"        endcase")
    lines.append(f"    endfunction")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=REPO / "rtl/weights_v7")
    p.add_argument("--out", type=Path, default=REPO / "rtl/scg_mac_array_v7.v")
    args = p.parse_args()

    meta = json.loads((args.weights / "meta.json").read_text())
    print(f"Architecture: {meta['architecture']}")
    print(f"Total weight bytes: {meta['weight_total_bytes']}")

    # Load per-layer arrays
    biases, m0s, shifts = [], [], []
    for li in range(4):
        biases  += load_int(args.weights / f"L{li}_b.mem", 32)
        m0s     += load_int(args.weights / f"L{li}_M0.mem", 16)
        shifts  += load_int(args.weights / f"L{li}_shift.mem", 5)
    print(f"Total per-channel entries: bias={len(biases)} m0={len(m0s)} shift={len(shifts)}")

    arch = meta["architecture"]   # e.g., [1, 32, 64, 128, 3]
    L_CIN  = [arch[0], arch[1], arch[2], arch[3]]
    L_COUT = [arch[1], arch[2], arch[3], arch[4]]
    L_K    = meta.get("K", [5, 5, 5, 1])
    L_LEN_IN  = meta.get("L_in",  [256, 128, 64, 32])
    L_LEN_OUT = meta.get("L_pool_out", [128, 64, 32, 32])
    L_PAD = [k // 2 for k in L_K]

    # weight base offsets
    w_offsets = [0]
    for li in range(3):
        w_offsets.append(w_offsets[-1] + L_CIN[li] * L_COUT[li] * L_K[li])

    flat_co_offsets = [0]
    for li in range(3):
        flat_co_offsets.append(flat_co_offsets[-1] + L_COUT[li])

    # Generate the V file
    s = []
    s.append("// scg_mac_array_v7.v - AUTO-GENERATED from rtl/weights_v7/")
    s.append(f"// Architecture: {arch}, K={L_K}")
    s.append(f"// In lengths : {L_LEN_IN}")
    s.append(f"// Out lengths: {L_LEN_OUT}  (stride-2 conv if In==2*Out)")
    s.append(f"// Total weight bytes: {meta['weight_total_bytes']}")
    s.append("")
    s.append("`timescale 1ns/1ps")
    s.append("")
    s.append("module scg_mac_array_v7 #(")
    s.append("    parameter integer N_LAYERS = 4,")
    s.append("    parameter integer NUM_PE   = 4")
    s.append(") (")
    s.append("    input  wire        clk_i,")
    s.append("    input  wire        rst_n_i,")
    s.append("    input  wire        start_i,")
    s.append("    output reg         busy_o,")
    s.append("    output reg         done_o,")
    s.append("    output reg [1:0]   class_o,")
    s.append("    output reg  [15:0] w_addr_o,")
    s.append("    input  wire [7:0]  w_data_i,")
    s.append("    output reg  [11:0] a_raddr_o,")
    s.append("    input  wire [7:0]  a_rdata_i,")
    s.append("    output reg  [11:0] a_waddr_o,")
    s.append("    output reg  [7:0]  a_wdata_o,")
    s.append("    output reg         a_we_o,")
    s.append("    output wire [1:0]  li_o")
    s.append(");")
    s.append("    reg [1:0] li;")
    s.append("    assign li_o = li;")
    s.append("")
    # Layer constant functions
    s.append("    // ---- Layer constants ----")
    for fn, table in [("L_CIN", L_CIN), ("L_COUT", L_COUT), ("L_K", L_K),
                      ("L_LEN_IN", L_LEN_IN), ("L_LEN_OUT", L_LEN_OUT),
                      ("L_PAD", L_PAD)]:
        width = 10 if max(table) >= 256 else 8
        s.append(f"    function automatic [{width-1}:0] {fn} (input [1:0] li_in);")
        s.append("        case (li_in)")
        for i, v in enumerate(table):
            s.append(f"            2'd{i}: {fn} = {width}'d{v};")
        s.append(f"            default: {fn} = {width}'d0;")
        s.append("        endcase")
        s.append("    endfunction")
        s.append("")
    # weight base
    s.append("    function automatic [15:0] L_W_BASE (input [1:0] li_in);")
    s.append("        case (li_in)")
    for i, v in enumerate(w_offsets):
        s.append(f"            2'd{i}: L_W_BASE = 16'd{v};")
    s.append("            default: L_W_BASE = 16'd0;")
    s.append("        endcase")
    s.append("    endfunction")
    s.append("")
    # flat co index = sum(L_COUT[0..li-1]) + co_within_layer
    s.append("    function automatic [7:0] L_FLAT_BASE (input [1:0] li_in);")
    s.append("        case (li_in)")
    for i, v in enumerate(flat_co_offsets):
        s.append(f"            2'd{i}: L_FLAT_BASE = 8'd{v};")
    s.append("            default: L_FLAT_BASE = 8'd0;")
    s.append("        endcase")
    s.append("    endfunction")
    s.append("")
    # bias / M0 / shift ROMs
    s.append("    // ---- Bias ROM (per-output-channel INT32 signed) ----")
    s.append(emit_case("L_BIAS", biases, 32, 8))
    s.append("")
    s.append("    // ---- M0 ROM (per-output-channel INT16 signed) ----")
    s.append(emit_case("L_M0", m0s, 16, 8))
    s.append("")
    s.append("    // ---- shift ROM (5-bit unsigned) ----")
    s.append("    function automatic [4:0] L_SHIFT (input [7:0] idx);")
    s.append("        case (idx)")
    for i, v in enumerate(shifts):
        s.append(f"            8'd{i}: L_SHIFT = 5'd{v};")
    s.append("            default: L_SHIFT = 5'd0;")
    s.append("        endcase")
    s.append("    endfunction")
    s.append("")

    # FSM body — same structure as scg_mac_array.v but with stride=2 step
    s.append("""\
    localparam S_IDLE=4'd0, S_INIT=4'd1, S_FETCH=4'd2, S_FETCH_DLY=4'd3,
               S_MAC=4'd4, S_REQ=4'd5, S_BIAS=4'd6, S_WRITE=4'd7,
               S_NEXT=4'd8, S_GAP=4'd9, S_DONE=4'd10;

    reg [3:0] state;
    reg [9:0] x_idx;     // 0..L_LEN_IN(li)-1, step=2 for stride-2 layers (li<3)
    reg [7:0] co_base, ci_idx;
    reg [3:0] k_idx;
    reg [2:0] write_idx;

    reg signed [31:0] acc [0:NUM_PE-1];
    reg signed [31:0] gap_acc [0:2];

    // Stride per layer: 2 for L0/L1/L2, 1 for L3
    function automatic [1:0] L_STRIDE (input [1:0] li_in);
        case (li_in) 2'd3: L_STRIDE = 2'd1;
                     default: L_STRIDE = 2'd2;
        endcase
    endfunction

    wire signed [10:0] pos_signed = $signed({1'b0, x_idx})
                                  + $signed({7'd0, k_idx})
                                  - $signed({7'd0, L_PAD(li)});
    wire act_valid = (pos_signed >= 11'sd0) &&
                     (pos_signed < $signed({1'b0, L_LEN_IN(li)}));
    wire [9:0] pos_eff = act_valid ? pos_signed[9:0] : 10'd0;

    // Activation read addr: ci * L_LEN_IN + pos_eff
    wire [11:0] a_addr_calc = ci_idx * L_LEN_IN(li) + {2'b0, pos_eff};
    // Weight addr: w_base + co * Cin * K + ci * K + k
    wire [15:0] w_addr_calc = L_W_BASE(li)
                            + ((co_base + {5'd0, write_idx_pe}) * L_CIN(li) * L_K(li))
                            + ci_idx * L_K(li) + {12'd0, k_idx};
    // We use co_base+0 for w lookup — all 4 PEs share the SAME co?  No, in
    // 4-PE arrangement each PE handles a distinct co.  But weight read is
    // serialized by co_base+0..3 inside S_FETCH; for simplicity we share.
    // Simpler: process 1 co at a time → effective NUM_PE=1. (Matches v0.)
    wire [2:0] write_idx_pe = 3'd0;   // single-channel-per-cycle conv

    // Latch act_valid by 1 cycle (BRAM read latency)
    reg act_valid_d;
    always @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) act_valid_d <= 1'b0;
        else if (state == S_FETCH) act_valid_d <= act_valid;
    end

    wire signed [7:0] act_byte = act_valid_d ? a_rdata_i : 8'sd0;
    wire signed [7:0] w_byte   = w_data_i;
    wire signed [15:0] mul     = act_byte * w_byte;

    function automatic signed [7:0] sat_int8_relu(input signed [47:0] v);
        if (v <= 0) sat_int8_relu = 8'sd0;
        else if (v > 127) sat_int8_relu = 8'sd127;
        else sat_int8_relu = v[7:0];
    endfunction
    function automatic signed [7:0] sat_int8_signed(input signed [47:0] v);
        if (v < -128) sat_int8_signed = -8'sd128;
        else if (v > 127) sat_int8_signed = 8'sd127;
        else sat_int8_signed = v[7:0];
    endfunction

    integer i;
    always @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            state <= S_IDLE; busy_o<=0; done_o<=0; class_o<=0;
            li<=0; x_idx<=0; co_base<=0; ci_idx<=0; k_idx<=0; write_idx<=0;
            for (i=0;i<NUM_PE;i=i+1) acc[i] <= 0;
            for (i=0;i<3;i=i+1) gap_acc[i] <= 0;
            w_addr_o<=0; a_raddr_o<=0; a_waddr_o<=0; a_wdata_o<=0; a_we_o<=0;
        end else begin
            done_o <= 1'b0;
            a_we_o <= 1'b0;
            case (state)
                S_IDLE: if (start_i) begin
                    busy_o<=1; li<=0; x_idx<=0; co_base<=0; ci_idx<=0; k_idx<=0;
                    for (i=0;i<3;i=i+1) gap_acc[i] <= 0;
                    state <= S_INIT;
                end
                S_INIT: begin
                    for (i=0;i<NUM_PE;i=i+1) acc[i] <= 0;
                    state <= S_FETCH;
                end
                S_FETCH: begin
                    w_addr_o  <= w_addr_calc;
                    a_raddr_o <= a_addr_calc;
                    state <= S_FETCH_DLY;
                end
                S_FETCH_DLY: state <= S_MAC;
                S_MAC: begin
                    acc[0] <= acc[0] + {{16{mul[15]}}, mul};
                    if (k_idx + 1 < L_K(li)) begin
                        k_idx <= k_idx + 4'd1; state <= S_FETCH;
                    end else begin
                        k_idx <= 0;
                        if (ci_idx + 1 < L_CIN(li)) begin
                            ci_idx <= ci_idx + 8'd1; state <= S_FETCH;
                        end else begin
                            ci_idx <= 0; state <= S_REQ;
                        end
                    end
                end
                S_REQ: begin
                    // Add bias for the current output channel index
                    acc[0] <= acc[0] + L_BIAS(L_FLAT_BASE(li) + co_base);
                    state <= S_BIAS;
                end
                S_BIAS: state <= S_WRITE;
                S_WRITE: begin
                    // Compute requantized output for current channel co_base
                    a_waddr_o <= co_base * L_LEN_OUT(li)
                               + (li == 2'd3 ? {2'b0, x_idx} : {2'b0, x_idx[9:1]});
                    a_wdata_o <= (li == 2'd3)
                        ? sat_int8_signed(($signed(acc[0]) * $signed(L_M0(L_FLAT_BASE(li) + co_base))) >>> L_SHIFT(L_FLAT_BASE(li) + co_base))
                        : sat_int8_relu  (($signed(acc[0]) * $signed(L_M0(L_FLAT_BASE(li) + co_base))) >>> L_SHIFT(L_FLAT_BASE(li) + co_base));
                    a_we_o <= 1'b1;
                    if (li == 2'd3) begin
                        gap_acc[co_base[1:0]] <= gap_acc[co_base[1:0]]
                            + sat_int8_signed(($signed(acc[0]) * $signed(L_M0(L_FLAT_BASE(li) + co_base))) >>> L_SHIFT(L_FLAT_BASE(li) + co_base));
                    end
                    state <= S_NEXT;
                end
                S_NEXT: begin
                    if (co_base + 1 < L_COUT(li)) begin
                        co_base <= co_base + 8'd1; state <= S_INIT;
                    end else begin
                        co_base <= 0;
                        // x_idx step = stride (2 for li<3, 1 for li=3)
                        if (x_idx + L_STRIDE(li) < L_LEN_IN(li)) begin
                            x_idx <= x_idx + {8'd0, L_STRIDE(li)}; state <= S_INIT;
                        end else begin
                            x_idx <= 0;
                            if (li == 2'd3) state <= S_GAP;
                            else begin li <= li + 2'd1; state <= S_INIT; end
                        end
                    end
                end
                S_GAP: begin
                    if (gap_acc[1] > gap_acc[0] && gap_acc[1] >= gap_acc[2]) class_o <= 2'd1;
                    else if (gap_acc[2] > gap_acc[0] && gap_acc[2] > gap_acc[1]) class_o <= 2'd2;
                    else class_o <= 2'd0;
                    state <= S_DONE;
                end
                S_DONE: begin busy_o <= 0; done_o <= 1; state <= S_IDLE; end
            endcase
        end
    end
endmodule
""")
    args.out.write_text("\n".join(s))
    print(f"-> wrote {args.out.relative_to(REPO)} ({len(s)} lines)")


if __name__ == "__main__":
    main()
