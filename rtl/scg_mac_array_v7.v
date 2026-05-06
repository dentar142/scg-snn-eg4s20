// scg_mac_array_v7.v - AUTO-GENERATED from rtl/weights_v7/
// Architecture: [1, 32, 64, 128, 3], K=[5, 5, 5, 1]
// In lengths : [256, 128, 64, 32]
// Out lengths: [128, 64, 32, 32]  (stride-2 conv if In==2*Out)
// Total weight bytes: 51744

`timescale 1ns/1ps

module scg_mac_array_v7 #(
    parameter integer N_LAYERS = 4,
    parameter integer NUM_PE   = 4
) (
    input  wire        clk_i,
    input  wire        rst_n_i,
    input  wire        start_i,
    output reg         busy_o,
    output reg         done_o,
    output reg [1:0]   class_o,
    output reg  [15:0] w_addr_o,
    input  wire [7:0]  w_data_i,
    output reg  [11:0] a_raddr_o,
    input  wire [7:0]  a_rdata_i,
    output reg  [11:0] a_waddr_o,
    output reg  [7:0]  a_wdata_o,
    output reg         a_we_o,
    output wire [1:0]  li_o
);
    reg [1:0] li;
    assign li_o = li;

    // ---- Layer constants ----
    function automatic [7:0] L_CIN (input [1:0] li_in);
        case (li_in)
            2'd0: L_CIN = 8'd1;
            2'd1: L_CIN = 8'd32;
            2'd2: L_CIN = 8'd64;
            2'd3: L_CIN = 8'd128;
            default: L_CIN = 8'd0;
        endcase
    endfunction

    function automatic [7:0] L_COUT (input [1:0] li_in);
        case (li_in)
            2'd0: L_COUT = 8'd32;
            2'd1: L_COUT = 8'd64;
            2'd2: L_COUT = 8'd128;
            2'd3: L_COUT = 8'd3;
            default: L_COUT = 8'd0;
        endcase
    endfunction

    function automatic [7:0] L_K (input [1:0] li_in);
        case (li_in)
            2'd0: L_K = 8'd5;
            2'd1: L_K = 8'd5;
            2'd2: L_K = 8'd5;
            2'd3: L_K = 8'd1;
            default: L_K = 8'd0;
        endcase
    endfunction

    function automatic [9:0] L_LEN_IN (input [1:0] li_in);
        case (li_in)
            2'd0: L_LEN_IN = 10'd256;
            2'd1: L_LEN_IN = 10'd128;
            2'd2: L_LEN_IN = 10'd64;
            2'd3: L_LEN_IN = 10'd32;
            default: L_LEN_IN = 10'd0;
        endcase
    endfunction

    function automatic [7:0] L_LEN_OUT (input [1:0] li_in);
        case (li_in)
            2'd0: L_LEN_OUT = 8'd128;
            2'd1: L_LEN_OUT = 8'd64;
            2'd2: L_LEN_OUT = 8'd32;
            2'd3: L_LEN_OUT = 8'd32;
            default: L_LEN_OUT = 8'd0;
        endcase
    endfunction

    function automatic [7:0] L_PAD (input [1:0] li_in);
        case (li_in)
            2'd0: L_PAD = 8'd2;
            2'd1: L_PAD = 8'd2;
            2'd2: L_PAD = 8'd2;
            2'd3: L_PAD = 8'd0;
            default: L_PAD = 8'd0;
        endcase
    endfunction

    function automatic [15:0] L_W_BASE (input [1:0] li_in);
        case (li_in)
            2'd0: L_W_BASE = 16'd0;
            2'd1: L_W_BASE = 16'd160;
            2'd2: L_W_BASE = 16'd10400;
            2'd3: L_W_BASE = 16'd51360;
            default: L_W_BASE = 16'd0;
        endcase
    endfunction

    function automatic [7:0] L_FLAT_BASE (input [1:0] li_in);
        case (li_in)
            2'd0: L_FLAT_BASE = 8'd0;
            2'd1: L_FLAT_BASE = 8'd32;
            2'd2: L_FLAT_BASE = 8'd96;
            2'd3: L_FLAT_BASE = 8'd224;
            default: L_FLAT_BASE = 8'd0;
        endcase
    endfunction

    // ---- Bias ROM (per-output-channel INT32 signed) ----
    function automatic signed [31:0] L_BIAS (input [7:0] idx);
        case (idx)
            8'd0: L_BIAS = 32'shFFFFFA17;
            8'd1: L_BIAS = 32'shFFFFFAF6;
            8'd2: L_BIAS = 32'shFFFFF845;
            8'd3: L_BIAS = 32'shFFFFFFD8;
            8'd4: L_BIAS = 32'sh0000098E;
            8'd5: L_BIAS = 32'shFFFFFFCF;
            8'd6: L_BIAS = 32'sh000001B6;
            8'd7: L_BIAS = 32'shFFFFF9E6;
            8'd8: L_BIAS = 32'shFFFFFFC8;
            8'd9: L_BIAS = 32'shFFFFFFFA;
            8'd10: L_BIAS = 32'shFFFFFE3D;
            8'd11: L_BIAS = 32'sh0000056B;
            8'd12: L_BIAS = 32'sh00000648;
            8'd13: L_BIAS = 32'sh00000156;
            8'd14: L_BIAS = 32'sh00000082;
            8'd15: L_BIAS = 32'sh0000009D;
            8'd16: L_BIAS = 32'sh000003ED;
            8'd17: L_BIAS = 32'sh000009F0;
            8'd18: L_BIAS = 32'sh00001E6B;
            8'd19: L_BIAS = 32'sh0000088A;
            8'd20: L_BIAS = 32'shFFFFFD16;
            8'd21: L_BIAS = 32'sh00002189;
            8'd22: L_BIAS = 32'sh00000324;
            8'd23: L_BIAS = 32'shFFFFFD2D;
            8'd24: L_BIAS = 32'shFFFFFBB7;
            8'd25: L_BIAS = 32'sh00000587;
            8'd26: L_BIAS = 32'shFFFFFF85;
            8'd27: L_BIAS = 32'shFFFFF5A2;
            8'd28: L_BIAS = 32'sh0000001E;
            8'd29: L_BIAS = 32'shFFFFFB51;
            8'd30: L_BIAS = 32'sh00000172;
            8'd31: L_BIAS = 32'shFFFFFFC5;
            8'd32: L_BIAS = 32'sh000013DF;
            8'd33: L_BIAS = 32'sh0000112C;
            8'd34: L_BIAS = 32'shFFFFFC02;
            8'd35: L_BIAS = 32'sh00000EA4;
            8'd36: L_BIAS = 32'sh00000A42;
            8'd37: L_BIAS = 32'sh000013E3;
            8'd38: L_BIAS = 32'sh00000CF8;
            8'd39: L_BIAS = 32'shFFFFFF62;
            8'd40: L_BIAS = 32'sh00001269;
            8'd41: L_BIAS = 32'sh00000228;
            8'd42: L_BIAS = 32'sh0000055B;
            8'd43: L_BIAS = 32'sh00000C40;
            8'd44: L_BIAS = 32'shFFFFE910;
            8'd45: L_BIAS = 32'sh0000094C;
            8'd46: L_BIAS = 32'sh00001287;
            8'd47: L_BIAS = 32'shFFFFFCCA;
            8'd48: L_BIAS = 32'sh0000037E;
            8'd49: L_BIAS = 32'sh00000E05;
            8'd50: L_BIAS = 32'shFFFFEDBD;
            8'd51: L_BIAS = 32'sh000003CE;
            8'd52: L_BIAS = 32'shFFFFFF38;
            8'd53: L_BIAS = 32'sh00000F88;
            8'd54: L_BIAS = 32'sh0000051A;
            8'd55: L_BIAS = 32'shFFFFEA36;
            8'd56: L_BIAS = 32'shFFFFF53D;
            8'd57: L_BIAS = 32'sh00000189;
            8'd58: L_BIAS = 32'sh00001A51;
            8'd59: L_BIAS = 32'sh000013E4;
            8'd60: L_BIAS = 32'shFFFFF6D8;
            8'd61: L_BIAS = 32'sh000002E2;
            8'd62: L_BIAS = 32'sh000006D6;
            8'd63: L_BIAS = 32'shFFFFFF9C;
            8'd64: L_BIAS = 32'sh000013C6;
            8'd65: L_BIAS = 32'sh00000C26;
            8'd66: L_BIAS = 32'sh000007AC;
            8'd67: L_BIAS = 32'sh00000A5B;
            8'd68: L_BIAS = 32'shFFFFF3D4;
            8'd69: L_BIAS = 32'sh000015EF;
            8'd70: L_BIAS = 32'sh000012ED;
            8'd71: L_BIAS = 32'shFFFFFCD8;
            8'd72: L_BIAS = 32'shFFFFEFDE;
            8'd73: L_BIAS = 32'sh00000D74;
            8'd74: L_BIAS = 32'sh00000C2A;
            8'd75: L_BIAS = 32'sh00001411;
            8'd76: L_BIAS = 32'sh000000B1;
            8'd77: L_BIAS = 32'shFFFFF1FC;
            8'd78: L_BIAS = 32'sh000017BF;
            8'd79: L_BIAS = 32'sh000016A2;
            8'd80: L_BIAS = 32'sh0000052E;
            8'd81: L_BIAS = 32'shFFFFFE2E;
            8'd82: L_BIAS = 32'shFFFFFAF3;
            8'd83: L_BIAS = 32'shFFFFFBC2;
            8'd84: L_BIAS = 32'sh00000940;
            8'd85: L_BIAS = 32'sh00000574;
            8'd86: L_BIAS = 32'shFFFFFF9F;
            8'd87: L_BIAS = 32'sh00000283;
            8'd88: L_BIAS = 32'shFFFFEFAB;
            8'd89: L_BIAS = 32'sh000005BB;
            8'd90: L_BIAS = 32'shFFFFF735;
            8'd91: L_BIAS = 32'sh0000100C;
            8'd92: L_BIAS = 32'shFFFFE895;
            8'd93: L_BIAS = 32'sh00001117;
            8'd94: L_BIAS = 32'sh000016DB;
            8'd95: L_BIAS = 32'sh00000947;
            8'd96: L_BIAS = 32'shFFFFFFF6;
            8'd97: L_BIAS = 32'sh00000963;
            8'd98: L_BIAS = 32'shFFFFF883;
            8'd99: L_BIAS = 32'sh000000E4;
            8'd100: L_BIAS = 32'shFFFFFFBF;
            8'd101: L_BIAS = 32'shFFFFFFEB;
            8'd102: L_BIAS = 32'sh000004FE;
            8'd103: L_BIAS = 32'shFFFFF836;
            8'd104: L_BIAS = 32'shFFFFFE06;
            8'd105: L_BIAS = 32'shFFFFF17A;
            8'd106: L_BIAS = 32'shFFFFF9E2;
            8'd107: L_BIAS = 32'shFFFFF2DD;
            8'd108: L_BIAS = 32'sh0000075D;
            8'd109: L_BIAS = 32'shFFFFF543;
            8'd110: L_BIAS = 32'sh0000017D;
            8'd111: L_BIAS = 32'sh0000001A;
            8'd112: L_BIAS = 32'shFFFFF58B;
            8'd113: L_BIAS = 32'shFFFFFAFA;
            8'd114: L_BIAS = 32'sh00000AE8;
            8'd115: L_BIAS = 32'sh00000AF2;
            8'd116: L_BIAS = 32'shFFFFF2A1;
            8'd117: L_BIAS = 32'shFFFFFFF0;
            8'd118: L_BIAS = 32'sh00000347;
            8'd119: L_BIAS = 32'shFFFFF63E;
            8'd120: L_BIAS = 32'sh0000076A;
            8'd121: L_BIAS = 32'sh00000048;
            8'd122: L_BIAS = 32'sh00000566;
            8'd123: L_BIAS = 32'sh00000322;
            8'd124: L_BIAS = 32'sh000006D2;
            8'd125: L_BIAS = 32'shFFFFF73D;
            8'd126: L_BIAS = 32'sh00000899;
            8'd127: L_BIAS = 32'sh00000647;
            8'd128: L_BIAS = 32'shFFFFF9DA;
            8'd129: L_BIAS = 32'shFFFFFD1D;
            8'd130: L_BIAS = 32'sh000003E0;
            8'd131: L_BIAS = 32'sh000001EF;
            8'd132: L_BIAS = 32'shFFFFFC36;
            8'd133: L_BIAS = 32'shFFFFF377;
            8'd134: L_BIAS = 32'shFFFFFCE2;
            8'd135: L_BIAS = 32'shFFFFFA3C;
            8'd136: L_BIAS = 32'sh000002A7;
            8'd137: L_BIAS = 32'shFFFFFB36;
            8'd138: L_BIAS = 32'shFFFFF52E;
            8'd139: L_BIAS = 32'shFFFFFC9B;
            8'd140: L_BIAS = 32'shFFFFFB65;
            8'd141: L_BIAS = 32'sh0000050F;
            8'd142: L_BIAS = 32'shFFFFFC2B;
            8'd143: L_BIAS = 32'shFFFFF71F;
            8'd144: L_BIAS = 32'sh0000011E;
            8'd145: L_BIAS = 32'sh000000D8;
            8'd146: L_BIAS = 32'sh00000411;
            8'd147: L_BIAS = 32'sh000004DC;
            8'd148: L_BIAS = 32'shFFFFF85F;
            8'd149: L_BIAS = 32'shFFFFF6EA;
            8'd150: L_BIAS = 32'shFFFFF7E7;
            8'd151: L_BIAS = 32'sh00001059;
            8'd152: L_BIAS = 32'sh000005AC;
            8'd153: L_BIAS = 32'shFFFFFDA7;
            8'd154: L_BIAS = 32'shFFFFF70F;
            8'd155: L_BIAS = 32'sh00000490;
            8'd156: L_BIAS = 32'shFFFFF23E;
            8'd157: L_BIAS = 32'sh0000083A;
            8'd158: L_BIAS = 32'sh0000035D;
            8'd159: L_BIAS = 32'sh000006B1;
            8'd160: L_BIAS = 32'sh00000316;
            8'd161: L_BIAS = 32'shFFFFFBAE;
            8'd162: L_BIAS = 32'sh000003E2;
            8'd163: L_BIAS = 32'sh00000AE8;
            8'd164: L_BIAS = 32'shFFFFFC2D;
            8'd165: L_BIAS = 32'sh0000046D;
            8'd166: L_BIAS = 32'sh000006E9;
            8'd167: L_BIAS = 32'shFFFFF94C;
            8'd168: L_BIAS = 32'shFFFFFE26;
            8'd169: L_BIAS = 32'shFFFFFF8E;
            8'd170: L_BIAS = 32'sh0000038E;
            8'd171: L_BIAS = 32'shFFFFFDB0;
            8'd172: L_BIAS = 32'sh0000031D;
            8'd173: L_BIAS = 32'shFFFFFA01;
            8'd174: L_BIAS = 32'sh000006BF;
            8'd175: L_BIAS = 32'shFFFFFF1F;
            8'd176: L_BIAS = 32'shFFFFF8FC;
            8'd177: L_BIAS = 32'sh0000096F;
            8'd178: L_BIAS = 32'shFFFFFB12;
            8'd179: L_BIAS = 32'shFFFFFE16;
            8'd180: L_BIAS = 32'shFFFFF06D;
            8'd181: L_BIAS = 32'sh000003A2;
            8'd182: L_BIAS = 32'sh000009FF;
            8'd183: L_BIAS = 32'sh000000F6;
            8'd184: L_BIAS = 32'shFFFFFB04;
            8'd185: L_BIAS = 32'shFFFFFABC;
            8'd186: L_BIAS = 32'shFFFFF98D;
            8'd187: L_BIAS = 32'shFFFFFB97;
            8'd188: L_BIAS = 32'shFFFFFCBF;
            8'd189: L_BIAS = 32'sh00000085;
            8'd190: L_BIAS = 32'shFFFFFB71;
            8'd191: L_BIAS = 32'shFFFFF436;
            8'd192: L_BIAS = 32'shFFFFF496;
            8'd193: L_BIAS = 32'sh00000A14;
            8'd194: L_BIAS = 32'sh00000003;
            8'd195: L_BIAS = 32'sh0000084D;
            8'd196: L_BIAS = 32'sh00000551;
            8'd197: L_BIAS = 32'shFFFFFFAA;
            8'd198: L_BIAS = 32'sh00000A0B;
            8'd199: L_BIAS = 32'sh000001BF;
            8'd200: L_BIAS = 32'sh0000097B;
            8'd201: L_BIAS = 32'sh00000C38;
            8'd202: L_BIAS = 32'shFFFFFC09;
            8'd203: L_BIAS = 32'sh00000439;
            8'd204: L_BIAS = 32'shFFFFFF0A;
            8'd205: L_BIAS = 32'sh000004C8;
            8'd206: L_BIAS = 32'sh00000148;
            8'd207: L_BIAS = 32'sh0000009D;
            8'd208: L_BIAS = 32'shFFFFFF63;
            8'd209: L_BIAS = 32'shFFFFF4AC;
            8'd210: L_BIAS = 32'sh00000869;
            8'd211: L_BIAS = 32'shFFFFFCE4;
            8'd212: L_BIAS = 32'sh00000A3B;
            8'd213: L_BIAS = 32'shFFFFFC5A;
            8'd214: L_BIAS = 32'shFFFFF8B7;
            8'd215: L_BIAS = 32'sh00000659;
            8'd216: L_BIAS = 32'sh000005AA;
            8'd217: L_BIAS = 32'shFFFFF3EF;
            8'd218: L_BIAS = 32'shFFFFFFCD;
            8'd219: L_BIAS = 32'shFFFFFAAA;
            8'd220: L_BIAS = 32'sh0000044B;
            8'd221: L_BIAS = 32'sh00000851;
            8'd222: L_BIAS = 32'shFFFFF750;
            8'd223: L_BIAS = 32'sh00000563;
            8'd224: L_BIAS = 32'sh000002B2;
            8'd225: L_BIAS = 32'sh00000237;
            8'd226: L_BIAS = 32'sh00000013;
            default: L_BIAS = 32'sd0;
        endcase
    endfunction

    // ---- M0 ROM (per-output-channel INT16 signed) ----
    function automatic signed [15:0] L_M0 (input [7:0] idx);
        case (idx)
            8'd0: L_M0 = 16'sh3A09;
            8'd1: L_M0 = 16'sh3F8F;
            8'd2: L_M0 = 16'sh1A13;
            8'd3: L_M0 = 16'sh0167;
            8'd4: L_M0 = 16'sh0A7D;
            8'd5: L_M0 = 16'sh3041;
            8'd6: L_M0 = 16'sh22B7;
            8'd7: L_M0 = 16'sh24B1;
            8'd8: L_M0 = 16'sh2113;
            8'd9: L_M0 = 16'sh255F;
            8'd10: L_M0 = 16'sh54F9;
            8'd11: L_M0 = 16'sh65E5;
            8'd12: L_M0 = 16'sh6539;
            8'd13: L_M0 = 16'sh57A1;
            8'd14: L_M0 = 16'sh1481;
            8'd15: L_M0 = 16'sh1A99;
            8'd16: L_M0 = 16'sh53BD;
            8'd17: L_M0 = 16'sh50FB;
            8'd18: L_M0 = 16'sh5279;
            8'd19: L_M0 = 16'sh7231;
            8'd20: L_M0 = 16'sh17AD;
            8'd21: L_M0 = 16'sh7F15;
            8'd22: L_M0 = 16'sh5427;
            8'd23: L_M0 = 16'sh237B;
            8'd24: L_M0 = 16'sh6419;
            8'd25: L_M0 = 16'sh32B1;
            8'd26: L_M0 = 16'sh5475;
            8'd27: L_M0 = 16'sh707F;
            8'd28: L_M0 = 16'sh5DA7;
            8'd29: L_M0 = 16'sh7479;
            8'd30: L_M0 = 16'sh2F47;
            8'd31: L_M0 = 16'sh2ECB;
            8'd32: L_M0 = 16'sh476D;
            8'd33: L_M0 = 16'sh2F05;
            8'd34: L_M0 = 16'sh2625;
            8'd35: L_M0 = 16'sh40FB;
            8'd36: L_M0 = 16'sh16E5;
            8'd37: L_M0 = 16'sh0539;
            8'd38: L_M0 = 16'sh0031;
            8'd39: L_M0 = 16'sh4963;
            8'd40: L_M0 = 16'sh2D09;
            8'd41: L_M0 = 16'sh4FBF;
            8'd42: L_M0 = 16'sh4183;
            8'd43: L_M0 = 16'sh7B35;
            8'd44: L_M0 = 16'sh27BB;
            8'd45: L_M0 = 16'sh746F;
            8'd46: L_M0 = 16'sh3F89;
            8'd47: L_M0 = 16'sh5B21;
            8'd48: L_M0 = 16'sh5A85;
            8'd49: L_M0 = 16'sh6AB7;
            8'd50: L_M0 = 16'sh5D51;
            8'd51: L_M0 = 16'sh05B7;
            8'd52: L_M0 = 16'sh65E7;
            8'd53: L_M0 = 16'sh19E7;
            8'd54: L_M0 = 16'sh1087;
            8'd55: L_M0 = 16'sh5465;
            8'd56: L_M0 = 16'sh69EF;
            8'd57: L_M0 = 16'sh47AF;
            8'd58: L_M0 = 16'sh245B;
            8'd59: L_M0 = 16'sh7E7B;
            8'd60: L_M0 = 16'sh106F;
            8'd61: L_M0 = 16'sh201B;
            8'd62: L_M0 = 16'sh4CBB;
            8'd63: L_M0 = 16'sh10E3;
            8'd64: L_M0 = 16'sh2D63;
            8'd65: L_M0 = 16'sh15C3;
            8'd66: L_M0 = 16'sh500B;
            8'd67: L_M0 = 16'sh46C5;
            8'd68: L_M0 = 16'sh28EB;
            8'd69: L_M0 = 16'sh47D1;
            8'd70: L_M0 = 16'sh4717;
            8'd71: L_M0 = 16'sh40EF;
            8'd72: L_M0 = 16'sh7A23;
            8'd73: L_M0 = 16'sh7E0B;
            8'd74: L_M0 = 16'sh060D;
            8'd75: L_M0 = 16'sh1A67;
            8'd76: L_M0 = 16'sh73F3;
            8'd77: L_M0 = 16'sh2421;
            8'd78: L_M0 = 16'sh443F;
            8'd79: L_M0 = 16'sh10B3;
            8'd80: L_M0 = 16'sh2D4F;
            8'd81: L_M0 = 16'sh7F3B;
            8'd82: L_M0 = 16'sh5D7D;
            8'd83: L_M0 = 16'sh2F39;
            8'd84: L_M0 = 16'sh1E4D;
            8'd85: L_M0 = 16'sh3A41;
            8'd86: L_M0 = 16'sh577D;
            8'd87: L_M0 = 16'sh5135;
            8'd88: L_M0 = 16'sh204F;
            8'd89: L_M0 = 16'sh2ABD;
            8'd90: L_M0 = 16'sh745D;
            8'd91: L_M0 = 16'sh63FB;
            8'd92: L_M0 = 16'sh3509;
            8'd93: L_M0 = 16'sh4ACB;
            8'd94: L_M0 = 16'sh7353;
            8'd95: L_M0 = 16'sh7587;
            8'd96: L_M0 = 16'sh606B;
            8'd97: L_M0 = 16'sh6D59;
            8'd98: L_M0 = 16'sh4D9D;
            8'd99: L_M0 = 16'sh42AD;
            8'd100: L_M0 = 16'sh2699;
            8'd101: L_M0 = 16'sh6577;
            8'd102: L_M0 = 16'sh3063;
            8'd103: L_M0 = 16'sh5D9D;
            8'd104: L_M0 = 16'sh5A49;
            8'd105: L_M0 = 16'sh6259;
            8'd106: L_M0 = 16'sh2883;
            8'd107: L_M0 = 16'sh607D;
            8'd108: L_M0 = 16'sh6703;
            8'd109: L_M0 = 16'sh2983;
            8'd110: L_M0 = 16'sh0B2D;
            8'd111: L_M0 = 16'sh57F1;
            8'd112: L_M0 = 16'sh4F65;
            8'd113: L_M0 = 16'sh5AAB;
            8'd114: L_M0 = 16'sh708F;
            8'd115: L_M0 = 16'sh6F3D;
            8'd116: L_M0 = 16'sh65C9;
            8'd117: L_M0 = 16'sh2591;
            8'd118: L_M0 = 16'sh0B6D;
            8'd119: L_M0 = 16'sh7703;
            8'd120: L_M0 = 16'sh30EB;
            8'd121: L_M0 = 16'sh4C39;
            8'd122: L_M0 = 16'sh58FF;
            8'd123: L_M0 = 16'sh4177;
            8'd124: L_M0 = 16'sh083F;
            8'd125: L_M0 = 16'sh18AF;
            8'd126: L_M0 = 16'sh54E7;
            8'd127: L_M0 = 16'sh583D;
            8'd128: L_M0 = 16'sh3367;
            8'd129: L_M0 = 16'sh23A1;
            8'd130: L_M0 = 16'sh30B7;
            8'd131: L_M0 = 16'sh45F1;
            8'd132: L_M0 = 16'sh0B8F;
            8'd133: L_M0 = 16'sh21D3;
            8'd134: L_M0 = 16'sh7A63;
            8'd135: L_M0 = 16'sh37FF;
            8'd136: L_M0 = 16'sh1061;
            8'd137: L_M0 = 16'sh53C7;
            8'd138: L_M0 = 16'sh5D05;
            8'd139: L_M0 = 16'sh460F;
            8'd140: L_M0 = 16'sh2D51;
            8'd141: L_M0 = 16'sh6D55;
            8'd142: L_M0 = 16'sh41CF;
            8'd143: L_M0 = 16'sh64AD;
            8'd144: L_M0 = 16'sh495F;
            8'd145: L_M0 = 16'sh752F;
            8'd146: L_M0 = 16'sh0E7B;
            8'd147: L_M0 = 16'sh2873;
            8'd148: L_M0 = 16'sh27C7;
            8'd149: L_M0 = 16'sh1A07;
            8'd150: L_M0 = 16'sh15E7;
            8'd151: L_M0 = 16'sh2923;
            8'd152: L_M0 = 16'sh4625;
            8'd153: L_M0 = 16'sh4765;
            8'd154: L_M0 = 16'sh4DF1;
            8'd155: L_M0 = 16'sh74FB;
            8'd156: L_M0 = 16'sh2473;
            8'd157: L_M0 = 16'sh5F39;
            8'd158: L_M0 = 16'sh1533;
            8'd159: L_M0 = 16'sh54E1;
            8'd160: L_M0 = 16'sh1DAB;
            8'd161: L_M0 = 16'sh34D7;
            8'd162: L_M0 = 16'sh24EF;
            8'd163: L_M0 = 16'sh4A85;
            8'd164: L_M0 = 16'sh6B89;
            8'd165: L_M0 = 16'sh4995;
            8'd166: L_M0 = 16'sh067B;
            8'd167: L_M0 = 16'sh1FA3;
            8'd168: L_M0 = 16'sh72FF;
            8'd169: L_M0 = 16'sh6AEF;
            8'd170: L_M0 = 16'sh6F7B;
            8'd171: L_M0 = 16'sh0F19;
            8'd172: L_M0 = 16'sh665D;
            8'd173: L_M0 = 16'sh01AF;
            8'd174: L_M0 = 16'sh1201;
            8'd175: L_M0 = 16'sh3FDD;
            8'd176: L_M0 = 16'sh507B;
            8'd177: L_M0 = 16'sh372B;
            8'd178: L_M0 = 16'sh180D;
            8'd179: L_M0 = 16'sh309B;
            8'd180: L_M0 = 16'sh53AF;
            8'd181: L_M0 = 16'sh0CC7;
            8'd182: L_M0 = 16'sh4ACF;
            8'd183: L_M0 = 16'sh54B1;
            8'd184: L_M0 = 16'sh2B13;
            8'd185: L_M0 = 16'sh3BD5;
            8'd186: L_M0 = 16'sh70E3;
            8'd187: L_M0 = 16'sh1995;
            8'd188: L_M0 = 16'sh4A9F;
            8'd189: L_M0 = 16'sh5829;
            8'd190: L_M0 = 16'sh6047;
            8'd191: L_M0 = 16'sh2FD3;
            8'd192: L_M0 = 16'sh0641;
            8'd193: L_M0 = 16'sh4B8D;
            8'd194: L_M0 = 16'sh0613;
            8'd195: L_M0 = 16'sh33C3;
            8'd196: L_M0 = 16'sh1BC3;
            8'd197: L_M0 = 16'sh248B;
            8'd198: L_M0 = 16'sh4905;
            8'd199: L_M0 = 16'sh4A71;
            8'd200: L_M0 = 16'sh2A47;
            8'd201: L_M0 = 16'sh3937;
            8'd202: L_M0 = 16'sh00CD;
            8'd203: L_M0 = 16'sh6773;
            8'd204: L_M0 = 16'sh44E3;
            8'd205: L_M0 = 16'sh77BB;
            8'd206: L_M0 = 16'sh64A9;
            8'd207: L_M0 = 16'sh35B3;
            8'd208: L_M0 = 16'sh2B27;
            8'd209: L_M0 = 16'sh5C4F;
            8'd210: L_M0 = 16'sh126F;
            8'd211: L_M0 = 16'sh24AD;
            8'd212: L_M0 = 16'sh494B;
            8'd213: L_M0 = 16'sh2DAB;
            8'd214: L_M0 = 16'sh4F53;
            8'd215: L_M0 = 16'sh3F93;
            8'd216: L_M0 = 16'sh6F49;
            8'd217: L_M0 = 16'sh16D5;
            8'd218: L_M0 = 16'sh6B8F;
            8'd219: L_M0 = 16'sh71ED;
            8'd220: L_M0 = 16'sh7503;
            8'd221: L_M0 = 16'sh086B;
            8'd222: L_M0 = 16'sh4709;
            8'd223: L_M0 = 16'sh35B9;
            8'd224: L_M0 = 16'sh3AA5;
            8'd225: L_M0 = 16'sh426D;
            8'd226: L_M0 = 16'sh3EC7;
            default: L_M0 = 16'sd0;
        endcase
    endfunction

    // ---- shift ROM (5-bit unsigned) ----
    function automatic [4:0] L_SHIFT (input [7:0] idx);
        case (idx)
            8'd0: L_SHIFT = 5'd23;
            8'd1: L_SHIFT = 5'd22;
            8'd2: L_SHIFT = 5'd22;
            8'd3: L_SHIFT = 5'd15;
            8'd4: L_SHIFT = 5'd22;
            8'd5: L_SHIFT = 5'd24;
            8'd6: L_SHIFT = 5'd22;
            8'd7: L_SHIFT = 5'd23;
            8'd8: L_SHIFT = 5'd21;
            8'd9: L_SHIFT = 5'd22;
            8'd10: L_SHIFT = 5'd22;
            8'd11: L_SHIFT = 5'd23;
            8'd12: L_SHIFT = 5'd23;
            8'd13: L_SHIFT = 5'd22;
            8'd14: L_SHIFT = 5'd20;
            8'd15: L_SHIFT = 5'd20;
            8'd16: L_SHIFT = 5'd22;
            8'd17: L_SHIFT = 5'd24;
            8'd18: L_SHIFT = 5'd25;
            8'd19: L_SHIFT = 5'd23;
            8'd20: L_SHIFT = 5'd22;
            8'd21: L_SHIFT = 5'd26;
            8'd22: L_SHIFT = 5'd22;
            8'd23: L_SHIFT = 5'd22;
            8'd24: L_SHIFT = 5'd23;
            8'd25: L_SHIFT = 5'd21;
            8'd26: L_SHIFT = 5'd22;
            8'd27: L_SHIFT = 5'd24;
            8'd28: L_SHIFT = 5'd23;
            8'd29: L_SHIFT = 5'd23;
            8'd30: L_SHIFT = 5'd21;
            8'd31: L_SHIFT = 5'd20;
            8'd32: L_SHIFT = 5'd23;
            8'd33: L_SHIFT = 5'd22;
            8'd34: L_SHIFT = 5'd21;
            8'd35: L_SHIFT = 5'd21;
            8'd36: L_SHIFT = 5'd21;
            8'd37: L_SHIFT = 5'd19;
            8'd38: L_SHIFT = 5'd14;
            8'd39: L_SHIFT = 5'd21;
            8'd40: L_SHIFT = 5'd21;
            8'd41: L_SHIFT = 5'd23;
            8'd42: L_SHIFT = 5'd22;
            8'd43: L_SHIFT = 5'd24;
            8'd44: L_SHIFT = 5'd22;
            8'd45: L_SHIFT = 5'd23;
            8'd46: L_SHIFT = 5'd23;
            8'd47: L_SHIFT = 5'd22;
            8'd48: L_SHIFT = 5'd23;
            8'd49: L_SHIFT = 5'd23;
            8'd50: L_SHIFT = 5'd23;
            8'd51: L_SHIFT = 5'd18;
            8'd52: L_SHIFT = 5'd22;
            8'd53: L_SHIFT = 5'd21;
            8'd54: L_SHIFT = 5'd20;
            8'd55: L_SHIFT = 5'd23;
            8'd56: L_SHIFT = 5'd23;
            8'd57: L_SHIFT = 5'd23;
            8'd58: L_SHIFT = 5'd22;
            8'd59: L_SHIFT = 5'd24;
            8'd60: L_SHIFT = 5'd21;
            8'd61: L_SHIFT = 5'd21;
            8'd62: L_SHIFT = 5'd22;
            8'd63: L_SHIFT = 5'd20;
            8'd64: L_SHIFT = 5'd22;
            8'd65: L_SHIFT = 5'd21;
            8'd66: L_SHIFT = 5'd24;
            8'd67: L_SHIFT = 5'd23;
            8'd68: L_SHIFT = 5'd21;
            8'd69: L_SHIFT = 5'd23;
            8'd70: L_SHIFT = 5'd22;
            8'd71: L_SHIFT = 5'd22;
            8'd72: L_SHIFT = 5'd23;
            8'd73: L_SHIFT = 5'd23;
            8'd74: L_SHIFT = 5'd18;
            8'd75: L_SHIFT = 5'd21;
            8'd76: L_SHIFT = 5'd23;
            8'd77: L_SHIFT = 5'd21;
            8'd78: L_SHIFT = 5'd22;
            8'd79: L_SHIFT = 5'd21;
            8'd80: L_SHIFT = 5'd21;
            8'd81: L_SHIFT = 5'd24;
            8'd82: L_SHIFT = 5'd22;
            8'd83: L_SHIFT = 5'd21;
            8'd84: L_SHIFT = 5'd21;
            8'd85: L_SHIFT = 5'd22;
            8'd86: L_SHIFT = 5'd22;
            8'd87: L_SHIFT = 5'd22;
            8'd88: L_SHIFT = 5'd21;
            8'd89: L_SHIFT = 5'd22;
            8'd90: L_SHIFT = 5'd23;
            8'd91: L_SHIFT = 5'd23;
            8'd92: L_SHIFT = 5'd22;
            8'd93: L_SHIFT = 5'd23;
            8'd94: L_SHIFT = 5'd23;
            8'd95: L_SHIFT = 5'd24;
            8'd96: L_SHIFT = 5'd22;
            8'd97: L_SHIFT = 5'd22;
            8'd98: L_SHIFT = 5'd22;
            8'd99: L_SHIFT = 5'd21;
            8'd100: L_SHIFT = 5'd21;
            8'd101: L_SHIFT = 5'd22;
            8'd102: L_SHIFT = 5'd21;
            8'd103: L_SHIFT = 5'd22;
            8'd104: L_SHIFT = 5'd22;
            8'd105: L_SHIFT = 5'd22;
            8'd106: L_SHIFT = 5'd21;
            8'd107: L_SHIFT = 5'd22;
            8'd108: L_SHIFT = 5'd22;
            8'd109: L_SHIFT = 5'd21;
            8'd110: L_SHIFT = 5'd19;
            8'd111: L_SHIFT = 5'd22;
            8'd112: L_SHIFT = 5'd22;
            8'd113: L_SHIFT = 5'd22;
            8'd114: L_SHIFT = 5'd22;
            8'd115: L_SHIFT = 5'd23;
            8'd116: L_SHIFT = 5'd22;
            8'd117: L_SHIFT = 5'd21;
            8'd118: L_SHIFT = 5'd19;
            8'd119: L_SHIFT = 5'd22;
            8'd120: L_SHIFT = 5'd21;
            8'd121: L_SHIFT = 5'd22;
            8'd122: L_SHIFT = 5'd22;
            8'd123: L_SHIFT = 5'd22;
            8'd124: L_SHIFT = 5'd19;
            8'd125: L_SHIFT = 5'd20;
            8'd126: L_SHIFT = 5'd22;
            8'd127: L_SHIFT = 5'd22;
            8'd128: L_SHIFT = 5'd21;
            8'd129: L_SHIFT = 5'd21;
            8'd130: L_SHIFT = 5'd21;
            8'd131: L_SHIFT = 5'd22;
            8'd132: L_SHIFT = 5'd19;
            8'd133: L_SHIFT = 5'd21;
            8'd134: L_SHIFT = 5'd22;
            8'd135: L_SHIFT = 5'd21;
            8'd136: L_SHIFT = 5'd20;
            8'd137: L_SHIFT = 5'd22;
            8'd138: L_SHIFT = 5'd22;
            8'd139: L_SHIFT = 5'd22;
            8'd140: L_SHIFT = 5'd21;
            8'd141: L_SHIFT = 5'd23;
            8'd142: L_SHIFT = 5'd21;
            8'd143: L_SHIFT = 5'd22;
            8'd144: L_SHIFT = 5'd22;
            8'd145: L_SHIFT = 5'd22;
            8'd146: L_SHIFT = 5'd19;
            8'd147: L_SHIFT = 5'd21;
            8'd148: L_SHIFT = 5'd21;
            8'd149: L_SHIFT = 5'd20;
            8'd150: L_SHIFT = 5'd20;
            8'd151: L_SHIFT = 5'd21;
            8'd152: L_SHIFT = 5'd22;
            8'd153: L_SHIFT = 5'd22;
            8'd154: L_SHIFT = 5'd22;
            8'd155: L_SHIFT = 5'd22;
            8'd156: L_SHIFT = 5'd21;
            8'd157: L_SHIFT = 5'd22;
            8'd158: L_SHIFT = 5'd20;
            8'd159: L_SHIFT = 5'd22;
            8'd160: L_SHIFT = 5'd21;
            8'd161: L_SHIFT = 5'd21;
            8'd162: L_SHIFT = 5'd21;
            8'd163: L_SHIFT = 5'd22;
            8'd164: L_SHIFT = 5'd22;
            8'd165: L_SHIFT = 5'd22;
            8'd166: L_SHIFT = 5'd18;
            8'd167: L_SHIFT = 5'd20;
            8'd168: L_SHIFT = 5'd22;
            8'd169: L_SHIFT = 5'd22;
            8'd170: L_SHIFT = 5'd22;
            8'd171: L_SHIFT = 5'd19;
            8'd172: L_SHIFT = 5'd22;
            8'd173: L_SHIFT = 5'd16;
            8'd174: L_SHIFT = 5'd20;
            8'd175: L_SHIFT = 5'd21;
            8'd176: L_SHIFT = 5'd22;
            8'd177: L_SHIFT = 5'd22;
            8'd178: L_SHIFT = 5'd20;
            8'd179: L_SHIFT = 5'd21;
            8'd180: L_SHIFT = 5'd22;
            8'd181: L_SHIFT = 5'd19;
            8'd182: L_SHIFT = 5'd22;
            8'd183: L_SHIFT = 5'd22;
            8'd184: L_SHIFT = 5'd21;
            8'd185: L_SHIFT = 5'd21;
            8'd186: L_SHIFT = 5'd22;
            8'd187: L_SHIFT = 5'd20;
            8'd188: L_SHIFT = 5'd22;
            8'd189: L_SHIFT = 5'd22;
            8'd190: L_SHIFT = 5'd22;
            8'd191: L_SHIFT = 5'd21;
            8'd192: L_SHIFT = 5'd18;
            8'd193: L_SHIFT = 5'd22;
            8'd194: L_SHIFT = 5'd18;
            8'd195: L_SHIFT = 5'd21;
            8'd196: L_SHIFT = 5'd21;
            8'd197: L_SHIFT = 5'd21;
            8'd198: L_SHIFT = 5'd22;
            8'd199: L_SHIFT = 5'd22;
            8'd200: L_SHIFT = 5'd21;
            8'd201: L_SHIFT = 5'd22;
            8'd202: L_SHIFT = 5'd15;
            8'd203: L_SHIFT = 5'd22;
            8'd204: L_SHIFT = 5'd22;
            8'd205: L_SHIFT = 5'd22;
            8'd206: L_SHIFT = 5'd22;
            8'd207: L_SHIFT = 5'd21;
            8'd208: L_SHIFT = 5'd21;
            8'd209: L_SHIFT = 5'd22;
            8'd210: L_SHIFT = 5'd20;
            8'd211: L_SHIFT = 5'd21;
            8'd212: L_SHIFT = 5'd22;
            8'd213: L_SHIFT = 5'd21;
            8'd214: L_SHIFT = 5'd22;
            8'd215: L_SHIFT = 5'd22;
            8'd216: L_SHIFT = 5'd22;
            8'd217: L_SHIFT = 5'd20;
            8'd218: L_SHIFT = 5'd22;
            8'd219: L_SHIFT = 5'd23;
            8'd220: L_SHIFT = 5'd22;
            8'd221: L_SHIFT = 5'd19;
            8'd222: L_SHIFT = 5'd22;
            8'd223: L_SHIFT = 5'd21;
            8'd224: L_SHIFT = 5'd20;
            8'd225: L_SHIFT = 5'd20;
            8'd226: L_SHIFT = 5'd20;
            default: L_SHIFT = 5'd0;
        endcase
    endfunction

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
    // We use co_base+0 for w lookup ¡ª all 4 PEs share the SAME co?  No, in
    // 4-PE arrangement each PE handles a distinct co.  But weight read is
    // serialized by co_base+0..3 inside S_FETCH; for simplicity we share.
    // Simpler: process 1 co at a time ¡ú effective NUM_PE=1. (Matches v0.)
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
