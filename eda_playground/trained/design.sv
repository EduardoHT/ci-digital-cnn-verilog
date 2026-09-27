// MAC combinacional compartilhado. O registrador acc fica no controlador.
// a: unsigned 0..255; b: signed -128..127. Nao reinterpretar a como int8!
module mac_u8s8(
    input  wire [7:0] a,
    input  wire signed [7:0] b,
    input  wire signed [31:0] acc,
    output wire signed [31:0] sum
);
    wire signed [8:0] a_signed = $signed({1'b0,a});
    wire signed [16:0] product = a_signed * b;
    assign sum = acc + {{15{product[16]}},product};
endmodule


// ReLU -> arredondamento half-up -> shift -> saturacao unsigned de 8 bits.
// O intermediario de 64 bits protege a soma do termo de arredondamento.
// O exportador garante shift entre 0 e 30 e acumulacao segura em int32.
module relu_requant(
    input  wire signed [31:0] acc,
    input  wire [7:0] shift,
    output reg [7:0] result
);
    reg signed [63:0] wide;
    always @* begin
        result = 8'd0;
        wide = 64'sd0;
        if (acc > 0 && shift <= 30) begin
            wide = {{32{acc[31]}},acc};
            if (shift != 0)
                wide = (wide + (64'sd1 << (shift-1))) >>> shift;
            if (wide > 255) result = 8'd255;
            else result = wide[7:0];
        end
    end
endmodule


// Pooling 2x2: comparacoes unsigned, sem somar nem tirar media.
module max4_u8(
    input wire [7:0] a,b,c,d,
    output wire [7:0] y
);
    wire [7:0] ab = (a>b) ? a : b;
    wire [7:0] cd = (c>d) ? c : d;
    assign y = (ab>cd) ? ab : cd;
endmodule


// CI Digital v2 -- inferencia inteira REAL, nao apenas contagem de ciclos.
// 28x28x1 -> Conv 8x3x3 valid -> ReLU/requant -> Pool 2x2
// -> Conv 16x3x3 valid -> ReLU/requant -> Pool 2x2 -> Dense 400x5 -> argmax.
//
// Contrato de entrada: escrever TODOS os 784 pixels enquanto busy=0;
// dar start por 1 ciclo apenas depois da carga. input_addr = linha*28+coluna.
// Pesos O,I,Ky,Kx; feature maps C,H,W. Kernels NAO sao invertidos.
// LOAD_FILES=0 e usado pelo testbench autocontido do EDA Playground.
// Memorias e enderecamento foram priorizados para SIMULACAO didatica.
// Nao ha afirmacao de sintese, timing, consumo ou validacao em FPGA.
module tiny_cnn #(
    parameter integer LOAD_FILES = 1,
    parameter WEIGHTS_DIR = "exports/trained"
)(
    input wire clk, rst, start,
    input wire input_we,
    input wire [9:0] input_addr,
    input wire [7:0] input_data,
    output reg busy, done,
    output reg [2:0] class_id,
    output reg [31:0] cycle_count,
    output wire signed [31:0] logit0,logit1,logit2,logit3,logit4
);
    // Memorias de parametros (substituidas automaticamente apos cada treino).
    reg signed [7:0] conv1_w [0:71];
    reg signed [31:0] conv1_b [0:7];
    reg signed [7:0] conv2_w [0:1151];
    reg signed [31:0] conv2_b [0:15];
    reg signed [7:0] fc_w [0:1999];
    reg signed [31:0] fc_b [0:4];
    reg [7:0] shifts [0:1];
    // Memorias de trabalho; nao precisam ser zeradas a cada frame.
    reg [7:0] image_mem [0:783];
    reg [7:0] conv1_mem [0:5407];
    reg [7:0] pool1_mem [0:1351];
    reg [7:0] conv2_mem [0:1935];
    reg [7:0] pool2_mem [0:399];
    reg signed [31:0] logits [0:4];
    assign logit0=logits[0]; assign logit1=logits[1];
    assign logit2=logits[2]; assign logit3=logits[3]; assign logit4=logits[4];

    initial begin
        if (LOAD_FILES != 0) begin
            $readmemh({WEIGHTS_DIR,"/conv1_w.mem"},conv1_w);
            $readmemh({WEIGHTS_DIR,"/conv1_b.mem"},conv1_b);
            $readmemh({WEIGHTS_DIR,"/conv2_w.mem"},conv2_w);
            $readmemh({WEIGHTS_DIR,"/conv2_b.mem"},conv2_b);
            $readmemh({WEIGHTS_DIR,"/fc_w.mem"},fc_w);
            $readmemh({WEIGHTS_DIR,"/fc_b.mem"},fc_b);
            $readmemh({WEIGHTS_DIR,"/shifts.mem"},shifts);
        end
    end

    localparam [3:0] IDLE=0, DOT_INIT=1, DOT_MAC=2, DOT_STORE=3,
                     POOL=4, ARG_INIT=5, ARG_SCAN=6, FINISH=7;
    reg [3:0] state;
    reg [1:0] layer; // 0=conv1; 1=conv2; 2=dense
    integer out_idx, tap_idx, pool_idx, arg_idx;
    reg signed [31:0] acc, best_logit;
    reg [7:0] mac_a;
    reg signed [7:0] mac_b;
    reg signed [31:0] dot_bias;
    wire signed [31:0] mac_sum;
    wire [7:0] quantized;
    reg [7:0] current_shift;
    integer dot_len, out_len;
    integer oc,oy,ox,ic,ky,kx,addr,waddr;
    reg [7:0] pa,pb,pc,pd;
    wire [7:0] pooled;
    integer pool_len,pchan,py,px,pbase;

    mac_u8s8 u_mac(.a(mac_a),.b(mac_b),.acc(acc),.sum(mac_sum));
    relu_requant u_rq(.acc(acc),.shift(current_shift),.result(quantized));
    max4_u8 u_pool(.a(pa),.b(pb),.c(pc),.d(pd),.y(pooled));

    // Selecao combinacional dos operandos. Ha UMA multiplicacao no MAC.
    // Divisoes e restos sao por constantes de dimensao, para clareza didatica.
    always @* begin
        mac_a=0; mac_b=0; dot_bias=0; current_shift=0;
        dot_len=1; out_len=1;
        oc=0;oy=0;ox=0;ic=0;ky=0;kx=0;addr=0;waddr=0;
        case(layer)
            0: begin
                dot_len=9; out_len=5408; current_shift=shifts[0];
                oc=out_idx/676; oy=(out_idx%676)/26; ox=out_idx%26;
                ky=tap_idx/3; kx=tap_idx%3;
                addr=(oy+ky)*28+ox+kx; waddr=oc*9+tap_idx;
                if (oc<8 && oc>=0) dot_bias=conv1_b[oc];
                if (addr>=0 && addr<784) mac_a=image_mem[addr];
                if (waddr>=0 && waddr<72) mac_b=conv1_w[waddr];
            end
            1: begin
                dot_len=72; out_len=1936; current_shift=shifts[1];
                oc=out_idx/121; oy=(out_idx%121)/11; ox=out_idx%11;
                ic=tap_idx/9; ky=(tap_idx%9)/3; kx=tap_idx%3;
                addr=ic*169+(oy+ky)*13+ox+kx; waddr=oc*72+tap_idx;
                if (oc<16 && oc>=0) dot_bias=conv2_b[oc];
                if (addr>=0 && addr<1352) mac_a=pool1_mem[addr];
                if (waddr>=0 && waddr<1152) mac_b=conv2_w[waddr];
            end
            2: begin
                dot_len=400; out_len=5;
                waddr=out_idx*400+tap_idx;
                if (out_idx>=0 && out_idx<5) dot_bias=fc_b[out_idx];
                if (tap_idx>=0 && tap_idx<400) mac_a=pool2_mem[tap_idx];
                if (waddr>=0 && waddr<2000) mac_b=fc_w[waddr];
            end
            default: begin end
        endcase
    end

    always @* begin
        pa=0;pb=0;pc=0;pd=0;pool_len=1;
        pchan=0;py=0;px=0;pbase=0;
        if (layer==0) begin
            pool_len=1352;
            pchan=pool_idx/169; py=(pool_idx%169)/13; px=pool_idx%13;
            pbase=pchan*676+(2*py)*26+2*px;
            if (pbase>=0 && pbase+27<5408) begin
                pa=conv1_mem[pbase];pb=conv1_mem[pbase+1];
                pc=conv1_mem[pbase+26];pd=conv1_mem[pbase+27];
            end
        end else if (layer==1) begin
            pool_len=400;
            pchan=pool_idx/25; py=(pool_idx%25)/5; px=pool_idx%5;
            pbase=pchan*121+(2*py)*11+2*px;
            if (pbase>=0 && pbase+12<1936) begin
                pa=conv2_mem[pbase];pb=conv2_mem[pbase+1];
                pc=conv2_mem[pbase+11];pd=conv2_mem[pbase+12];
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state<=IDLE;layer<=0;busy<=0;done<=0;class_id<=0;cycle_count<=0;
            out_idx<=0;tap_idx<=0;pool_idx<=0;arg_idx<=0;acc<=0;best_logit<=0;
        end else begin
            done<=0; // pulso por exatamente um ciclo
            if (!busy && input_we && input_addr<784)
                image_mem[input_addr]<=input_data;
            if (busy) cycle_count<=cycle_count+1;
            case(state)
                IDLE: if (start) begin
                    busy<=1;cycle_count<=0;layer<=0;out_idx<=0;state<=DOT_INIT;
                end
                DOT_INIT: begin
                    acc<=dot_bias;tap_idx<=0;state<=DOT_MAC;
                end
                DOT_MAC: begin
                    acc<=mac_sum;
                    if (tap_idx==dot_len-1) state<=DOT_STORE;
                    else tap_idx<=tap_idx+1;
                end
                DOT_STORE: begin
                    // Estado separado: acc JA contem o ultimo produto.
                    case(layer)
                        0: conv1_mem[out_idx]<=quantized;
                        1: conv2_mem[out_idx]<=quantized;
                        2: logits[out_idx]<=acc;
                        default: begin end
                    endcase
                    if (out_idx==out_len-1) begin
                        if (layer==2) state<=ARG_INIT;
                        else begin pool_idx<=0;state<=POOL;end
                    end else begin out_idx<=out_idx+1;state<=DOT_INIT;end
                end
                POOL: begin
                    if (layer==0) pool1_mem[pool_idx]<=pooled;
                    else pool2_mem[pool_idx]<=pooled;
                    if (pool_idx==pool_len-1) begin
                        layer<=layer+1;out_idx<=0;state<=DOT_INIT;
                    end else pool_idx<=pool_idx+1;
                end
                ARG_INIT: begin
                    best_logit<=logits[0];class_id<=0;arg_idx<=1;state<=ARG_SCAN;
                end
                ARG_SCAN: begin
                    // > e nao >=: em empate, vence o menor indice, como np.argmax.
                    if (logits[arg_idx]>best_logit) begin
                        best_logit<=logits[arg_idx];class_id<=arg_idx[2:0];
                    end
                    if (arg_idx==4) state<=FINISH;
                    else arg_idx<=arg_idx+1;
                end
                FINISH: begin done<=1;busy<=0;state<=IDLE;end
                default: begin state<=IDLE;busy<=0;end
            endcase
        end
    end
endmodule
