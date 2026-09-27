// Pooling 2x2: comparacoes unsigned, sem somar nem tirar media.
module max4_u8(
    input wire [7:0] a,b,c,d,
    output wire [7:0] y
);
    wire [7:0] ab = (a>b) ? a : b;
    wire [7:0] cd = (c>d) ? c : d;
    assign y = (ab>cd) ? ab : cd;
endmodule
