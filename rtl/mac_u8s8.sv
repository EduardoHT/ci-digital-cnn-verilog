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
