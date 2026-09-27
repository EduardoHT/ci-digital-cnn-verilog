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
