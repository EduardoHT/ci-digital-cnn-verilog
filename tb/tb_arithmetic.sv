`timescale 1ns/1ps
module tb_arithmetic;
    reg [7:0] a;reg signed [7:0] b;reg signed [31:0] acc;
    wire signed [31:0] sum;
    reg [7:0] shift;wire [7:0] result;
    reg [7:0] p0,p1,p2,p3;wire [7:0] pmax;
    mac_u8s8 mac(.a(a),.b(b),.acc(acc),.sum(sum));
    relu_requant rq(.acc(acc),.shift(shift),.result(result));
    max4_u8 pool(.a(p0),.b(p1),.c(p2),.d(p3),.y(pmax));
    initial begin
        a=255;b=-128;acc=0;shift=0;p0=0;p1=0;p2=0;p3=0;
        #1;if(sum!==-32'sd32640) $fatal(1,"uint8/int8 -128");
        b=127;acc=-100;#1;if(sum!==32'sd32285) $fatal(1,"MAC soma/sinal");
        acc=-32'sd1;shift=3;#1;if(result!==0) $fatal(1,"ReLU negativa");
        acc=3;#1;if(result!==0) $fatal(1,"round abaixo da metade");
        acc=4;#1;if(result!==1) $fatal(1,"round half-up");
        acc=2044;#1;if(result!==255) $fatal(1,"saturacao");
        acc=32'sh7fffffff;shift=1;#1;if(result!==255) $fatal(1,"round overflow");
        acc=7;shift=0;#1;if(result!==7) $fatal(1,"shift zero");
        acc=-32'sd2147483647-1;shift=0;#1;if(result!==0) $fatal(1,"INT32_MIN");
        p0=0;p1=128;p2=255;p3=2;#1;if(pmax!==255) $fatal(1,"pool unsigned");
        $display("PASS_ARITHMETIC tests=10");$finish;
    end
endmodule
