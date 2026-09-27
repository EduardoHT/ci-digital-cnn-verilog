`timescale 1ns/1ps
// Campanha em escala: executa o tiny_cnn original e confere logits/classe.
// A verificacao integral dos mapas internos permanece em tb_cnn.sv.
module tb_dataset_batch #(
    parameter integer NTEST=200,
    parameter WEIGHTS_DIR="exports/demo",
    parameter VECTOR_DIR="build/full_dataset/shard_0"
);
    reg clk=0,rst=1,start=0,input_we=0;
    reg [9:0] input_addr=0;
    reg [7:0] input_data=0;
    wire busy,done;
    wire [2:0] class_id;
    wire [31:0] cycle_count;
    wire signed [31:0] l0,l1,l2,l3,l4;
    tiny_cnn #(.LOAD_FILES(1),.WEIGHTS_DIR(WEIGHTS_DIR)) dut(
        .clk(clk),.rst(rst),.start(start),.input_we(input_we),
        .input_addr(input_addr),.input_data(input_data),.busy(busy),.done(done),
        .class_id(class_id),.cycle_count(cycle_count),
        .logit0(l0),.logit1(l1),.logit2(l2),.logit3(l3),.logit4(l4));
    always #5 clk=~clk;

    reg [7:0] images [0:NTEST*784-1];
    reg [7:0] labels [0:NTEST-1];
    reg signed [31:0] exp_logits [0:NTEST*5-1];
    reg [7:0] exp_class [0:NTEST-1];
    integer t,j,wait_cycles;

    task require_file(input string path);
        integer fd;
        begin
            fd=$fopen(path,"r");
            if (fd==0) $fatal(1,"Arquivo ausente: %s",path);
            $fclose(fd);
        end
    endtask

    initial begin
        require_file({VECTOR_DIR,"/images.mem"});
        require_file({VECTOR_DIR,"/labels.mem"});
        require_file({VECTOR_DIR,"/expected_logits.mem"});
        require_file({VECTOR_DIR,"/expected_class.mem"});
        $readmemh({VECTOR_DIR,"/images.mem"},images);
        $readmemh({VECTOR_DIR,"/labels.mem"},labels);
        $readmemh({VECTOR_DIR,"/expected_logits.mem"},exp_logits);
        $readmemh({VECTOR_DIR,"/expected_class.mem"},exp_class);
        repeat(3) @(negedge clk);
        rst=0;
        for(t=0;t<NTEST;t=t+1) begin
            for(j=0;j<784;j=j+1) begin
                @(negedge clk);
                input_we=1;input_addr=j;input_data=images[t*784+j];
            end
            @(negedge clk);input_we=0;start=1;
            @(negedge clk);start=0;
            wait_cycles=0;
            while (!done && wait_cycles<220000) begin
                @(negedge clk);wait_cycles=wait_cycles+1;
            end
            if (!done) $fatal(1,"Timeout na amostra %0d",t);
            if (busy!==1'b0) $fatal(1,"busy ativo junto de done");
            if (cycle_count!==32'd206520) $fatal(1,"Ciclos inesperados t=%0d got=%0d",t,cycle_count);
            if (l0!==exp_logits[t*5+0] || l1!==exp_logits[t*5+1] ||
                l2!==exp_logits[t*5+2] || l3!==exp_logits[t*5+3] ||
                l4!==exp_logits[t*5+4])
                $fatal(1,"Logits divergentes t=%0d",t);
            if ({5'b0,class_id}!==exp_class[t]) $fatal(1,"Classe divergente t=%0d",t);
            $display("RESULT,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,1",
                     t,labels[t],class_id,l0,l1,l2,l3,l4,cycle_count);
            @(negedge clk);
            if (done!==1'b0) $fatal(1,"done durou mais de um ciclo t=%0d",t);
        end
        $display("PASS_DATASET_BATCH vectors=%0d",NTEST);
        $finish;
    end
    initial begin #10000000000; $fatal(1,"Watchdog global"); end
endmodule
