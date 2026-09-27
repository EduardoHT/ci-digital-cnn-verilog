`timescale 1ns/1ps
// Verificacao completa: TODOS os feature maps, logits e classes.
// Os valores esperados vem da referencia NUMPY INTEIRA, nao de float32.
module tb_cnn #(
    parameter integer NTEST=20,
    parameter WEIGHTS_DIR="exports/trained",
    parameter VECTOR_DIR="exports/trained/vectors"
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
    always #5 clk=~clk; // periodo FICTICIO 10 ns; nao e medicao de FPGA

    reg [7:0] images [0:NTEST*784-1];
    reg [7:0] labels [0:NTEST-1];
    reg [7:0] exp_c1 [0:NTEST*5408-1];
    reg [7:0] exp_p1 [0:NTEST*1352-1];
    reg [7:0] exp_c2 [0:NTEST*1936-1];
    reg [7:0] exp_p2 [0:NTEST*400-1];
    reg signed [31:0] exp_logits [0:NTEST*5-1];
    reg [7:0] exp_class [0:NTEST-1];
    integer t,j,wait_cycles,labeled_correct,labeled_count;

    task require_file(input string path);
        integer fd;
        begin
            fd=$fopen(path,"r");
            if (fd==0) $fatal(1,"Arquivo ausente: %s",path);
            $fclose(fd);
        end
    endtask
    initial begin
        require_file({WEIGHTS_DIR,"/conv1_w.mem"});
        require_file({WEIGHTS_DIR,"/conv1_b.mem"});
        require_file({WEIGHTS_DIR,"/conv2_w.mem"});
        require_file({WEIGHTS_DIR,"/conv2_b.mem"});
        require_file({WEIGHTS_DIR,"/fc_w.mem"});
        require_file({WEIGHTS_DIR,"/fc_b.mem"});
        require_file({WEIGHTS_DIR,"/shifts.mem"});
        require_file({VECTOR_DIR,"/images.mem"});
        require_file({VECTOR_DIR,"/labels.mem"});
        require_file({VECTOR_DIR,"/expected_conv1.mem"});
        require_file({VECTOR_DIR,"/expected_pool1.mem"});
        require_file({VECTOR_DIR,"/expected_conv2.mem"});
        require_file({VECTOR_DIR,"/expected_pool2.mem"});
        require_file({VECTOR_DIR,"/expected_logits.mem"});
        require_file({VECTOR_DIR,"/expected_class.mem"});
        $readmemh({VECTOR_DIR,"/images.mem"},images);
        $readmemh({VECTOR_DIR,"/labels.mem"},labels);
        $readmemh({VECTOR_DIR,"/expected_conv1.mem"},exp_c1);
        $readmemh({VECTOR_DIR,"/expected_pool1.mem"},exp_p1);
        $readmemh({VECTOR_DIR,"/expected_conv2.mem"},exp_c2);
        $readmemh({VECTOR_DIR,"/expected_pool2.mem"},exp_p2);
        $readmemh({VECTOR_DIR,"/expected_logits.mem"},exp_logits);
        $readmemh({VECTOR_DIR,"/expected_class.mem"},exp_class);
        if ($test$plusargs("WAVES")) begin
            $dumpfile("cnn.vcd");
            $dumpvars(0,clk,rst,start,input_we,busy,done,class_id,cycle_count,
                        dut.state,dut.layer,dut.out_idx,dut.tap_idx,dut.acc);
        end
        #1;
        for(j=0;j<72;j=j+1) if ((^dut.conv1_w[j])===1'bx) $fatal(1,"conv1_w incompleto");
        for(j=0;j<1152;j=j+1) if ((^dut.conv2_w[j])===1'bx) $fatal(1,"conv2_w incompleto");
        for(j=0;j<2000;j=j+1) if ((^dut.fc_w[j])===1'bx) $fatal(1,"fc_w incompleto");
        for(j=0;j<8;j=j+1) if ((^dut.conv1_b[j])===1'bx) $fatal(1,"conv1_b incompleto");
        for(j=0;j<16;j=j+1) if ((^dut.conv2_b[j])===1'bx) $fatal(1,"conv2_b incompleto");
        for(j=0;j<5;j=j+1) if ((^dut.fc_b[j])===1'bx) $fatal(1,"fc_b incompleto");
        for(j=0;j<2;j=j+1)
            if ((^dut.shifts[j])===1'bx || dut.shifts[j]>30) $fatal(1,"Shift invalido");
        for(j=0;j<NTEST*784;j=j+1) if ((^images[j])===1'bx) $fatal(1,"Imagens incompletas");
        labeled_correct=0;labeled_count=0;
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
            if (busy!==1'b0) $fatal(1,"busy deveria cair junto de done");
            if (cycle_count!==32'd206520) $fatal(1,"Ciclos inesperados: %0d",cycle_count);
            for(j=0;j<5408;j=j+1)
                if (dut.conv1_mem[j]!==exp_c1[t*5408+j]) $fatal(1,"conv1 t=%0d idx=%0d got=%0d exp=%0d",t,j,dut.conv1_mem[j],exp_c1[t*5408+j]);
            for(j=0;j<1352;j=j+1)
                if (dut.pool1_mem[j]!==exp_p1[t*1352+j]) $fatal(1,"pool1 t=%0d idx=%0d",t,j);
            for(j=0;j<1936;j=j+1)
                if (dut.conv2_mem[j]!==exp_c2[t*1936+j]) $fatal(1,"conv2 t=%0d idx=%0d got=%0d exp=%0d",t,j,dut.conv2_mem[j],exp_c2[t*1936+j]);
            for(j=0;j<400;j=j+1)
                if (dut.pool2_mem[j]!==exp_p2[t*400+j]) $fatal(1,"pool2 t=%0d idx=%0d",t,j);
            for(j=0;j<5;j=j+1)
                if (dut.logits[j]!==exp_logits[t*5+j]) $fatal(1,"logit t=%0d idx=%0d got=%0d exp=%0d",t,j,dut.logits[j],exp_logits[t*5+j]);
            if ({5'b0,class_id}!==exp_class[t]) $fatal(1,"argmax t=%0d",t);
            if (labels[t]!=255) begin
                labeled_count=labeled_count+1;
                if (class_id==labels[t]) labeled_correct=labeled_correct+1;
            end
            $display("PASS vector=%0d class=%0d cycles=%0d logits=[%0d %0d %0d %0d %0d]",t,class_id,cycle_count,l0,l1,l2,l3,l4);
            @(negedge clk);
            if (done!==1'b0) $fatal(1,"done durou mais de um ciclo");
        end
        $display("PASS_ALL vectors=%0d exact_values_per_vector=9102 labeled_correct=%0d/%0d",NTEST,labeled_correct,labeled_count);
        $finish;
    end
    initial begin #1000000000; $fatal(1,"Watchdog global");end
endmodule
