// adder

`timescale 1ps/10ns

module histogram_bin (
    input increment;
    input read;
    input reset;
    output [63:0] out;
);

    reg [63:0] count;

    always @(posedge reset) begin
        count = 64'd0;
        out = 64'd0;
    end
    
    always @(posedge increment) begin
        count += 1;
    end

    always @(posedge read) begin
        out = count;
    end

endmodule