close;
clear all;

matrixRow = 1000;
matrixColumn = 2000;

tic

a = 0:1:matrixRow*matrixColumn-1;
b = 0:1:matrixRow*matrixColumn-1;
a = reshape(a, matrixRow, matrixColumn);
b = reshape(b, matrixColumn, matrixRow);
c = a * b;

size(c)

toc