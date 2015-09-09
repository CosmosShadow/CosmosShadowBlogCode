clear;
close all;
addpath('mnistHelper')

% 加载
images = loadMNISTImages('mnist_data/train-images.idx3-ubyte');
labels = loadMNISTLabels('mnist_data/train-labels.idx1-ubyte');

% 显示前100张
display_network(images(:,1:100));
disp(labels(1:10));