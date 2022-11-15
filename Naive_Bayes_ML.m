% Naive Bayes Project
% Final project for EE510 - Write a program to determine
% handwritten digits given the largest probability of
% the digit. Use Maximum Likelihood (ML) and 
% Maximum Posteriori Probability (MAP) approaches.
%
% @author Mike Lehmann
% @author Alec ???
% @date 11/5/2022
% @version 1

train_digits = 50000;
test_digits = 10000;

% Pull in training data
[train_imgs train_labels] = readMNIST('train-images-idx3-ubyte/train-images.idx3-ubyte', 'train-labels-idx1-ubyte/train-labels.idx1-ubyte', train_digits, 0);

% Pull in test data
[test_imgs test_labels] = readMNIST('t10k-images-idx3-ubyte/t10k-images.idx3-ubyte', 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte', test_digits, 0);
