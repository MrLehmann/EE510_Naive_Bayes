% Naive Bayes Project
% Final project for EE510 - Write a program to determine
% handwritten digits given the largest probability of
% the digit. Use Maximum Likelihood (ML) and 
% Maximum Posteriori Probability (MAP) approaches.
%
% @author Mike Lehmann
% @author Alec Moravec
% @date 11/5/2022
% @version 1

% TODO 1: Test with Gaussian Distribution
% TODO 2: Test with MAP instead of ML
% TODO 3: Test with Pixel Groups as Features
% TODO 4: Test with edge detection (vertical and horizontal)
% TODO 5: Test with our own handwritten digits
clear;
clc;
close all;

train_digits = 60000;
test_digits = 10000;

% Pull in training data
[train_imgs, train_labels] = readMNIST('train-images-idx3-ubyte/train-images.idx3-ubyte', 'train-labels-idx1-ubyte/train-labels.idx1-ubyte', train_digits, 0);

% Create instance vector for each digit label
digit_labels = zeros(1,10);

% Get train_labels size for loops
labels_size = size(train_labels,1);

% Create matrix for each vector from 0-9 to hold cumulative training data
mean_matrix = zeros(size(train_imgs,1)*size(train_imgs,2),10);

% Loop through train_labels to calculate how many per digit
for i = 1:labels_size
    num = train_labels(i); % Grab digit label number
    digit_labels(num+1) = digit_labels(num+1) + 1; % Increment instance
    % Setup the digit_matrix
    img = train_imgs(:,:,i); % Grab image
    vec = img(:); % Get as a vector
    % Sum vector values into single matrix
    mean_matrix(:,num+1) = mean_matrix(:,num+1) + vec;
    %for j = 1:size(vec,1)
    %    mean_matrix(j,num+1) = mean_matrix(j,num+1) + vec(j);
    %end
end

% Divide mean sum by number of images
for i = 1:10
    mean_matrix(:,i)=mean_matrix(:,i)/digit_labels(i);
end

% 3D matrix to hold Covariance matrix for each digit
cov_matrix = zeros(size(train_imgs,1)*size(train_imgs,2),size(train_imgs,1)*size(train_imgs,2),10);

% Loop through train_labels to calculate how many per digit
for i = 1:labels_size
    % Setup the digit_matrix
    num = train_labels(i); % Grab number value
    img = train_imgs(:,:,i); % Grab image
    vec = img(:); % Get as a vector
    exp_diff = (vec - mean_matrix(:, num+1));
    cov_matrix(:,:,num+1) = cov_matrix(:,:,num+1) + exp_diff*(exp_diff');
end

% Regularize cov_matrix and get inverse of cov_matrix
icov_matrix = zeros(size(train_imgs,1)*size(train_imgs,2),size(train_imgs,1)*size(train_imgs,2),10);
sigma = 0.01;
sigma_I = sigma*eye(784,784);

% Loop through each digit to normalize, regularize, and invert
for j=1:10
    % Divide cov sum by number of images-1 (because mean is not 0)
    cov_matrix(:,:,j)=cov_matrix(:,:,j)/(digit_labels(j) - 1);
    % Regularize the matrix
    cov_matrix(:,:,j) = cov_matrix(:,:,j) + sigma_I;
    % Get the inverse
    icov_matrix(:,:,j) = inv(cov_matrix(:,:,j));
end

% How to do a heatmap/colormap for each digit?
%vector_0 = digit_matrix(:,1);
%matrix_0 = reshape(vector_0, 20, 20);
%heatmap(matrix_0);

% Pull in test data
[test_imgs, test_labels] = readMNIST('t10k-images-idx3-ubyte/t10k-images.idx3-ubyte', 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte', test_digits, 0);

test_labels_size = size(test_labels, 1);
% How many of each digit in the test images
test_digit_labels = zeros(1,10);

% Create confusion matrix
confusion = zeros(10,10);
error = 0;
% Loop through every test image
for i = 1:test_labels_size
    % Do same thing for digit_labels vector to get percentages for
    % confusion matrix
    num = test_labels(i);
    test_digit_labels(num+1) = test_digit_labels(num+1)+1;
    test = test_imgs(:,:,i); % Grab test image
    max_prob = zeros(1,10); % vector to hold the probabilities for each number
    % Image vector of test image
    img = test(:);
    % Loop through each digit in the digit matrix
    for j = 1:10
        col_vec = img - mean_matrix(:,j);
        inv_mat_vec = icov_matrix(:,:,j)*col_vec;
        row_vec = col_vec';
        part_gauss_dens = row_vec*inv_mat_vec; % First part of Gaussian Density
        det_mat = det(cov_matrix(:,:,j)); % Issue here should not need if statement
        if det_mat == 0
            % Psuedo Det
            % Loop through cov_matrix diagonal
            result = 1;
            %for k = 1:400
                %if cov_matrix(k,k,j) > 0.5
            %        result = result * cov_matrix(k,k,j);
                %end
            %end
            det_mat = result;
        end
        log_det_mat = log(det_mat);
        gauss_dens = 0.5*part_gauss_dens - 0.5*log_det_mat; % Full Gaussian Density
        % Put total prob per digit in max_prob matrix
        total_prob = log(digit_labels(j)/60000) - gauss_dens; % MAP
        %total_prob = - gauss_dens; % MLE
        max_prob(j) = total_prob;
    end
    [maxNum, index] = max(max_prob); % argmax P(y=j|x)
    if index ~= (num + 1)
        error = error + 1;
    end
    % test_labels(i)+1 gives proper index value (0-9) becomes (1-10).
    % Increment value at confusion matrix by 1.
    % (row, column)
    confusion(num+1, index) = confusion(num+1, index)+1;
end

confusion = (confusion./test_digit_labels)*100;
confusion
(error/10000)*100
