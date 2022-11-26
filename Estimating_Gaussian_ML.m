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

% TODO 1: Test with Gaussian Distribution
% TODO 2: Test with MAP instead of ML
% TODO 3: Test with Pixel Groups as Features
% TODO 4: Test with edge detection (vertical and horizontal)
% TODO 5: Test with our own handwritten digits
clear;
clc;

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
    number_value = train_labels(i); % Grab digit label number
    digit_labels(number_value+1) = digit_labels(number_value+1) + 1; % Increment instance
    % Setup the digit_matrix
    num = train_labels(i); % Grab number value
    img = train_imgs(:,:,i); % Grab image
    vec = img(:); % Get as a vector
    % Sum vector values into single matrix
    for j = 1:size(vec,1)
        mean_matrix(j,num+1) = mean_matrix(j,num+1) + vec(j);
    end
end

% Divide mean sum by number of images
for i = 1:10
    mean_matrix(:,i)=mean_matrix(:,i)/digit_labels(i);
end

% 3D matrix to hold Covariance matrix for each digit
cov_matrix = zeros(size(train_imgs,1)*size(train_imgs,2),size(train_imgs,1)*size(train_imgs,2),10);

% Testing sort methods
[train_labels, idx] = sort(train_labels);
train_imgs = train_imgs(:,:,idx);

for j=1:10
    total_sum = 0;
    for i=1:size(digit_labels(j))
        img = train_imgs(:,:,i); % Grab image
        vec = img(:); % Get as a vector
        %((vec - mean_matrix(:,j)))
        cov_matrix(:,:,j) = cov_matrix(:,:,j) + ((vec - mean_matrix(j))*(vec - mean_matrix(j))');
    end
    
end

% Divide cov sum by number of images
for i = 1:10
    cov_matrix(:,:,i)=cov_matrix(:,:,i)/digit_labels(i);
end

% Regularize cov_matrix and get inverse of cov_matrix
icov_matrix = zeros(size(train_imgs,1)*size(train_imgs,2),size(train_imgs,1)*size(train_imgs,2),10);
sigma = 0.1;
for i = 1:10
    cov_matrix(:,:,i)=cov_matrix(:,:,i) + sigma * eye(400,400);
    icov_matrix(:,:,i) = inv(cov_matrix(:,:,i));
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
% Loop through every test image
for i = 1:test_labels_size
    % Do same thing for digit_labels vector to get percentages for
    % confusion matrix
    number_value = test_labels(i);
    test_digit_labels(number_value+1) = test_digit_labels(number_value+1)+1;
    test = test_imgs(:,:,i); % Grab test image
    max_prob = zeros(1,10); % vector to hold the probabilities for each number
    % Image vector of test image
    img = test(:);
    % Loop through each digit in the digit matrix
    for j = 1:10
        % Set final probability to 1 (or 0 if doing sum of logs)
        total_prob = 0.5*(img - mean_matrix(j))'*icov_matrix(:,:,j)*(img - mean_matrix);
        total_prob = total_prob - 0.5*log(det(cov_matrix(:,:,j)));
        % Put total prob per digit in max_prob matrix
        total_prob = log(digit_labels(j))-total_prob;
    end
    [maxNum, index] = max(total_prob); % argmax P(y=j|x)
    numb = index;
    % test_labels(i)+1 gives proper index value (0-9) becomes (1-10).
    % Increment value at confusion matrix by 1.
    confusion(numb, test_labels(i)+1) = confusion(numb, test_labels(i)+1)+1;
end

confusion = (confusion./test_digit_labels)*100;
