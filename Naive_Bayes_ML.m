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

% Create instance vector for each digit label
digit_labels = zeros(1,10);

% Get train_labels size for loops
labels_size = size(train_labels,1);

% Loop through train_labels to calculate how many per digit
for i=1:labels_size
    number_value = train_labels(i); % Grab digit label number
    digit_labels(number_value+1) = digit_labels(number_value+1) + 1; % Increment instance
end
% Can show percentage of each label in training set
% digit_labels/50000 * 100
% Create matrix for each vector from 0-9 to hold cumulative training data
digit_matrix = zeros(size(train_imgs,1)*size(train_imgs,2),10);
% example_img = train_imgs(:,:,1); % Extract image (20x20 matrix)
% example_img(example_img >= 0.5) = 1; % Set values greater than 0.5 to 1
% example_img(example_img < 0.5) = 0;
% example_vec = example_img(:); % Set 20x20 image as single vector

% Loop through train_imgs to fill in digit_matrix
for i=1:labels_size
    num = train_labels(i); % Grab number value
    img = train_imgs(:,:,i); % Grab image
    img(img >= 0.5) = 1; % Set to either 0 or 1
    img(img < 0.5) = 0;
    vec = img(:); % Get as a vector
    % Sum vector values into single matrix
    for j=1:size(vec,1)
        digit_matrix(j,num+1) = digit_matrix(j,num+1) + vec(j);
    end
end
% Normalize matrix?
for i= 1:10
    digit_matrix(:,i)=digit_matrix(:,i)/digit_labels(i);
end

% How to do a heatmap/colormap for each digit?
vector_0 = digit_matrix(:,1);
matrix_0 = reshape(vector_0, 20, 20);
heatmap(matrix_0);
% Pull in test data
%[test_imgs test_labels] = readMNIST('t10k-images-idx3-ubyte/t10k-images.idx3-ubyte', 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte', test_digits, 0);
