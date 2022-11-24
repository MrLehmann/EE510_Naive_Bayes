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

% TODO 1: Test with Laplace Smoothing
% TODO 2: Test with 28x28 images
% TODO 3: Test with Pixel Groups as Features
% TODO 4: Test with edge detection (vertical and horizontal)

train_digits = 50000;
test_digits = 10000;

% Pull in training data
[train_imgs, train_labels] = readMNIST('train-images-idx3-ubyte/train-images.idx3-ubyte', 'train-labels-idx1-ubyte/train-labels.idx1-ubyte', train_digits, 0);

% Create instance vector for each digit label
digit_labels = zeros(1,10);

% Get train_labels size for loops
labels_size = size(train_labels,1);

% Create matrix for each vector from 0-9 to hold cumulative training data
digit_matrix = zeros(size(train_imgs,1)*size(train_imgs,2),10);

% Loop through train_labels to calculate how many per digit
for i = 1:labels_size
    number_value = train_labels(i); % Grab digit label number
    digit_labels(number_value+1) = digit_labels(number_value+1) + 1; % Increment instance
    % Setup the digit_matrix
    num = train_labels(i); % Grab number value
    img = train_imgs(:,:,i); % Grab image
    img(img >= 0.5) = 1; % Set to either 0 or 1
    img(img < 0.5) = 0;
    vec = img(:); % Get as a vector
    % Sum vector values into single matrix
    for j = 1:size(vec,1)
        digit_matrix(j,num+1) = digit_matrix(j,num+1) + vec(j);
    end
end
% Can show percentage of each label in training set
% digit_labels/50000 * 100

% Normalize matrix?
for i = 1:10
    digit_matrix(:,i)=digit_matrix(:,i)/digit_labels(i);
end

% How to do a heatmap/colormap for each digit?
%vector_0 = digit_matrix(:,9);
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
    % Loop through each digit in the digit matrix
    for j = 1:10
        % Image vector of test image
        img = test(:);
        % Set final probability to 1
        total_prob = 1;
        % Loop through each pixel
        for k = 1:size(img,1)
            % x value (pixels)
            pixel = img(k);
            % Value (0 or 1) of each pixel within the digit matrix
            Pji = digit_matrix(k, j);
            % Pji(x) = (Pji^x)*((1-Pji)^(1-x))
            total_prob = total_prob*((Pji^pixel)*((1-Pji)^(1-pixel)));
        end
        % Put total prob per digit in max_prob matrix
        max_prob(j) = total_prob; 
    end
    [maxNum, index] = max(max_prob); % argmax P(y=j|x)
    numb = index;
    % test_labels(i)+1 gives proper index value (0-9) becomes (1-10).
    % Increment value at confusion matrix by 1.
    confusion(numb, test_labels(i)+1) = confusion(numb, test_labels(i)+1)+1;
end

confusion = (confusion./test_digit_labels)*100;
