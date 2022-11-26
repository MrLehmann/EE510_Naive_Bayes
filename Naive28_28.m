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
% TODO 5: Test with our own handwritten digits
clear;
clc;

data_size = 1000; % Size per digit file

% Create matrix for each vector from 0-9 to hold cumulative training data
digit_matrix = zeros(784,10);

% Loop through each digit
for i=1:10
    filename = sprintf('data%d', i-1); % Grab new file name
    fid=fopen(filename,'r');
    
    % Loop through each image
    for j=1:data_size
        [train_data ,N]=fread(fid,[28 28],'uchar'); % read in the first training
                                 % example and store it in a 28x28  
        train_data(train_data < (255/2)) = 0;
        train_data(train_data >= (255/2)) = 1;
        vec = train_data(:); % Get as a vector

        % Sum vector values into single matrix
        for k=1:size(vec,1)
            digit_matrix(k,i) = digit_matrix(k,i) + vec(k);
        end
    end
    fclose(fid);
end

% How to do a heatmap/colormap for each digit?
vector_0 = digit_matrix(:,7);
matrix_0 = reshape(vector_0, 28, 28);
heatmap(matrix_0);

fid = fopen('testlabels', 'r');
[test_labels, label_size] = fread(fid, 'uchar'); % Get test_labels
fclose(fid); 

fid = fopen('testimages', 'r');
for i=1:label_size
    [test_img, test_size] = fread(fid, [28 28], 'uchar'); % Get test_images
end
fclose(fid);
