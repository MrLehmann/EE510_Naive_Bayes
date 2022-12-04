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

data_size = 5000; % Size per digit file

% Create matrix for each vector from 0-9 to hold cumulative training data
digit_matrix = zeros(784,10);
file_label = 'traininglabels';
fid_label = fopen(file_label,'r');
%filename1 = 'pixel-frame-0.png';%sprintf('data%d', i-1); % Grab new file name
%fid1=fopen(filename1,'r');
[img, map, alpha] = imread("pixil-frame-0.png");
heatmap(alpha);

label = fread(fid_label,'char');
labels = str2num(char(label));

% Create instance vector for each digit label
digit_labels = zeros(1,10);
row = 28;
% Loop through each digit
for i=1:data_size
    number_value = labels(i); % Grab digit label number
    digit_labels(number_value+1) = digit_labels(number_value+1) + 1; % Increment instance
    [train_data,N]=fread(fid1,[28 28],'char'); % read in the first training
                                 % example and store it in a 28x28  
    %train_data = str2num(char(train_data));
    for l = 1:28
        train_data(:,l) = circshift(train_data(:,l),(-l+1),2);
    end
    
    %train_data = fliplr(train_data);
    train_data = rot90(train_data);
    train_data = rot90(train_data);
    train_data = rot90(train_data);
    train_data = fliplr(train_data);
    for l = 1:28
        train_data(l,:) = circshift(train_data(l,:),(-l+1),2);
    end
    %train_data(train_data < (255/2)) = 0;
    %train_data(train_data >= (255/2)) = 1;
    train_data(train_data <= 32) = 0;
    train_data(train_data > 32) = 1;
    vec = train_data(:); % Get as a vector
    

    % Sum vector values into single matrix
    for k=1:size(vec,1)
        digit_matrix(k,number_value+1) = digit_matrix(k,number_value+1) + vec(k);
    end
end
fclose(fid1);
fclose(fid_label);

% How to do a heatmap/colormap for each digit?
vector_0 = digit_matrix(:,1);
matrix_0 = reshape(vec, 28, 28);
heatmap(matrix_0);

fid = fopen('testlabels', 'r');
[test_labels, label_size] = fread(fid, 'uchar'); % Get test_labels
fclose(fid); 

fid = fopen('testimages', 'r');
for i=1:label_size
    [test_img, test_size] = fread(fid, [28 28], 'uchar'); % Get test_images
end
fclose(fid);
