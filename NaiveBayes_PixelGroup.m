% Naive Bayes Project
% Final project for EE510 - Write a program to determine
% handwritten digits given the largest probability of
% the digit. Use Maximum Likelihood (ML) and 
% Maximum Posteriori Probability (MAP) approaches.
%
% @author Mike Lehmann
% @author Alec Moravec
% @date 12/7/2022
% @version 1

% Pixel Groups as Features
% Approach: 
% 1.Load in training labels and images. 
% 2.Transform each training image into matrix of features
% 3.Count the number of each feature at each location 
% 4.Count the number of each digit in the training data
% 5.For each feature type in each location and each label do that count/the
% total count for that label in the training data. This is the Likelyhood
% 6.Laplace smooth - for the count of each feature in each location for
% each digit add k (could be 1). For the count of training examples from
% this class add k*V (V is the number of possible values of the feature)
% 7.Load in Test data

clear;
close all;
clc;

train_digits = 60000;
test_digits = 10000;


% Pull in training data
[train_imgs, train_labels] = readMNIST('train-images-idx3-ubyte/train-images.idx3-ubyte', 'train-labels-idx1-ubyte/train-labels.idx1-ubyte', train_digits, 0);

% Create instance vector for each digit label
digit_labels = zeros(1,10);

% Get train_labels size for loops
labels_size = size(train_labels,1);

%make picture a matrix of Gij
%2x2 overlapping

%Initialze matrix. 4-D matrix.
%First two parameters are i,j coordinates of pixel group feature
%Third parameter is the index of the training data (nth image)
%Fourth parameter is the index of the pixel group feature
%trainG = zeros(size(train_imgs,1)-1,size(train_imgs,2)-1, labels_size, 4); 

%Initialize 3-D matrix. Each layer is a 2D matrix. Each entry in that
%matrix is the pixel group ID for that region of the image. This is done
%for each image in the training data
imgG = zeros(size(train_imgs,1)-1,size(train_imgs,2)-1, labels_size); 

%Number of individual features in the transformed image
%lenG = (size(train_imgs,1)-1)*(size(train_imgs,2)-1);

%Matrix holding count of each group value for each location in the image
%for each digit
countG = zeros((size(train_imgs,1)-1), (size(train_imgs,2)-1), 16, 10);

%Vector that represents an individual feature. This will be turned into a
%number fore simplicity.
Gij = zeros(1,4);

sumG = 0;
labelG = 0;
for i = 1:labels_size
    img = train_imgs(:,:,i); % Grab image
    % Set to either 0 or 1. %fore/background the image
    img(img >= 0.5) = 1; 
    img(img < 0.5) = 0;
    %vec = img(:); % Get as a vector
    labelG = train_labels(i); %grab what the label of this training digit is
    digit_labels(labelG+1) = digit_labels(labelG+1) + 1; %Add to count of the total of this digit in the training data
    for j = 1:size(train_imgs,1)-1 %The groups are overlapping. The last pixel location of each row does not have a pixel group due to the 2X2 size

        for k= 1:size(train_imgs,2)-1
            Gij = [img(j,k), img(j+1,k), img(j,k+1), img(j+1,k+1)];
            %trainG(j,k,i,:) = Gij; %make this faster? takes 28 seconds
            sumG = Gij(1) + Gij(2)*2 + Gij(3)*4 + Gij(4)*8;
            %imgG(j,k,i) = sum;
            
            %Add to count of the feature at that location for that digit 
            countG(j,k,sumG+1,labelG+1) = countG(j,k, sumG+1,labelG+1)+ 1;
        end
    end
end

% %heatmaps
% heatmap(countG(:,:,1,1));

% Laplace Smoothing. Adding Ls to numerator and Ls*V to the denomenator
las = 1; %Laplace smoothing constant
V = 16;% number of states that the featurs can take
lasV = las*V;
lsmooth = zeros((size(train_imgs,1)-1), (size(train_imgs,2)-1), 16, 10);
for i = 1:10
    lsmooth(:,:,:,i) = (countG(:,:,:,i) + las)/(digit_labels(i) + lasV);
end
%heatmap(lsmooth(:,:,1,1));


% Pull in test data
[test_imgs, test_labels] = readMNIST('t10k-images-idx3-ubyte/t10k-images.idx3-ubyte', 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte', test_digits, 0);

%Size of the test label vector
test_labels_size = size(test_labels, 1);

% How many of each digit in the test images
test_digit_labels = zeros(1,10);

%make picture a matrix of Gij
%2x2 overlapping

%Initialize 3-D matrix. Each layer is a 2D matrix. Each entry in that
%matrix is the pixel group ID for that region of the image. This is done
%for each image in the training data
test_imgG = zeros(size(test_imgs,1)-1,size(test_imgs,2)-1, test_labels_size); 

%Matrix holding count of each group value for each location in the image
%for each digit
test_countG = zeros((size(test_imgs,1)-1), (size(test_imgs,2)-1), 16, 10);

%Vector that represents an individual feature. This will be turned into a
%number fore simplicity.
Gij = zeros(4,1);

%Vector containing probabilities for each digit
Pij = zeros(1,10);

logPtotal = zeros(1,10);


%vector of priors(odds of getting that class at random)
Pc = digit_labels/train_digits;
sumG = 0;
test_labelG = 0;

% Create confusion matrix
confusion = zeros(10,10);

for i = 1:test_labels_size
    img = test_imgs(:,:,i); % Grab image
    % Set to either 0 or 1. %fore/background the image
    img(img >= 0.5) = 1; 
    img(img < 0.5) = 0;
    %vec = img(:); % Get as a vector
    test_labelG = test_labels(i);
    
    %test_labelG = test_labels(i); %grab what the label of this training digit is
    test_digit_labels(test_labelG+1) = test_digit_labels(test_labelG+1) + 1; %Add to count of the total of this digit in the training data
    
    logPtotal = log(Pc);
    

    for j = 1:size(test_imgs,1)-1 %The groups are overlapping. The last pixel location of each row does not have a pixel group due to the 2X2 size

        for k= 1:size(test_imgs,2)-1
            Gij = [img(j,k), img(j+1,k), img(j,k+1), img(j+1,k+1)];
            %trainG(j,k,i,:) = Gij; %make this faster? takes 28 seconds
            sumG = Gij(1) + Gij(2)*2 + Gij(3)*4 + Gij(4)*8;
            %imgG(j,k,i) = sum;

            Pij(:) = lsmooth(j,k,sumG+1,:); %now add this to the count 
            logPtotal = logPtotal + log(Pij);
            %Add to count of the feature at that location for that digit 
            %countG(j,k,sumG+1,labelG+1) = countG(j,k, sumG+1,labelG+1)+ 1;
        end
    end
    guess = find(logPtotal==max(logPtotal));
    confusion(guess, test_labels(i)+1) = confusion(guess, test_labels(i)+1) +1;

end

for  i = 1:10 
    confusion(:,i) = 100*confusion(:,i)/test_digit_labels(i);
end    
heatmap(confusion);


