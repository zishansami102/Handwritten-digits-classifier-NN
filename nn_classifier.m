clear ; close all; clc

%% Setting the required parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 , 0 has been maped to 10
                         

% loading data
load('ex4data1.mat');
m = size(X, 1);

% Randomly selecting 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

% Initializing Neural Network Parameters ...
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unrolling parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('Program paused. Press enter to start training the neural network...\n');
pause;
options = optimset('MaxIter', 100);
lambda = 1;
costFunction = @(p) nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtaining Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to compute the accuracy of training set.\n');
pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


