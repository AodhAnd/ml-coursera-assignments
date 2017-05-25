function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%-- PART 1 --%
% Extending X with the bias unit
X = [ones(m,1) X];

% Computing the hidden layer results and adding its bias unit
a2 = sigmoid(X * Theta1');
a2 = [ones(size(a2,1),1) a2];

% Computing the output (or activation) layer
a3 = sigmoid(a2 * Theta2');

% Creating binary (0,1) vectors back from the given y 
% (containing values from 1 to 10)
Y = zeros(m,num_labels);
for i=1:m
  Y(i,y(i,1)) = 1;
end

% Calculating cost for each output
for k=1:num_labels
  temp(k) = sum(-Y(:,k) .* log(a3(:,k)) - (ones(m,1) - Y(:,k)) .* log(ones(m,1) - a3(:,k)));
end

% Unregularized cost
J = 1/m * sum(temp);

regterm = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J += regterm;

clear a2,a3;
%-- PART 2 --%
Delta1 = zeros(input_layer_size+1,hidden_layer_size);
Delta2 = zeros(hidden_layer_size+1,num_labels);
% All vectors are kept as column vectors for ease of understanding
for i=1:m
  % Feedforward prop.
  a1 = X(i,:)';
  z2 = Theta1 * a1;
  a2 = [1 sigmoid(z2)']';
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  % Output layer's delta
  delta3 = a3 - Y(i,:)';
  
  % Hidden layer's delta
  delta2 = Theta2' * delta3;
  delta2 = delta2(2:end) .* sigmoidGradient(z2);

  % Hidden layer's accumulated gradient
  Delta2 = Delta2 + (delta3 * a2')';
  % Input layer's accumulated gradient
  Delta1 = Delta1 + (delta2 * a1')';
end

Theta1_grad = 1/m * Delta1';
Theta2_grad = 1/m * Delta2';

% disp('delta2');
% disp(delta2);
% disp('delta3');
% disp(delta3);
% disp('Delta1');
% disp(Delta1');
% disp('Delta2');
% disp(Delta2');

%-- PART 3 --%
Theta1_grad(:,2:end) += lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) += lambda/m * Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
