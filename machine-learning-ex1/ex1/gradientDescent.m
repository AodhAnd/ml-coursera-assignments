function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%   GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp = zeros(2,1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    err = (theta' * X') - y'; 	% (h(x) - y)
    delta = (1/m * err * X)';	% d/d(theta) [J(theta)]
    theta = theta - alpha * delta;	% calculating theta
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    % fprintf("J(theta) is : %f	 \t theta is : %f %f\n", J_history(iter), theta(1), theta(2));

end

end
