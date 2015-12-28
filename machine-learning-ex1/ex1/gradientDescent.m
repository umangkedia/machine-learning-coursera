function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    hypothesis = ((X * theta) - y)';

    tempTheta = theta - alpha * (1 / m) * (hypothesis * X)'; 
    %temp1 = theta(1) - alpha * (1 / m) * hypothesis * X(:,1);
    %temp2 = theta(2) - alpha * (1 / m) * hypothesis * X(:, 2);

    %theta(1) = temp1;
    %theta(2) = temp2;
    theta = tempTheta;

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    J_history(iter)

end

end
