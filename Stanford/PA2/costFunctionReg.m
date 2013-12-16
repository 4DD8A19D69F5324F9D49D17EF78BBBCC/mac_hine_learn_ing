function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hx = sigmoid(X * theta);
d = theta;
d(1) = 0;


J = -1.0 / m * sum(y .* log(hx) + (1.0.-y ).*log(1.0.-hx));
J += lambda / 2.0 / m * sum(d.^2);



grad = 1.0/ m * ( hx .- y)' * X;
grad = grad + lambda / m * d';


%grad(1) -= lambda/m * theta;



% =============================================================

end
