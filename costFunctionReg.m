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


[r, c] = size(theta);

for i = 1:m
 J = J + (1/m)* ( -y(i,1)*log(1/(1 + exp(-X(i,:)*theta))) - (1 - y(i,1))*log(1- (1/(1 + exp(-X(i,:)*theta)))));
end

for t = 2:r
 J = J + lambda*(1/(2*m))*(theta(t,1)^2);
end
for j = 1:m
    grad(1,1) = grad(1,1) + ((1/m)* ((1/(1 + exp(-X(j,:)*theta))) -y(j,1))*X(j,1));
end
    
for i = 2:r
    for j = 1:m
    grad(i,1) = grad(i,1) + ((1/m)* ((1/(1 + exp(-X(j,:)*theta))) -y(j,1))*X(j,i)) ;
    end
    grad(i,1) = grad(i,1) + (lambda / m) * theta(i, 1)
end



% =============================================================

end
