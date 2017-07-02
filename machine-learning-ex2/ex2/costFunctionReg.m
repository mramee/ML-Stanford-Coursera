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


h=sigmoid(X*theta)
n=length(theta)

J1 = 1/m * sum(-y .* log( h ) - (1 - y) .* log(1 - h) )
reg=0
theta
for j=2:n
    reg = reg + theta(j)^2
end
J = J1 + (lambda/(2*m)) * reg


grad0 = zeros(size(theta));
%grad0 = (1/m) * sum(h - y).*X(,1)
%for j=1:n
   for i=1:m
%        grad0(j) = grad0(j) + (h(i)-y(i))*X(i,1)
        grad(1) = grad(1) + (h(i)-y(i))*X(i,1)

    end
grad(1) = (1/m)*grad(1)

for j=2:n
    for i=1:m
        grad(j) = grad(j) + 1/m * (h(i) - y(i))*X(i,j)
    end
end

for j=2:n
    grad(j) = grad(j) + lambda/m * (theta(j))
end
%grad =  + grad1n + grad1nreg

% =============================================================

end
