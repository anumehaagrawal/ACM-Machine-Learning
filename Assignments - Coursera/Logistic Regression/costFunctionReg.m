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
gradi=zeros(size(theta));
cost=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================
hypothesis=sigmoid(X*theta);
cost=(-1/m)*sum((y.*log(hypothesis))+(1-y).*(log(1-hypothesis)));
regcost=(lambda/(2*m))*sum(theta(2:end).^2);
J=cost+regcost;

for i=1: m
  gradi=gradi + (hypothesis(i)-y(i))*X(i,:)';
end
  reggrad=(lambda/m)*([0;theta(2:end)]);
  grad=(1/m)*gradi+reggrad;

end
