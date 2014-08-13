function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
J= 0;
grad = zeros(size(theta));
% You need to return the following variables correctly 
%J = 0;

%J = 0;

%hyp=X * theta


%J=(1/(2*m))*sum((hyp-y).^2)
%J=J+(lambda/(2*m))*(sum(theta.^2)-theta(1,1).^2)

%grad = zeros(size(theta));


%for i=1:size(theta,1)

%grad(i,1)=(1/m)*sum((hyp-y).*X(:,i))
%if i>1
%grad(i,1)=grad(i,1)+(lambda/m)*theta(i,1)
%endif

%end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

H_theta = X * theta;

J = 1/(2*m) * (H_theta - y)' * (H_theta - y) + ...
   (lambda/(2*m)) * norm(theta([2:end]))^2;

G = (lambda/m) .* theta;
G(1) = 0; % extra term for gradient

grad = ((1/m) .* X' * (H_theta - y)) + G;

% =========================================================================

grad = grad(:);

end

