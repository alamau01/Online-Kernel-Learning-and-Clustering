clear all
clc
% DCA_Assignment_2
a = 2; v = 1; % constant real variables
m = 4; n = 4; % number of rows and columns
A = ones(n,1); rng(1); X = rand(n,1);

for k = 1:3
    dh_X = v*(X/norm(X, 2));
    % Y = norm((a - transpose(A)*X), 2).^2 + (v*norm(X, 1)) - (transpose(dh_X*X));
    cvx_begin quiet
        variable X(n, 1)
        minimize(square_pos(norm(a - transpose(A)*X, 2)) + (v*norm(X,1)) - (transpose(dh_X)*X))
        subject to
        vec(X)>=0;
    cvx_end
    
    X
end     



