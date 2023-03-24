% f(x) = ||x||2^2 - ||x||1
% f(x) = g(x) - h(x)
% g(x) = ||x||2^2 --> l2 norm
% h(x) = ||x||1 --> l1 norm

% starting point:
x0 = rand(2,1);

% using DCA
for k = 1:10
    % dh(x) = {1 for xi > 0, -1 for xi < 0 and -1 or 1 for x = 0}
    % majorization function: g(x) - (dh(x)^T.x) = x1^2 + x2^2 - (dh(x)^T.x) 
    % Value of dh(x) changes with the sign of the elements of x
    xk = norm(xk, 2) - (transpose(dh_xk)*x0);
    yk = norm(xk, 2) - norm(xk, 1);
    
end 