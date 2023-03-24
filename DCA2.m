a = 2; v = 3; m = 32; n = 12; A = ones(m,n);

cvx_begin
    variable x(m)
    minimize(norm(a - (transpose(A)*x), 2) + v*norm(x))
cvx_end

x
