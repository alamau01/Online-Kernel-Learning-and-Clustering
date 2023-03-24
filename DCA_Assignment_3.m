clear all
clc
% DCA_Assignment_3
V = 0.01; % constant variable
P = 8; Q = 2; % number of rows and columns
A = eye(2, 2); B = ones(4, 4); K = kron(A, B); X_k = rand(P, Q); Y_k = rand(P, Q);
dh_X = []; dh_Y = [];

for k = 1:10
    for i = 1:size(X_k,1)
        for j = 1:size(Y_k,2)
            dh_X(i,j) = X_k(i,j)/sqrt(sum(X_k(i,:).^2));
            dh_Y(i,j) = Y_k(i,j)/sqrt(sum(Y_k(i,:).^2));
        end
    end
    
    if mod(k,2) == 0
        cvx_begin quiet
            variable X(P,Q)
            for i = 1:size(X_k, 1)
                X_norm = norm(X(i,:), 1);
            end
            minimize(square_pos(norm(K-X*transpose(Y_k), 'fro')) + V*sum(X_norm) - V*sum(vec(dh_X).*vec(transpose(X))))
            subject to
            vec(X)>=0;
        cvx_end
        X_k = X;
    else
        cvx_begin quiet
            variable Y(P,Q)
            for i = 1:size(Y_k, 1)
                Y_norm = norm(Y(i,:), 1);
            end
            minimize(square_pos(norm(K-X_k*transpose(Y), 'fro')) + V*sum(Y_norm) - V*sum(vec(dh_Y).*vec(transpose(Y))))
            subject to
            vec(Y)>=0;
        cvx_end
        Y_k = Y;
    end
end