% DCA_Assignment_4
clear all
clc
K_b = zeros(60,60,6);
K_b(1:15,1:15,2) = 1;
K_b(31:45,31:45,2) = 1;
K_b(16:30,16:30,3) = 1;
K_b(46:60,46:60,4) = 1;
K_b(:,:,5) = 1;
K_b(:,:,6) = eye(60);
X = rand(60,4); Y = rand(60,4); alpha_b = (1/6)*ones(6,1); v = 3;
X_0 = X;

for b = 1:size(K_b,3)
    K_btemp(:,:) = K_b(:,:,b);
    K_btemp = K_btemp/(eps+trace(K_btemp));
    K_b(:,:,b) = K_btemp;
    K_bnew(:,:,b) = K_b(:,:,b);
end

for k = 1:10
    % Calculating the subradient for norm 2
    Y_t = [];
    for i = 1:size(X,1)
        X_norm2 = norm(X(i,:),2);
        sg_X = X(i,:)/(eps+X_norm2);
        Y_t(4*(i-1)+1:4*i,1) = sg_X;
    end
    
    for i = 1:size(Y,1)
        for j = 1:size(Y,2)
            dh_Y(i,j) = Y(i,j)/sqrt(sum(Y(i,:).^2));
        end
    end
    
    % Calculating the subgradients for the eigenvalues
    sum_Kb = zeros(60,60);
    for b = 1:size(K_b,3)
        sum_Kb = sum_Kb + alpha_b(b)*K_bnew(:,:,b);
    end
     
    [eig_vec,eig_val] = eig(sum_Kb)  ; 
    [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
    V = eig_vec(:,pos_sort(1:4));
    for b = 1:size(K_b,3)
        K_btemp(1:60,1:60) = K_bnew(:,:,b);
        t = transpose(V)*K_btemp*V;
        Y_t(60*4+i,1) = sum(diag(t));
    end
    
    if mod(k,2) == 0
        cvx_begin quiet
        cvx_precision low
            variable X(60,4)
            variable alpha_b(6)

            l1 = 0;
            for i = 1:size(X,1)
                l1 = l1 + norm(X(i,:),1);
            end

            for i = 1:size(X,1)
                for j = 1:4
                    X(i*4+j,1) = X(i,j);
                end
            end
            X(60*4+1:60*4+6,1) = alpha_b;

            minimize(norm(sum_Kb-X*transpose(X_0),'fro')+l1+lambda_sum_largest(sum_Kb,4) - transpose(X)*Y_t)
            subject to
            sum(alpha_b) == 1;
            alpha_b>=0;
            vec(X)>=0;
        cvx_end
    else
        cvx_begin quiet
            variable Y(60,4)
            for i = size(Y,1)
                Y_norm = norm(Y(i,:), 1);
            end
            minimize(square_pos(norm(sum_Kb-X_0*transpose(Y), 'fro')) + v*sum(Y_norm) - v*sum(vec(dh_Y).*vec(transpose(Y))))
            subject to
            vec(Y)>=0;
        cvx_end
    end
end




 

