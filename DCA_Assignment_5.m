% DCA_Assignment_5
clear all
clc
K_b = zeros(60,60,6);
K_b(1:15,1:15,2) = 1;
K_b(31:45,31:45,2) = 1;
K_b(16:30,16:30,3) = 1;
K_b(46:60,46:60,4) = 1;
K_b(:,:,5) = 1;
K_b(:,:,6) = eye(60);
X = rand(60,4); Y = rand(60,4); alpha_b = 1/6*ones(6,1); mu_2 = 10; Q = 4; P = size(X,1);

K_bnew = [];
for b = 1:size(K_b,3)
    K_btemp(:,:) = K_b(:,:,b);
    K_btemp = K_btemp/(eps+trace(K_btemp));
    K_b(:,:,b) = K_btemp;
    K_bnew(:,:,b) = K_b(:,:,b);
end

while 1
    X_prev2 = X;
    alpha_bprev2 = alpha_b;
        
    % optimizing problem using projected subgradient descent
    cnt = 1;
    while 1
        %subgradient calculations in terms of x and y
        %dh_X = [];
        %for i = size(X,1)
            %for j = size(X,2)
                %dh_X(i,j) = X(i,j)./sqrt(sum(X(i,:).^2));
            %end
        %end
        
        % Calculating the subgradients for the eigenvalues
        sum_Kb = zeros(60,60);
        for b = 1:size(K_b,3)
            sum_Kb = sum_Kb + alpha_b(b)*K_bnew(:,:,b);
        end
    
        [eig_vec, eig_val] = eig(sum_Kb);
        [v_sort,pos_sort] = sort(diag(eig_val),'ascend'); 
    
        V = eig_vec(:,pos_sort(1:P-Q));
        alpha_q = [];
        for b = 1:size(K_b,3)
            K_btemp(1:60,1:60) = K_bnew(:,:,b);
            t = transpose(V)*K_btemp*V;
            alpha_q(b,1) = sum(diag(t));
        end
        
        X_prev = X;
        alpha_bprev = alpha_b;
        delta_fx = 2*Y*transpose(Y)*X - 2*sum_Kb*Y; %+ v*sign(X_prev) - v*dh_X;
        X = max(X_prev - (0.001/sqrt(cnt))*delta_fx,0);
        delta_falpha_b = 2*trace(K_bnew(:,:,b).*K_bnew(:,:,b)).*alpha_bprev - 2*trace(K_bnew(:,:,b)*Y*transpose(X_prev)) + mu_2*alpha_q;
        alpha_b = Simplex_Projection(alpha_bprev-(0.001/sqrt(cnt))*delta_falpha_b);
        cnt = cnt+1;
        norm(X_prev-X)
        norm(alpha_bprev-alpha_b)
        if norm(X_prev-X) < 0.001 && norm(alpha_bprev-alpha_b) < 0.001
            break;
        end
    end
    
sum_Kb = zeros(60,60);
for b = 1:size(K_b,3)
    sum_Kb = sum_Kb + alpha_b(b)*K_bnew(:,:,b);
end

%dh_Y = [];
%for i = size(Y,1)
    %for j = size(Y,2)
        %dh_Y(i,j) = Y(i,j)./sqrt(sum(Y(i,:).^2));
    %end
%end

Y_prev = Y;
cvx_begin quiet
    variable Y2(60,4)
    for i = size(Y,1)
        Y2_norm = norm(Y2(i,:), 1);
    end
    minimize(square_pos(norm(sum_Kb-X*transpose(Y2), 'fro'))) %+ v*sum(Y2_norm) - v*sum(vec(transpose(dh_Y)).*vec(Y)))
    subject to
    vec(Y2)>=0;
cvx_end
Y = Y2;
norm(Y_prev-Y)
norm(X_prev2-X_prev)
norm(alpha_bprev2-alpha_bprev)
alpha_b
X
if norm(Y_prev-Y) < 0.001 && norm(X_prev2-X_prev) < 0.001 && norm(alpha_bprev2-alpha_bprev) < 0.001
    break;
end
end