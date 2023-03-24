% DCA_Assignment_4_3
clear all
clc
X = rand(60,4); Y = rand(60,4); alpha_b = 1/6*ones(6,1); v = 0.001; mu_2 = 50; K = 4; beta = 0.9999; T = 0;

k = 0;
while k <= 760
    
    K_b = zeros(60,60,6,760);
    for t = 1:k
        K_b(1:15,1:15,1,t) = 1 - beta^k;
        K_b(1:15,1:15,2,t) = beta^k;
        K_b(31:45,31:45,2,t) = 1;
        K_b(16:30,16:30,3,t) = 1;
        K_b(46:60,46:60,4,t) = 1;
        K_b(:,:,5,t) = 1;
        K_b(:,:,6,t) = eye(60);
    end
    
    K_bnew = zeros(60,60,6,760);
    for b = 1:size(K_b,3)
        K_btemp(:,:) = K_b(:,:,b,t);
        K_btemp = K_btemp/(eps+trace(K_btemp));
        K_b(:,:,b,t) = K_btemp;
        K_bnew(:,:,b,t) = K_b(:,:,b,t);
    end
    
    while 1
        X_prev2 = X;
        alpha_bprev2 = alpha_b;
        % subgradient calculations in terms of x and y
        dh_X = [];
        for i = size(X,1)
            for j = size(X,2)
                dh_X(i,j) = X(i,j)./sqrt(sum(X(i,:).^2));
            end
        end
        
        % calculating the subgradients for the eigenvalues
        sum_Kb = zeros(60,60,760);
        for b = 1:size(K_b,3)
            sum_Kb(:,:,t) = sum_Kb(:,:,t) + alpha_b(b)*K_bnew(:,:,b,t);
        end
        
        [eig_vec, eig_val] = eig(sum_Kb(:,:,k));
        [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
        V = eig_vec(:,pos_sort(1:K));
        alpha_q = zeros(6,1);
        for b = 1:size(K_b,3)
            K_btemp(1:60,1:60) = K_bnew(:,:,b,k);
            t = transpose(V)*K_btemp*V;
            alpha_q(b,1) = sum(diag(t));
        end
    
        % optimizing problem using projected subgradient descent
        while 1
            X_prev = X;
            alpha_bprev = alpha_b;
            fbnorm_def = 0;
            for t = 1:k
                fbnorm_def = 2*sum(beta^(1600-t)*Y*transpose(Y)*X - sum_Kb*Y);
            end
            delta_fx = fbnorm_def + v*sign(X_prev) - v*dh_X;
            X = max(X_prev - 0.001*delta_fx,0);
            
            sum_Kb = zeros(60,60,760);
            for b = 1:size(K_b,3)
                sum_Kb(:,:,t) = sum_Kb(:,:,t) + alpha_bprev(b)*K_bnew(:,:,b,t);
            end
    
            [eig_vec, eig_val] = eig(sum_Kb(:,:,k));
            [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
            V = eig_vec(:,pos_sort(1:K-1));
            alpha_q_1 = [];
            for b = 1:size(K_b,3)
                K_btemp(1:60,1:60) = K_bnew(:,:,b,k);
                t = transpose(V)*K_btemp*V;
                alpha_q_1(b,1) = sum(diag(t));
            end
            
            M = zeros(6,6,760); m = zeros(6,760);
            for b = 1:size(K_b,3)
                for b1 = 1:size(K_b,3)
                    for t = 1:k
                        M(b,b1,t) = trace(K_bnew(:,:,b,t)*K_bnew(:,:,b1,t));
                        m(b,t) = trace(K_bnew(:,:,b,t)*Y*transpose(X_prev));
                    end
                end
            end
            
            alphab_Kb = 0;
            for t = 1:k
                alphab_Kb = 2*sum(beta^(1600-t)*M*alpha_bprev - 2*sum(beta^(1600-t)*m(:)));
            end
            delta_falpha_b = alphab_Kb + mu_2*alpha_q_1 - mu_2*alpha_q;
            alpha_b = Simplex_Projection(alpha_bprev-0.001*delta_falpha_b);
            if norm(X_prev-X,1) < 0.01 && norm(alpha_bprev-alpha_b,1) < 0.01
                break;
            end
        end
        if norm(X_prev2-X_prev,1) < 0.01 && norm(alpha_bprev2-alpha_bprev,1) < 0.01
            break;
        end
    end
    
    sum_Kb = zeros(60,60,760);
    for b = 1:size(K_b,3)
        sum_Kb(:,:,t) = sum_Kb(:,:,t) + alpha_b(b)*K_bnew(:,:,b,t);
    end
        
    while 1
        dh_Y = [];
        for i = size(Y,1)
            for j = size(Y,2)
                dh_Y(i,j) = Y(i,j)./sqrt(sum(Y(i,:).^2));
            end
        end
        
        Y_prev = Y;
        cvx_begin quiet
            variable Y(60,4)
            for i = size(Y,1)
                Y_norm = norm(Y(i,:), 1);
            end
            
            fbnorm = 0;
            for t = 1:k
                fbnorm = sum(beta^(1600-t)*square_pos(norm(sum_Kb-X*transpose(Y), 'fro')));
            end
            minimize(fbnorm + v*sum(Y_norm) - v*sum(vec(transpose(dh_Y)).*vec(Y)))
            subject to
            vec(Y)>=0;
        cvx_end
        alpha_b
        if norm(Y_prev-Y,1) < 0.01
            break;
        end
    end
    
    k = k+1;
    k
    
end