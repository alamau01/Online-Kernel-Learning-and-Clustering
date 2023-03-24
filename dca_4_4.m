function dca_4_4(X,Y,alpha_b,v,mu_2,K,beta)

k = 1;
while k <= 500
    
    K_b = zeros(60,60,6,1000);
    for t = 1:k
        if k > 35
            K_b(1:15,1:15,1,t) = beta^(k-1);
            K_b(1:15,1:15,2,t) = 1 - beta^((k-1)-761);
            K_b(31:45,31:45,2,t) = 1;
            K_b(16:30,16:30,3,t) = 1;
            K_b(46:60,46:60,4,t) = 1;
            K_b(:,:,5,t) = 1;
            K_b(:,:,6,t) = eye(60);
        else
            K_b(1:15,1:15,1,t) = 1 - beta^(k-1);
            K_b(1:15,1:15,2,t) = beta^(k-1);
            K_b(31:45,31:45,2,t) = 1;
            K_b(16:30,16:30,3,t) = 1;
            K_b(46:60,46:60,4,t) = 1;
            K_b(:,:,5,t) = 1;
            K_b(:,:,6,t) = eye(60);
        end
    end
    
    K_bnew = zeros(60,60,6,1000);
    for b = 1:size(K_b,3)
        for t = 1:k
            K_btemp(:,:) = K_b(:,:,b,t);
            K_btemp = K_btemp/(eps+trace(K_btemp));
            K_b(:,:,b,t) = K_btemp;
            K_bnew(:,:,b,t) = K_b(:,:,b,t);
        end
    end

    while 1
        X_prev2 = X;
        alpha_bprev2 = alpha_b;
        % subgradient calculations in terms of x and y
        dh_X = zeros(60,4);
        for i = size(X,1)
            for j = size(X,2)
                dh_X(i,j) = X(i,j)./sqrt(sum(X(i,:).^2));
            end
        end
        
        % calculating the subgradients for the eigenvalues
        sum_Kb = zeros(60,60,1000);
        for b = 1:size(K_b,3)
            for t = 1:k
                sum_Kb(:,:,t) = sum_Kb(:,:,t) + alpha_b(b)*K_bnew(:,:,b,t);
            end
        end
        
        [eig_vec, eig_val] = eig(sum_Kb(:,:,k));
        [~,pos_sort] = sort(diag(eig_val),'descend'); 
    
        V = eig_vec(:,pos_sort(1:K));
        alpha_q = zeros(6,1);
        for b = 1:size(K_b,3)
            K_btemp(1:60,1:60) = K_bnew(:,:,b,k);
            x = transpose(V)*K_btemp(:,:)*V;
            alpha_q(b,1) = sum(diag(x));
        end
    
        % optimizing problem using projected subgradient descent
        while 1
            X_prev = X;
            alpha_bprev = alpha_b;
            fbnorm_def = 0;
            for t = 1:k
                fbnorm_def = fbnorm_def + 2*beta^(k-t)*Y*transpose(Y)*X - 2*beta^(k-t)*sum_Kb(:,:,t)*Y; 
            end
            delta_fx = fbnorm_def + v*sign(X_prev) - v*dh_X;
            X = max(X_prev - 0.0001*delta_fx,0);
            
            sum_Kb = zeros(60,60,1000);
            for b = 1:size(K_b,3)
                for t = 1:k
                    sum_Kb(:,:,t) = sum_Kb(:,:,t) + alpha_bprev(b)*K_bnew(:,:,b,t);
                end
            end
    
            [eig_vec, eig_val] = eig(sum_Kb(:,:,k));
            [~,pos_sort] = sort(diag(eig_val),'descend'); 
    
            V = eig_vec(:,pos_sort(1:K-1));
            alpha_q_1 = zeros(6,1);
            for b = 1:size(K_b,3)
                K_btemp(1:60,1:60) = K_bnew(:,:,b,k);
                x = transpose(V)*K_btemp(:,:)*V;
                alpha_q_1(b,1) = sum(diag(x));
            end
            
            M = zeros(6,6,1000); m = zeros(6,1000);
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
                alphab_Kb = alphab_Kb + 2*beta^(k-t)*M(:,:,t)*alpha_bprev - 2*beta^(k-t)*m(:,t);
            end
            delta_falpha_b = alphab_Kb + mu_2*alpha_q_1 - mu_2*alpha_q;
            alpha_b = Simplex_Projection(alpha_bprev-0.0001*delta_falpha_b);
            if norm(X_prev-X,1) < 0.01 && norm(alpha_bprev-alpha_b,1) < 0.01
                break;
            end
        end
        if norm(X_prev2-X_prev,1) < 0.01 && norm(alpha_bprev2-alpha_bprev,1) < 0.01
            break;
        end
    end
    
    sum_Kb = zeros(60,60,1000);
        for t = 1:k
            sum_Kb(:,:,t) = sum_Kb(:,:,t) + alpha_b(b)*K_bnew(:,:,b,t);
        end
        
    while 1
        dh_Y = zeros(60,4);
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
                fbnorm = fbnorm + beta^(k-t)*square_pos(norm(sum_Kb(:,:,t) - X*transpose(Y), 'fro'));
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
end