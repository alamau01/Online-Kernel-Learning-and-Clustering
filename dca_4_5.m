function dca_4_5(X,Y,alpha_b,v,mu_2,K,beta)

k = 1;
fbnorm_def = 0;
alphab_Kb = 0;
fbnorm = 0;
fbnorm_1 = 0;
fbnorm_2 = 0;
fbnorm_3 = 0;
while k <= 2000
    
    K_b = zeros(60,60,6,2000);
    if k > 1000
        K_b(1:15,1:15,1,k) = beta^(k-1000);
        K_b(1:15,1:15,2,k) = 1 - beta^(k-1000);
        K_b(31:45,31:45,2,k) = 1;
        K_b(16:30,16:30,3,k) = 1;
        K_b(46:60,46:60,4,k) = 1;
        K_b(:,:,5,k) = 1;
        K_b(:,:,6,k) = eye(60);
    else
        K_b(1:15,1:15,1,k) = 1 - beta^(k-1);
        K_b(1:15,1:15,2,k) = beta^(k-1);
        K_b(31:45,31:45,2,k) = 1;
        K_b(16:30,16:30,3,k) = 1;
        K_b(46:60,46:60,4,k) = 1;
        K_b(:,:,5,k) = 1;
        K_b(:,:,6,k) = eye(60);
    end
    
    K_bnew = zeros(60,60,6,2000);
    for b = 1:size(K_b,3)
        K_btemp(:,:) = K_b(:,:,b,k);
        K_btemp = K_btemp/(eps+trace(K_btemp));
        K_b(:,:,b,k) = K_btemp;
        K_bnew(:,:,b,k) = K_b(:,:,b,k);
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
        sum_Kb = zeros(60,60,2000);
        for b = 1:size(K_b,3)
            sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_b(b)*K_bnew(:,:,b,k);
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
        
        fbnorm_def = beta*fbnorm_def + 2*Y*transpose(Y)*X - 2*sum_Kb(:,:,k)*Y;
        
        M = zeros(6,6,2000); m = zeros(6,2000);
        for b = 1:size(K_b,3)
            for b1 = 1:size(K_b,3)
                M(b,b1,k) = trace(K_bnew(:,:,b,k)*K_bnew(:,:,b1,k));
                m(b,k) = trace(K_bnew(:,:,b,k)*Y*transpose(X));
            end
        end
            
        alphab_Kb = beta*alphab_Kb + 2*M(:,:,k)*alpha_b - 2*m(:,k);
            
        % optimizing problem using projected subgradient descent
        while 1
            X_prev = X;
            alpha_bprev = alpha_b;
            %fbnorm_def = 2*sum(beta^(k-1-(1:k))*Y*transpose(Y)*X) - 2*sum(beta^(k-1-t)*sum_Kb(:,:,k)*Y);
            delta_fx = fbnorm_def + v*sign(X_prev) - v*dh_X;
            X = max(X_prev - 0.0001*delta_fx,0);
            
            sum_Kb = zeros(60,60,2000);
            for b = 1:size(K_b,3)
                sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_bprev(b)*K_bnew(:,:,b,k);
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
            
            delta_falpha_b = alphab_Kb + mu_2*alpha_q_1 - mu_2*alpha_q;
            alpha_b = Simplex_Projection(alpha_bprev-0.0001*delta_falpha_b);
            if norm(X_prev-X,1) < 0.1 && norm(alpha_bprev-alpha_b,1) < 0.1
                break;
            end
        end
        if norm(X_prev2-X_prev,1) < 0.1 && norm(alpha_bprev2-alpha_bprev,1) < 0.1
            break;
        end
    end
    
    sum_Kb = zeros(60,60,2000);
    for b = 1:size(K_b,3)
        sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_bprev(b)*K_bnew(:,:,b,k);
    end
        
    while 1
        dh_Y = zeros(60,4);
        for i = size(Y,1)
            for j = size(Y,2)
                dh_Y(i,j) = Y(i,j)./sqrt(sum(Y(i,:).^2));
            end
        end
        
        fbnorm_1 = beta*fbnorm_1 + trace(sum_Kb(:,:,k)*transpose(sum_Kb(:,:,k)));
        fbnorm_2 = beta*fbnorm_2 + 2*transpose(sum_Kb(:,:,k));
        fbnorm_3 = beta*fbnorm_3 + 1;
        Y_prev = Y;
        
        cvx_begin quiet
            variable Y(60,4)
            for i = size(Y,1)
                Y_norm = norm(Y(i,:), 1);
            end
            %fbnorm = 0;
            %for t = 1:k
            %fbnorm = fbnorm + beta^(k-t)*square_pos(norm(sum_Kb(:,:,t) - X*transpose(Y), 'fro'));
            %end
            fbnorm = fbnorm + fbnorm_1 - trace(fbnorm_2*X*transpose(Y)) + fbnorm_3*square_pos(norm(X*transpose(Y), 'fro'));%+ fbnorm_3*trace(X*transpose(Y)*Y*transpose(X));
            minimize(fbnorm + v*sum(Y_norm) - v*sum(vec(transpose(dh_Y)).*vec(Y)))
            subject to
            vec(Y)>=0;
        cvx_end
        k
        alpha_b
        if norm(Y_prev-Y,1) < 0.01
            break;
        end
    end
    
    k = k+1;
    
end
end