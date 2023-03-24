function dca_4_9(X,Y,alpha_b,v,mu_2,K,beta)

k = 1;
fbnorm_def = 0;
alphab_Kb = 0;
fbnorm_def1 = 0;
alpha_q_k_1 = 0;
alpha_q_1_k_1_b = 0;
while k <= 6000
    
    K_b = zeros(60,60,6,6000);
    if k <= 2000
        K_b(1:15,1:15,1,k) = 1 - beta^(k);
        K_b(1:15,1:15,2,k) = beta^(k);
        K_b(31:45,31:45,2,k) = 1;
        K_b(16:30,16:30,3,k) = 1;
        K_b(46:60,46:60,4,k) = 1;
        K_b(:,:,5,k) = 1;
        K_b(:,:,6,k) = eye(60); 
    elseif k > 2000 && k <= 4000
        K_b(1:15,1:15,1,k) = beta^(k-2000);
        K_b(1:15,1:15,2,k) = 1 - beta^(k-2000);
        K_b(31:45,31:45,2,k) = 1;
        K_b(16:30,16:30,3,k) = 1;
        K_b(46:60,46:60,4,k) = 1;
        K_b(:,:,5,k) = 1;
        K_b(:,:,6,k) = eye(60);
    else
        K_b(1:15,1:15,1,k) = 1 - beta^(k-4000);
        K_b(1:15,1:15,2,k) = beta^(k-4000);
        K_b(31:45,31:45,2,k) = 1;
        K_b(16:30,16:30,3,k) = 1;
        K_b(46:60,46:60,4,k) = 1;
        K_b(:,:,5,k) = 1;
        K_b(:,:,6,k) = eye(60);
    end
    
    K_bnew = zeros(60,60,6,6000);
    for b = 1:size(K_b,3)
        K_btemp(:,:) = K_b(:,:,b,k);
        K_btemp = K_btemp/(eps+trace(K_btemp));
        K_b(:,:,b,k) = K_btemp;
        K_bnew(:,:,b,k) = K_b(:,:,b,k);
    end
    
    while 1
        X_prev2 = X;
        Y_prev2 = Y;
        alpha_bprev2 = alpha_b;
        % subgradient calculations in terms of x and y
        dh_X = zeros(60,4);
        dh_Y = zeros(60,4);
        for i = size(X,1)
            for j = size(Y,2)
                dh_X(i,j) = X(i,j)./sqrt(sum(X(i,:).^2));
                dh_Y(i,j) = Y(i,j)./sqrt(sum(Y(i,:).^2));
            end
        end
                
        % calculating the subgradients for the eigenvalues
        sum_Kb = zeros(60,60,6000);
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
            %alpha_q_k_1(b,1) = sum(beta^(k-t-1)*alpha_q(b,1));
            %alpha_q(b,1) = alpha_q(b,1) + beta*alpha_q_k_1(b,1);
        end
        
        V = eig_vec(:,pos_sort(1:K-1));
        alpha_q_1_b = zeros(6,1);
        for b = 1:size(K_b,3)
            K_btemp(1:60,1:60) = K_bnew(:,:,b,k);
            x = transpose(V)*K_btemp(:,:)*V;
            alpha_q_1_b(b,1) = sum(diag(x));
            %alpha_q_k_1(b,1) = sum(beta^(k-t-1)*alpha_q(b,1));
            %alpha_q(b,1) = alpha_q(b,1) + beta*alpha_q_k_1(b,1);
        end

        alpha_q_k_1 = alpha_q_k_1 + beta*alpha_q(b,1);
        alpha_q_1_k_1_b = alpha_q_1_k_1_b + beta*alpha_q_1_b(b,1);
        
        fbnorm_def = beta*fbnorm_def + 2*Y*transpose(Y)*X - 2*sum_Kb(:,:,k)*Y;
        fbnorm_def1 = beta*fbnorm_def1 + 2*X*transpose(X)*Y - 2*sum_Kb(:,:,k)*X;
        
        M = zeros(6,6,6000); m = zeros(6,6000);
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
            Y_prev = Y;
            alpha_bprev = alpha_b;
            delta_fx = fbnorm_def + v*sign(X_prev) - v*dh_X;
            delta_fy = fbnorm_def1 + v*sign(Y_prev) - v*dh_Y;
            X = max(X_prev - 0.0001*delta_fx,0);
            Y = max(Y_prev - 0.0001*delta_fy,0);
            
            sum_Kb = zeros(60,60,6000);
            for b = 1:size(K_b,3)
                sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_bprev(b)*K_bnew(:,:,b,k);
            end
    
            [eig_vec, eig_val] = eig(sum_Kb(:,:,k));
            [~,pos_sort] = sort(diag(eig_val),'descend'); 
            
            %alpha_q_1_k_1_b = alpha_q_1_k_1_b - beta*alpha_q_1_b(b,1);
            
            V = eig_vec(:,pos_sort(1:K-1));
            alpha_q_1 = zeros(6,1);
            alpha_q_1_k_1 = zeros(6,1);
            for b = 1:size(K_b,3)
                for t = 1:k
                    K_btemp(1:60,1:60) = K_bnew(:,:,b,k);
                    x = transpose(V)*K_btemp(:,:)*V;
                    alpha_q_1_b(b,1) = sum(diag(x));
                    alpha_q_1_k_1(b,1) = sum(beta^(k-t-1)*alpha_q_1(b,1));
                    alpha_q_1(b,1) = alpha_q_1(b,1) + beta*alpha_q_1_k_1(b,1);
                end
            end
            
            %alpha_q_1_k_1_b = alpha_q_1_k_1_b + beta*alpha_q_1_b(b,1);
            
            delta_falpha_b = alphab_Kb + mu_2*alpha_q_k_1 - mu_2*alpha_q_1_k_1_b;
            alpha_b = Simplex_Projection(alpha_bprev-0.0001*delta_falpha_b);
            if norm(X_prev-X,1) < 0.01 && norm(alpha_bprev-alpha_b,1) < 0.01 || norm(Y_prev-Y,1) < 0.01
                break;
            end
        end
        if norm(X_prev2-X_prev,1) < 0.01 && norm(alpha_bprev2-alpha_bprev,1) < 0.01 || norm(Y_prev2-Y_prev,1) < 0.01
            break;
        end
    end
    
    k
    %alpha_b
    X
    k = k+1;
    
end