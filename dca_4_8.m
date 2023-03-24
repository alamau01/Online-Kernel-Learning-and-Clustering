function dca_4_8(X,Y,alpha_b,v,mu_2,K,beta)

k = 1;
fbnorm_def = 0;
alphab_Kb = 0;
fbnorm_def1 = 0;
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
                dh_X(i,j) = -3*X(i,j)./sqrt(sum(X(i,:).^2));
                dh_Y(i,j) = -3*Y(i,j)./sqrt(sum(Y(i,:).^2));
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
        end
        
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
            X = max(X_prev - 0.001*delta_fx,0);
            Y = max(Y_prev - 0.001*delta_fy,0);
            
            sum_Kb = zeros(60,60,6000);
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
            alpha_b = Simplex_Projection(alpha_bprev-0.001*delta_falpha_b);
            if norm(X_prev-X,1) < 0.6 && norm(alpha_bprev-alpha_b,1) < 0.6 || norm(Y_prev-Y,1) < 0.6
                break;
            end
        end
        if norm(X_prev2-X_prev,1) < 0.6 && norm(alpha_bprev2-alpha_bprev,1) < 0.6 || norm(Y_prev2-Y_prev,1) < 0.6
            break;
        end
    end

    gnd = [ones(1,15) 2*ones(1,15) 3*ones(1,15) 4*ones(1,15)];
    [~,idx_X] = max((X+Y)/2,[],2);
    result = ClusteringMeasure(gnd,idx_X);
    
    %[val, idx] = max(X,[],2);
    %a11 = length(find(idx(1:15)==1));
    %a12 = length(find(idx(1:15)==2));
    %a13 = length(find(idx(1:15)==3));
    %a14 = length(find(idx(1:15)==4));
    
    %a21 = length(find(idx(15+1:2*15)==1));
    %a22 = length(find(idx(15+1:2*15)==2));
    %a23 = length(find(idx(15+1:2*15)==3));
    %a24 = length(find(idx(15+1:2*15)==4));
    
    %a31 = length(find(idx((2*15)+1:3*15)==1));
    %a32 = length(find(idx((2*15)+1:3*15)==2));
    %a33 = length(find(idx((2*15)+1:3*15)==3));
    %a34 = length(find(idx((2*15)+1:3*15)==4));
    
    %a41 = length(find(idx((3*15)+1:4*15)==1));
    %a42 = length(find(idx((3*15)+1:4*15)==2));
    %a43 = length(find(idx((3*15)+1:4*15)==3));
    %a44 = length(find(idx((3*15)+1:4*15)==4));
    
    %a_perf = [a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44];
    
    %amat = vec2mat(a_perf,4);
    %P_m = perms([1:4]);
    
    %Pc = zeros(1,8);
    %for ind_perm = 1:size(P_m,1)
        %Pc(ind_perm) = trace(amat(:,P_m(ind_perm,:)));
    %end
    
    %acc_max(k) = max(Pc)
    k = k + 1
    alpha_b1(k) = alpha_b(1);
    alpha_b2(k) = alpha_b(2);
    alpha_b3(k) = alpha_b(3);
    alpha_b4(k) = alpha_b(4);
    alpha_b5(k) = alpha_b(5);
    alpha_b6(k) = alpha_b(6); 
end