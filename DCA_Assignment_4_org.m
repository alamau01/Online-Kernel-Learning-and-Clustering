%DCA_Assignment_4_org
clear all
clc
X = rand(60,4); Y = rand(60,4); alpha_b = 1/6*ones(6,1); v = 0.001; mu_2 = 100; K = 4;


k = 1;
while k <= 1000
    
    K_b = zeros(60,60,6);
    K_b(1:15,1:15,2) = 1;
    K_b(31:45,31:45,2) = 1;
    K_b(16:30,16:30,3) = 1;
    K_b(46:60,46:60,4) = 1;
    K_b(:,:,5) = 1;
    K_b(:,:,6) = eye(60);
    
    K_bnew = zeros(60,60,6);
    for b = 1:size(K_b,3)
        K_btemp(:,:) = K_b(:,:,b);
        K_btemp = K_btemp/(eps+trace(K_btemp));
        K_b(:,:,b) = K_btemp;
        K_bnew(:,:,b) = K_b(:,:,b);
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
        sum_Kb = zeros(60,60);
        for b = 1:size(K_b,3)
            sum_Kb(:,:) = sum_Kb(:,:) + alpha_b(b)*K_bnew(:,:,b);
        end
        
        [eig_vec, eig_val] = eig(sum_Kb(:,:));
        [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
        V = eig_vec(:,pos_sort(1:K));
        alpha_q = zeros(6,1);
        for b = 1:size(K_b,3)
            K_btemp(1:60,1:60) = K_bnew(:,:,b);
            x = transpose(V)*K_btemp(:,:)*V;
            alpha_q(b,1) = sum(diag(x));
        end
    
        % optimizing problem using projected subgradient descent
        while 1
            X_prev = X;
            alpha_bprev = alpha_b;
            fbnorm_def = 2*Y*transpose(Y)*X - 2*sum_Kb(:,:)*Y; 
            delta_fx = fbnorm_def + v*sign(X_prev) - v*dh_X;
            X = max(X_prev - 0.001*delta_fx,0);
            
            sum_Kb = zeros(60,60);
            for b = 1:size(K_b,3)
                sum_Kb(:,:) = sum_Kb(:,:) + alpha_bprev(b)*K_bnew(:,:,b);
            end
    
            [eig_vec, eig_val] = eig(sum_Kb(:,:));
            [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
            V = eig_vec(:,pos_sort(1:K-1));
            alpha_q_1 = zeros(6,1);
            for b = 1:size(K_b,3)
                K_btemp(1:60,1:60) = K_bnew(:,:,b,k);
                x = transpose(V)*K_btemp(:,:)*V;
                alpha_q_1(b,1) = sum(diag(x));
            end
            
            M = zeros(6,6); m = zeros(6);
            for b = 1:size(K_b,3)
                for b1 = 1:size(K_b,3)
                    M(b,b1) = trace(K_bnew(:,:,b,k)*K_bnew(:,:,b1));
                    m(b) = trace(K_bnew(:,:,b)*Y*transpose(X_prev));
                end
            end
            
            alphab_Kb = 2*M(:,:)*alpha_bprev - 2*m(:);
            
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
    
    sum_Kb = zeros(60,60);
    for b = 1:size(K_b,3)
        sum_Kb(:,:) = sum_Kb(:,:) + alpha_bprev(b)*K_bnew(:,:,b);
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
            fbnorm = square_pos(norm(sum_Kb(:,:) - X*transpose(Y), 'fro'));
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
    
    [val, idx] = max(X,[],2);
    a11 = length(find(idx(1:15)==1));
    a12 = length(find(idx(1:15)==2));
    a13 = length(find(idx(1:15)==3));
    a14 = length(find(idx(1:15)==4));
    
    a21 = length(find(idx(range_overall+1:2*15)==1));
    a22 = length(find(idx(range_overall+1:2*15)==2));
    a23 = length(find(idx(range_overall+1:2*15)==3));
    a24 = length(find(idx(range_overall+1:2*15)==4));
    
    a31 = length(find(idx((2*range_overall)+1:3*15)==1));
    a32 = length(find(idx((2*range_overall)+1:3*15)==2));
    a33 = length(find(idx((2*range_overall)+1:3*15)==3));
    a34 = length(find(idx((2*range_overall)+1:3*15)==4));
    
    a41 = length(find(idx((2*range_overall)+1:4*15)==1));
    a42 = length(find(idx((2*range_overall)+1:4*15)==2));
    a43 = length(find(idx((2*range_overall)+1:4*15)==3));
    a44 = length(find(idx((2*range_overall)+1:4*15)==4));
    
    a_perf = [a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44];
    
    amat = vec2mat(a_perf,4);
    P_m = perms([1:4]);
    
    Pc = zeros(1,8);
    for ind_perm = 1:size(P_m,1)
        Pc(ind_perm) = trace(amat(:,P_m(ind_perm,:)));
    end
    
    a_perf
    amat
    Pc
    
    acc_max(k) = max(Pc)
end