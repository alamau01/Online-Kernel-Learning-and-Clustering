tic
X = 1/60*rand(60,4); Y = 1/60*rand(60,4); alpha_b = 1/6*ones(6,1);
v = 0.001;mu_2 = 100;K = 4;beta = 0.85;
k = 1;
fbnorm_def = 0;
alphab_Kb = 0;
fbnorm_def1 = 0;
K_bnew_beta = zeros(60,60,6);
K_b = zeros(60,60,6,300);
while k <= 300
    if k <= 100
        K_b(1:15,1:15,1,k) = 1 - beta^(k);
        K_b(1:15,1:15,2,k) = beta^(k);
        K_b(31:45,31:45,2,k) = 1;
        K_b(16:30,16:30,3,k) = 1;
        K_b(46:60,46:60,4,k) = 1;
        K_b(:,:,5,k) = 1;
        K_b(:,:,6,k) = eye(60); 
    elseif k <= 200
        K_b(1:15,1:15,1,k) = beta^(k-100);
        K_b(1:15,1:15,2,k) = 1 - beta^(k-100);
        K_b(31:45,31:45,2,k) = 1;
        K_b(16:30,16:30,3,k) = 1;
        K_b(46:60,46:60,4,k) = 1;
        K_b(:,:,5,k) = 1;
        K_b(:,:,6,k) = eye(60);
    else
        K_b(1:15,1:15,1,k) = 1 - beta^(k-200);
        K_b(1:15,1:15,2,k) = beta^(k-200);
        K_b(31:45,31:45,2,k) = 1;
        K_b(16:30,16:30,3,k) = 1;
        K_b(46:60,46:60,4,k) = 1;
        K_b(:,:,5,k) = 1;
        K_b(:,:,6,k) = eye(60);
    end
    
    K_bnew = zeros(60,60,6,300);
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
        
        %calculating the subgradients for the eigenvalues
        for b = 1:size(K_b,3)
            K_bnew_beta(:,:,b) = beta*K_bnew_beta(:,:,b) + K_bnew(:,:,b,k);
        end
        
        beta_sum_Kb = zeros(60,60);
        for b = 1:size(K_b,3)
            beta_sum_Kb(:,:) = beta_sum_Kb(:,:) + alpha_b(b)*K_bnew_beta(:,:,b);
        end
        
        [eig_vec, eig_val] = eig(beta_sum_Kb(:,:));
        [~,pos_sort] = sort(diag(eig_val),'descend'); 
    
        V = eig_vec(:,pos_sort(1:K));
        alpha_q = zeros(6,1);
        for b = 1:size(K_b,3)
            K_btemp(1:60,1:60) = K_bnew_beta(:,:,b);
            x = transpose(V)*K_btemp(:,:)*V;
            alpha_q(b,1) = sum(diag(x));
        end
        
        %K_bnew_beta(:,:,b) = sum(beta^(k-300))*K_bnew(:,:,b,300);
        
        fbnorm_def = beta*fbnorm_def + 2*Y*transpose(Y)*X - 2*beta_sum_Kb(:,:)*Y;
        fbnorm_def1 = beta*fbnorm_def1 + 2*X*transpose(X)*Y - 2*beta_sum_Kb(:,:)*X;
        
        M = zeros(6,6,300); m = zeros(6,300);
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
            
            beta_sum_Kb = zeros(60,60);
            for b = 1:size(K_b,3)
                beta_sum_Kb(:,:) = beta_sum_Kb(:,:) + alpha_b(b)*K_bnew_beta(:,:,b);
            end

            [eig_vec, eig_val] = eig(beta_sum_Kb(:,:));
            [~,pos_sort] = sort(diag(eig_val),'descend'); 
    
            V = eig_vec(:,pos_sort(1:K-1));
            alpha_q_1 = zeros(6,1);
            for b = 1:size(K_b,3)
                K_btemp(1:60,1:60) = K_bnew_beta(:,:,b);
                x = transpose(V)*K_btemp(:,:)*V;
                alpha_q_1(b,1) = sum(diag(x));
            end
            
            delta_falpha_b = alphab_Kb + mu_2*alpha_q_1 - mu_2*alpha_q;
            alpha_b = Simplex_Projection(alpha_bprev-0.0001*delta_falpha_b);
            if norm(X_prev-X,1) < 0.06 && norm(alpha_bprev-alpha_b,1) < 0.06 || norm(Y_prev-Y,1) < 0.06
                break;
            end
        end
        if norm(X_prev2-X_prev,1) < 0.06 && norm(alpha_bprev2-alpha_bprev,1) < 0.06 || norm(Y_prev2-Y_prev,1) < 0.06
            break;
        end
    end
    
    gnd = [ones(1,15) 2*ones(1,15) 3*ones(1,15) 4*ones(1,15)];
    [~,idx_X] = max((X+Y)/2,[],2);
    result = ClusteringMeasure(gnd,idx_X);
    %k
    %alpha_b
    
    %Method 1:
    %Matr = (X + Y)/2;
    %algo_acc = abs(Matr);
    
    %Method 2:
    %M_1 = 0.5*(X*Y'+Y*X');
    %[U,D] = eig(M_1);
    %[~,pos_sort1] = sort(diag(D),'descend');
    %Matr1 = U(:,pos_sort1(1:K));
    %algo_acc1 = abs(Matr1);
    
    %[~, idx] = max(algo_acc,[],2);
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
    
    %acc_max(k) = max(Pc);
    
    %[~, idx1] = max(algo_acc1,[],2);
    %a111 = length(find(idx1(1:15)==1));
    %a122 = length(find(idx1(1:15)==2));
    %a133 = length(find(idx1(1:15)==3));
    %a144 = length(find(idx1(1:15)==4));
    
    %a211 = length(find(idx1(15+1:2*15)==1));
    %a222 = length(find(idx1(15+1:2*15)==2));
    %a233 = length(find(idx1(15+1:2*15)==3));
    %a244 = length(find(idx1(15+1:2*15)==4));
    
    %a311 = length(find(idx1((2*15)+1:3*15)==1));
    %a322 = length(find(idx1((2*15)+1:3*15)==2));
    %a333 = length(find(idx1((2*15)+1:3*15)==3));
    %a344 = length(find(idx1((2*15)+1:3*15)==4));
    
    %a411 = length(find(idx1((3*15)+1:4*15)==1));
    %a422 = length(find(idx1((3*15)+1:4*15)==2));
    %a433 = length(find(idx1((3*15)+1:4*15)==3));
    %a444 = length(find(idx1((3*15)+1:4*15)==4));
    
    %a_perf1 = [a111,a122,a133,a144,a211,a222,a233,a244,a311,a322,a333,a344,a411,a422,a433,a444];
    
    %amat1 = vec2mat(a_perf1,4);
    %P_m1 = perms([1:4]);
    
    %Pc1 = zeros(1,8);
    %for ind_perm1 = 1:size(P_m1,1)
        %Pc1(ind_perm1) = trace(amat1(:,P_m1(ind_perm1,:)));
    %end
    
    %acc_max1(k) = max(Pc);
    %result
    
    k = k + 1
    alpha_b1(k) = alpha_b(1);
    alpha_b2(k) = alpha_b(2);
    alpha_b3(k) = alpha_b(3);
    alpha_b4(k) = alpha_b(4);
    alpha_b5(k) = alpha_b(5);
    alpha_b6(k) = alpha_b(6);
end

%figure;
%subplot(2,3,1);
%spy(K_b(:,:,1,200) > 10^-4,'w');
%title('K_1(X)');
%subplot(2,3,2);
%spy(K_b(:,:,2,200));
%title('K_2(X)');
%subplot(2,3,3);
%spy(K_b(:,:,3,200));
%title('K_3(X)');
%subplot(2,3,4);
%spy(K_b(:,:,4,200));
%title('K_4(X)');
%subplot(2,3,5);
%spy(K_b(:,:,5,200));
%title('K_5(X)');
%subplot(2,3,6);
%spy(K_b(:,:,6,200));
%title('K_6(X)');

%figure;
%subplot(2,3,1);
%spy(K_b(:,:,1,100));
%title('K_1(X)');
%subplot(2,3,2);
%spy(K_b(:,:,2,100) > 10^(-4));
%title('K_2(X)');
%subplot(2,3,3);
%spy(K_b(:,:,3,100));
%title('K_3(X)');
%subplot(2,3,4);
%spy(K_b(:,:,4,100));
%title('K_4(X)');
%subplot(2,3,5);
%spy(K_b(:,:,5,100));
%title('K_5(X)');
%subplot(2,3,6);
%spy(K_b(:,:,6,100));
%title('K_6(X)');

k = 1:300;
plot(k,alpha_b1(k),'-b','LineWidth',1);
hold on
plot(k,alpha_b2(k),'--r','LineWidth',1);
plot(k,alpha_b3(k),':m','LineWidth',1);
plot(k,alpha_b4(k),'-.k','LineWidth',1);
plot(k,alpha_b5(k),'.g','LineWidth',1);
plot(k,alpha_b6(k),'.-y','LineWidth',1);
axis tight
hold off
xlabel("Time (t)")
ylabel("Entries of \alpha_b")
legend("\alpha_b_1","\alpha_b_2","\alpha_b_3","\alpha_b_4","\alpha_b_5","\alpha_b_6");
toc