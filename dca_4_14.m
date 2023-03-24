function dca_4_14(X,Y,alpha_b)

load('acc_data.mat')
load('acc_labels_453.mat')

k = 1;
v = 0.001;
mu_2 = 100;
beta = 0.5;
fbnorm_def = 0;
fbnorm_def1 = 0;
alphab_Kb = 0;
K_bnew_beta = zeros(78,78,17);
K_bnew = zeros(78,78,17,3000);
num_users = 30;

while k <= 3000
    for experiment_num = num_users
    activity_labels = [3 4 6];
    %num_class = length(activity_labels);
    %The UniMib data set
    j_labels = find(acc_labels(1:7579,2) == experiment_num);

    range_1 = find(acc_labels(j_labels,1) == activity_labels(1));
    if (floor(length(range_1)/2) ~= length(range_1)/2) 
        range_1(end) = [];
    end

    range_2 = find(acc_labels(j_labels,1) == activity_labels(2));
    if (floor(length(range_2)/2) ~= length(range_2)/2) 
        range_2(end) =[];
    end        
    range_3 = find(acc_labels(j_labels,1) == activity_labels(3));
    if (floor(length(range_3)/2) ~= length(range_3)/2) 
        range_3(end) = [];
    end 
    poss_ranges = [length(range_1),length(range_2),length(range_3)];
    range_overall = min(poss_ranges);

    if (range_overall > 10)
    clear data x_in gnd K_b M K_b_u
    
    for t = 1:k
        if(t<1001)    
            data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);  acc_data(j_labels(range_3(1:range_overall)),51:102)];
            gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];
        elseif(t<=2000)
            data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);...
                    (beta^(1000-t)*acc_data(j_labels(range_3(1:range_overall)),51:102))+...
                    ((1-beta^(1000-t))*acc_data(j_labels(range_2(range_overall+1:2*range_overall)),51:102))];
            gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 2*ones(1,range_overall)];
        else
            data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);...
                    (1-beta^(2000-t)*acc_data(j_labels(range_3(1:range_overall)),51:102))+...
                    ((beta^(2000-t))*acc_data(j_labels(range_2(range_overall+1:2*range_overall)),51:102))];
            gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];
        end
    end

    x_in = data;
    p = range_overall*3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    for i = 1:size(x_in,1)
        x_in(i,:) = x_in(i,:)/norm(x_in(i,:),2);
    end

    delta_gaussian = [10^8 10^6 10^4 10^3 1*10^2 1*10^0 1*10^-1 1*10^-2 1*10^-3 1*10^-3 ];
    delta_poly = [1:7];

    K_b_gaussian=zeros(p,p,length(delta_gaussian));
    K_b_poly=zeros(p,p,length(delta_poly));

    for ii = 1:length(delta_gaussian)
        for i = 1:p
            for j = i:p
                x1 = x_in(i,:);
                x2 = x_in(j,:);
                dist_xx_temp = sum((x1 - x2).^2);
                K_b_u(i,j) = exp(-1*(dist_xx_temp)/(2*delta_gaussian(ii)));       
                K_b_u(j,i) = K_b_u(i,j);
            end
        end

        K_b_u = log(K_b_u+1);
        K_b_u = K_b_u^2;

        one_mat = (1/size(K_b_u,1))*ones(size(K_b_u));
        K= K_b_u;
        K_b_u = K - one_mat*K - K*one_mat + one_mat*K*one_mat;
        K_b_gaussian(:,:,ii) = K_b_u;
    end

    for ii = 1:length(delta_poly)
        for i = 1:p
            for j = i:p
                x1 = x_in(i,:);
                x2 = x_in(j,:);
                K_b_u(i,j) = (sum(x1.*x2))^delta_poly(ii);
                K_b_u(j,i) = K_b_u(i,j);
            end
        end

        K_b_u = log(K_b_u+1);
        K_b_u = K_b_u^2;

        one_mat = (1/size(K_b_u,1))*ones(size(K_b_u));
        K= K_b_u;
        K_b_u = K - one_mat*K - K*one_mat + one_mat*K*one_mat;

        K_b_poly(:,:,ii) = K_b_u;
        %variance_K_b(ii) = (var(reshape(K_b_temp,num_sensors*num_sensors,1)));
    end
    K_b(:,:,:,t) = cat(3,K_b_gaussian,K_b_poly);
    end
    end
    
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
        dh_X = zeros(78,3);
        for i = size(X,1)
            for j = size(Y,2)
                dh_X(i,j) = X(i,j)./sqrt(sum(X(i,:).^2)+10^-3);
            end
        end
        
        % calculating the subgradients for the eigenvalues
        for b = 1:size(K_b,3)
            K_bnew_beta(:,:,b) = beta*K_bnew_beta(:,:,b) + K_bnew(:,:,b,k);
        end
        
        beta_sum_Kb = zeros(78,78);
        for b = 1:size(K_b,3)
            beta_sum_Kb(:,:) = beta_sum_Kb(:,:) + alpha_b(b)*K_bnew_beta(:,:,b);
        end

        [eig_vec, eig_val] = eig(beta_sum_Kb(:,:));
        [~,pos_sort] = sort(diag(eig_val),'descend'); 
    
        V = eig_vec(:,pos_sort(1:3));
        alpha_q = zeros(17,1);
        for b = 1:size(K_b,3)
            K_btemp(1:78,1:78) = K_bnew_beta(:,:,b);
            x3 = transpose(V)*K_btemp(:,:)*V;
            alpha_q(b,1) = sum(diag(x3));
        end
        
        fbnorm_def = beta*fbnorm_def + 2*Y*transpose(Y)*X - 2*beta_sum_Kb(:,:)*Y;
        
        M = zeros(17,17,3000); m = zeros(17,3000);
        for b = 1:size(K_b,3)
            M(b,b,k) = trace(K_bnew(:,:,b,k)*K_bnew(:,:,b,k));
            m(b,k) = trace(K_bnew(:,:,b,k)*Y*transpose(X));
        end
            
        alphab_Kb = beta*alphab_Kb + 2*M(:,:,k)*alpha_b - 2*m(:,k);

        % optimizing problem using projected subgradient descent
        while 1
            X_prev = X;
            alpha_bprev = alpha_b;
            delta_fx = fbnorm_def + v*sign(X_prev) - v*dh_X;
            X = max(X_prev - 0.0001*delta_fx,0);
            
            beta_sum_Kb = zeros(78,78);
            for b = 1:size(K_b,3)
                beta_sum_Kb(:,:) = beta_sum_Kb(:,:) + alpha_b(b)*K_bnew_beta(:,:,b);
            end

            [eig_vec1, eig_val1] = eig(beta_sum_Kb(:,:));
            [~,pos_sort1] = sort(diag(eig_val1),'descend'); 
    
            V1 = eig_vec1(:,pos_sort1(1:2));
            alpha_q_1 = zeros(17,1);
            for b = 1:size(K_b,3)
                K_btemp(1:78,1:78) = K_bnew_beta(:,:,b);
                x4 = transpose(V1)*K_btemp(:,:)*V1;
                alpha_q_1(b,1) = sum(diag(x4));
            end
            
            delta_falpha_b = alphab_Kb + mu_2*alpha_q_1 - mu_2*alpha_q;
            alpha_b = Simplex_Projection(alpha_bprev-0.0001*delta_falpha_b);
            if norm(X_prev-X,1) < 10^-3 && norm(alpha_bprev-alpha_b,1) < 10^-3 %|| norm(Y_prev-Y,1) < 0.06
                break;
            end
        end
        if norm(X_prev2-X_prev,1) < 10^-3 && norm(alpha_bprev2-alpha_bprev,1) < 10^-3 %|| norm(Y_prev2-Y_prev,1) < 0.06
            break;
        end
    end
    
    while 1
        dh_Y = zeros(78,3);
        for i = size(Y,1)
            for j = size(Y,2)
                dh_Y(i,j) = Y(i,j)./sqrt(sum(Y(i,:).^2));
            end
        end
        
        Y_prev = Y;
        cvx_begin quiet
            variable Y(78,4)
            beta_sum_Kb = zeros(78,78);
            for b = 1:size(K_b,3)
                beta_sum_Kb(:,:) = beta_sum_Kb(:,:) + alpha_b(b)*K_bnew_beta(:,:,b);
            end
    
            for i = size(Y,1)
                Y_norm = norm(Y(i,:), 1);
            end
            
            fbnorm_def1 = beta*fbnorm_def1 + 2*X*transpose(X)*Y - 2*beta_sum_Kb(:,:)*X;
            minimize(fbnorm + v*sum(Y_norm) - v*sum(vec(transpose(dh_Y)).*vec(Y)))
            subject to
            vec(Y)>=0;
        cvx_end
        if norm(Y_prev-Y,1) < 10^-3
            break;
        end
    end
    
    %Method 1:
    Matr = (X + Y)/2;
    algo_acc = abs(Matr);
    
    Method 2:
    M_1 = 0.5*(X*Y'+Y*X');
    [U,D] = eig(M_1);
    [~,pos_sort2] = sort(diag(D),'descend');
    Matr1 = U(:,pos_sort2(1:3));
    algo_acc1 = abs(Matr1);
    
    [val, idx] = max(algo_acc,[],2);
    [val1,idx1] = max(algo_acc1,[],2);
    
    a11 = length(find(idx(1:range_overall)==1));
    a12 = length(find(idx(1:range_overall)==2));
    a13 = length(find(idx(1:range_overall)==3));
    
    a21 = length(find(idx(range_overall+1:2*range_overall)==1));
    a22 = length(find(idx(range_overall+1:2*range_overall)==2));
    a23 = length(find(idx(range_overall+1:2*range_overall)==3));
    
    a31 = length(find(idx((2*range_overall)+1:3*range_overall)==1));
    a32 = length(find(idx((2*range_overall)+1:3*range_overall)==2));
    a33 = length(find(idx((2*range_overall)+1:3*range_overall)==3));
    
    a_perf = [a11,a12,a13,a21,a22,a23,a31,a32,a33];
    
    amat = vec2mat(a_perf,3);
    P_m = perms([1:3]);
    
    Pc = zeros(1,6);
    for ind_perm = 1:size(P_m,1)
        Pc(ind_perm) = trace(amat(:,P_m(ind_perm,:)));
    end
    
    acc_max(k) = max(Pc)
    
    a111 = length(find(idx1(1:range_overall)==1));
    a121 = length(find(idx1(1:range_overall)==2));
    a131 = length(find(idx1(1:range_overall)==3));
    
    a212 = length(find(idx1(range_overall+1:2*range_overall)==1));
    a222 = length(find(idx1(range_overall+1:2*range_overall)==2));
    a232 = length(find(idx1(range_overall+1:2*range_overall)==3));
    
    a313 = length(find(idx1((2*range_overall)+1:3*range_overall)==1));
    a323 = length(find(idx1((2*range_overall)+1:3*range_overall)==2));
    a333 = length(find(idx1((2*range_overall)+1:3*range_overall)==3));
    
    a_perf1 = [a111,a121,a131,a212,a222,a232,a313,a323,a333];
    
    amat1 = vec2mat(a_perf1,3);
    P_m1 = perms([1:3]);
    
    Pc1 = zeros(1,6);
    for ind_perm1 = 1:size(P_m1,1)
        Pc1(ind_perm1) = trace(amat1(:,P_m1(ind_perm1,:)));
    end
    
    acc_max1(k) = max(Pc1)
end

k = 1:3000;
figure;
plot(k,acc_max(k),'r--','LineWidth',2);
hold on
plot(k,acc_max1(k),'g-','LineWidth',1);
xlabel('k');
ylabel('acc_max(k) vs acc_max1(k)');
title('User 30');