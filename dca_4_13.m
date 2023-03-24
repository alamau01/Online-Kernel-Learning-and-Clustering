%dca_4_13
tic
load('acc_data.mat')
load('acc_labels_453.mat')

k = 1;
v = 0.001;
mu_2 = 100;
beta = 0.85;
alpha_b = 1/17*ones(17,1);
fbnorm_def = 0;
alphab_Kb = 0;
fbnorm_def1 = 0;
K_bnew = zeros(84,84,17,300);
num_users = 6;

while k <= 300
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

    for t = k
        if(t<=100)    
            data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);  acc_data(j_labels(range_3(1:range_overall)),51:102)];
            gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];
        elseif(t<=200)
            data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);...
                    (beta^(100-t)*acc_data(j_labels(range_3(1:range_overall)),51:102))+...
                    ((1-beta^(100-t))*acc_data(j_labels(range_2(range_overall+1:2*range_overall)),51:102))];
            gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 2*ones(1,range_overall)];
        else
            data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);...
                    (1-beta^(200-t)*acc_data(j_labels(range_3(1:range_overall)),51:102))+...
                    ((beta^(200-t))*acc_data(j_labels(range_2(range_overall+1:2*range_overall)),51:102))];
            gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];
        end
    end

    x_in = data;
    p = range_overall*3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    for i = 1:size(x_in,1)
        x_in(i,:) = x_in(i,:)/norm(x_in(i,:),2);
    end

    delta_gaussian = [10^8 10^6 10^4 10^3 10^2 10^0 10^-1 10^-2 10^-3 10^-4];
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
    end
    
    K_b(:,:,:,k) = cat(3,K_b_gaussian,K_b_poly);
    end
    end
    
    for b = 1:size(K_b,3)
        K_btemp(:,:) = K_b(:,:,b,k);
        K_btemp = K_btemp/(eps+trace(K_btemp));
        K_b(:,:,b,k) = K_btemp;
        K_bnew(:,:,b,k) = K_b(:,:,b,k);
    end
    
    K_avg = zeros(p);
    for b = 1:size(K_b,3)
        K_avg = K_avg + alpha_b(b)*K_bnew(:,:,b,k);
    end
    for t = k
        if (t<=100)
            [X,Y] = nnmf(K_avg,3);
        elseif (t<=200)
            [X,Y] = nnmf(K_avg,2);
        else
            [X,Y] = nnmf(K_avg,3);
        end
    end
    Y = Y';
    
    while 1
        X_prev2 = X;
        Y_prev2 = Y;
        alpha_bprev2 = alpha_b;
        % subgradient calculations in terms of x and y
        dh_X = [];
        dh_Y = [];
        for i = 1:size(X,1)
            for j = 1:size(Y,2)
                dh_X(i,j) = -3*X(i,j)./sqrt(sum(X(i,:).^2)+10^-8);
                dh_Y(i,j) = -3*Y(i,j)./sqrt(sum(Y(i,:).^2)+10^-8);
            end
        end
        
        % calculating the subgradients for the eigenvalues
        %for b = 1:size(K_b,3)
            %K_bnew_beta(:,:,b) = beta*K_bnew_beta(:,:,b) + K_bnew(:,:,b,k);
        %end
        
        sum_Kb = zeros(84,84,300);
        for b = 1:size(K_b,3)
            sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_b(b)*K_bnew(:,:,b,k);
        end

        [eig_vec, eig_val] = eig(sum_Kb(:,:,k));
        [~,pos_sort] = sort(diag(eig_val),'descend'); 
    
        for t = k
            if (t<=100)
                V = eig_vec(:,pos_sort(1:3));
                alpha_q = [];
                for b = 1:size(K_b,3)
                    K_btemp(1:84,1:84) = K_bnew(:,:,b,k);
                    t1 = transpose(V)*K_btemp*V;
                    alpha_q(b,1) = sum(diag(t1));
                end
            elseif (t<=200)
                V = eig_vec(:,pos_sort(1:2));
                alpha_q = [];
                for b = 1:size(K_b,3)
                    K_btemp(1:84,1:84) = K_bnew(:,:,b,k);
                    t1 = transpose(V)*K_btemp*V;
                    alpha_q(b,1) = sum(diag(t1));
                end
            else
                V = eig_vec(:,pos_sort(1:3));
                alpha_q = [];
                for b = 1:size(K_b,3)
                    K_btemp(1:84,1:84) = K_bnew(:,:,b,k);
                    t1 = transpose(V)*K_btemp*V;
                    alpha_q(b,1) = sum(diag(t1));
                end
            end
        end
        
        for t = k
            if(t<=100)
                fbnorm_def = beta*fbnorm_def + 2*Y*transpose(Y)*X - 2*sum_Kb(:,:,k)*Y;
                fbnorm_def1 = beta*fbnorm_def1 + 2*X*transpose(X)*Y - 2*sum_Kb(:,:,k)*X;
            elseif(t<=200)
                fbnorm_def = 0;
                fbnorm_def1 = 0;
                fbnorm_def = beta*fbnorm_def + 2*Y*transpose(Y)*X - 2*sum_Kb(:,:,k)*Y;
                fbnorm_def1 = beta*fbnorm_def1 + 2*X*transpose(X)*Y - 2*sum_Kb(:,:,k)*X;
            else
                fbnorm_def = 0;
                fbnorm_def1 = 0;
                fbnorm_def = beta*fbnorm_def + 2*Y*transpose(Y)*X - 2*sum_Kb(:,:,k)*Y;
                fbnorm_def1 = beta*fbnorm_def1 + 2*X*transpose(X)*Y - 2*sum_Kb(:,:,k)*X;
            end
        end
        
        M = zeros(17,17,300); m = zeros(17,300);
        for b = 1:size(K_b,3)
            M(b,b,k) = trace(K_bnew(:,:,b,k)*K_bnew(:,:,b,k));
            m(b,k) = trace(K_bnew(:,:,b,k)*Y*transpose(X));
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
            
            sum_Kb = zeros(84,84,300);
            for b = 1:size(K_b,3)
                sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_b(b)*K_bnew(:,:,b,k);
            end

            [eig_vec1, eig_val1] = eig(sum_Kb(:,:,k));
            [~,pos_sort1] = sort(diag(eig_val1),'descend'); 
    
            for t = k
                if(t<=100)
                    V2 = eig_vec(:,pos_sort1(1:2));
                    alpha_q_1 = [];
                    for b = 1:size(K_b,3)
                        K_btemp(1:84,1:84) = K_bnew(:,:,b,k);
                        t2 = transpose(V2)*K_btemp*V2;
                        alpha_q_1(b,1) = sum(diag(t2));
                    end
                elseif (t<=200)
                    V2 = eig_vec(:,pos_sort1(1:1));
                    alpha_q_1 = [];
                    for b = 1:size(K_b,3)
                        K_btemp(1:84,1:84) = K_bnew(:,:,b,k);
                        t2 = transpose(V2)*K_btemp*V2;
                        alpha_q_1(b,1) = sum(diag(t2));
                    end
                else
                    V2 = eig_vec(:,pos_sort1(1:2));
                    alpha_q_1 = [];
                    for b = 1:size(K_b,3)
                        K_btemp(1:84,1:84) = K_bnew(:,:,b,k);
                        t2 = transpose(V2)*K_btemp*V2;
                        alpha_q_1(b,1) = sum(diag(t2));
                    end
                end
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
    
    %Method 1:
    %Matr = (X + Y)/2;
    %algo_acc = abs(Matr);
    
    %Method 2:
    %M_1 = 0.5*(X*Y'+Y*X');
    %[U,D] = eig(M_1);
    %[~,pos_sort2] = sort(diag(D),'descend');
    %Matr1 = U(:,pos_sort2(1:3));
    [val_X,idx_X] = max((X+Y)/2,[],2);
    result(k,:) = ClusteringMeasure(gnd(k,:),idx_X);
    %algo_acc1 = abs(Matr1);
    
    %[val, idx] = max(algo_acc,[],2);
    %[val1,idx1] = max(algo_acc1,[],2);
    
    %a11 = length(find(idx(1:range_overall)==1));
    %a12 = length(find(idx(1:range_overall)==2));
    %a13 = length(find(idx(1:range_overall)==3));
    
    %a21 = length(find(idx(range_overall+1:2*range_overall)==1));
    %a22 = length(find(idx(range_overall+1:2*range_overall)==2));
    %a23 = length(find(idx(range_overall+1:2*range_overall)==3));
    
    %a31 = length(find(idx((2*range_overall)+1:3*range_overall)==1));
    %a32 = length(find(idx((2*range_overall)+1:3*range_overall)==2));
    %a33 = length(find(idx((2*range_overall)+1:3*range_overall)==3));
    
    %a_perf = [a11,a12,a13,a21,a22,a23,a31,a32,a33];
    
    %amat = vec2mat(a_perf,3);
    %P_m = perms([1:3]);
    
    %Pc = zeros(1,6);
    %for ind_perm = 1:size(P_m,1)
        %Pc(ind_perm) = trace(amat(:,P_m(ind_perm,:)));
    %end
    
    %acc_max(k) = max(Pc)
    
    %a111 = length(find(idx1(1:range_overall)==1));
    %a121 = length(find(idx1(1:range_overall)==2));
    %a131 = length(find(idx1(1:range_overall)==3));
    
    %a212 = length(find(idx1(range_overall+1:2*range_overall)==1));
    %a222 = length(find(idx1(range_overall+1:2*range_overall)==2));
    %a232 = length(find(idx1(range_overall+1:2*range_overall)==3));
    
    %a313 = length(find(idx1((2*range_overall)+1:3*range_overall)==1));
    %a323 = length(find(idx1((2*range_overall)+1:3*range_overall)==2));
    %a333 = length(find(idx1((2*range_overall)+1:3*range_overall)==3));
    
    %a_perf1 = [a111,a121,a131,a212,a222,a232,a313,a323,a333];
    
    %amat1 = vec2mat(a_perf1,3);
    %P_m1 = perms([1:3]);
    
    %Pc1 = zeros(1,6);
    %for ind_perm1 = 1:size(P_m1,1)
        %Pc1(ind_perm1) = trace(amat1(:,P_m1(ind_perm1,:)));
    %end
    
    %acc_max1(k) = max(Pc1)
    k = k + 1
end
k = 1:300;
subplot(3,1,1);
plot(k,smooth(result(:,1)))
xlabel('no. of iterations (Time)')
ylabel('ACC')
title('Proposed Method')
subplot(3,1,2);
plot(k,smooth(result(:,2)))
xlabel('no. of iterations (Time)')
ylabel('NMI')
subplot(3,1,3);
plot(k,smooth(result(:,3)))
xlabel('no. of iterations (Time)')
ylabel('Purity')
toc