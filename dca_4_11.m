function dca_4_11(X,Y,alpha_b,v,mu_2,K,beta)

k = 1;
fbnorm_def = 0;
alphab_Kb = 0;
fbnorm_def1 = 0;
K_bnew_beta = zeros(78,78,17);

load acc_data.mat
load acc_labels_453.mat

num_users = 30 ;

for experiment_num = 1:num_users

    disp(experiment_num)
        
        
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
        
            data_1 = acc_data(j_labels(range_1(1:range_overall)),51:102);
            data_2 = acc_data(j_labels(range_2(1:range_overall)),51:102);
            data_3_store = acc_data(j_labels(range_3(1:range_overall)),51:102);
        
            data = [data_1;data_2;data_3_store];

        
            %gnd = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];
        
            x_in = data;
        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
            for i = 1:size(x_in,1)
                x_in(i,:) = x_in(i,:)/norm(x_in(i,:),2);
            end
        end
end

while k <= 2100
    for t = 1:k
        data_3 = (beta^k)*data_3_store - (1-beta^k)*data_1;
        data = [data_1; data_2; data_3];
        
        x_in = data;
        p = range_overall*3;
        
        for i = 1:size(x_in,1)
            x_in(i,:) = x_in(i,:)/norm(x_in(i,:),2);
        end


        delta_gaussian = [10^8 10^6 10^4 10^3 1*10^2 1*10^0 1*10^-1 1*10^-2 1*10^-3 1*10^-4 ];
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

        K_b = cat(3,K_b_gaussian,K_b_poly);
        
        K_bnew = zeros(78,78,17,2100);
        for b = 1:size(K_b,3)
            K_btemp(:,:) = K_b(:,:,b);
            K_btemp = K_btemp/(eps+trace(K_btemp));
            K_b(:,:,b,t) = K_btemp;
            K_bnew(:,:,b,t) = K_b(:,:,b,t);
        end
    end
    
    while 1
        X_prev2 = X;
        Y_prev2 = Y;
        alpha_bprev2 = alpha_b;
        % subgradient calculations in terms of x and y
        dh_X = zeros(78,4);
        dh_Y = zeros(78,4);
        for i = size(X,1)
            for j = size(Y,2)
                dh_X(i,j) = X(i,j)./sqrt(sum(X(i,:).^2));
                dh_Y(i,j) = Y(i,j)./sqrt(sum(Y(i,:).^2));
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
        size(eig_vec)
        V = eig_vec(:,pos_sort(1:K));
        alpha_q = zeros(17,1);
        for b = 1:size(K_b,3)
            K_btemp(1:78,1:78) = K_bnew_beta(:,:,b);
            x = transpose(V)*K_btemp(:,:)*V;
            alpha_q(b,1) = sum(diag(x));
        end
        
        %K_bnew_beta(:,:,b) = sum(beta^(k-2100))*K_bnew(:,:,b,2100);
        
        fbnorm_def = beta*fbnorm_def + 2*Y*transpose(Y)*X - 2*beta_sum_Kb(:,:)*Y;
        fbnorm_def1 = beta*fbnorm_def1 + 2*X*transpose(X)*Y - 2*beta_sum_Kb(:,:)*X;
        
        M = zeros(17,17,2100); m = zeros(17,2100);
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
            
            beta_sum_Kb = zeros(78,78);
            for b = 1:size(K_b,3)
                beta_sum_Kb(:,:) = beta_sum_Kb(:,:) + alpha_b(b)*K_bnew_beta(:,:,b);
            end
 
            [eig_vec, eig_val] = eig(beta_sum_Kb(:,:));
            [~,pos_sort] = sort(diag(eig_val),'descend'); 
    
            V = eig_vec(:,pos_sort(1:K-1));
            alpha_q_1 = zeros(17,1);
            for b = 1:size(K_b,3)
                K_btemp(1:78,1:78) = K_bnew_beta(:,:,b);
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
    
    k
    %alpha_b
    %X
    k = k+1;
    
end