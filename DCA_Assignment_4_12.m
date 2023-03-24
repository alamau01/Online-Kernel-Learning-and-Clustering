%DCA_Assignment_4_12
clear all
clc

load acc_data.mat
load acc_labels_453.mat

num_users = 30;
beta=0;

X = rand(78,4); 
Y = rand(78,4);
alpha_b = 1/17*ones(17,1);
v = 0.001; 
mu_2 = 100; 
K = 4;

for t=1:1000
    for experiment_num = 1:num_users
        activity_labels = [3 4 6];
        num_class = length(activity_labels);
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
        if(t<500)
            data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);  acc_data(j_labels(range_3(1:range_overall)),51:102)];
            gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];
                
            elseif(t<=1000)
                data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);...
                (beta^(600-t)*acc_data(j_labels(range_3(1:range_overall)),51:102))+...
                ((1-beta^(600-t))*acc_data(j_labels(range_2(range_overall+1:2*range_overall)),51:102))];

                gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 2*ones(1,range_overall)];
                %Try beta=0
            else
                data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);...
                (1-beta^(600-t)*acc_data(j_labels(range_3(1:range_overall)),51:102))+...
                ((beta^(600-t))*acc_data(j_labels(range_2(range_overall+1:2*range_overall)),51:102))];

                gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];
                %Try beta=0
            end     
   
            x_in = data;
        
            p = range_overall*3;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
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
        
            %Use an extra index t to store across time
            K_b(:,:,:,t) = cat(3,K_b_gaussian,K_b_poly);
            end
        end
end
    
k = 1;
while k <= 1500
    K_bnew = [];
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
        dh_X = [];
        for i = size(X,1)
            for j = size(X,2)
                dh_X(i,j) = X(i,j)./sqrt(sum(X(i,:).^2));
            end
        end
        
        % calculating the subgradients for the eigenvalues
        sum_Kb = zeros(78,78);
        for b = 1:size(K_b,3)
            sum_Kb = sum_Kb + alpha_b(b)*K_bnew(:,:,b);
        end
        
        [eig_vec, eig_val] = eig(sum_Kb);
        [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
        V = eig_vec(:,pos_sort(1:K));
        alpha_q = [];
        for b = 1:size(K_b,3)
            K_btemp(1:78,1:78) = K_bnew(:,:,b);
            t = transpose(V)*K_btemp*V;
            alpha_q(b,1) = sum(diag(t));
        end
    
        % optimizing problem using projected subgradient descent
        while 1
            X_prev = X;
            alpha_bprev = alpha_b;
            delta_fx = 2*Y*transpose(Y)*X - 2*sum_Kb*Y + v*sign(X_prev) - v*dh_X;
            X = max(X_prev - 0.0001*delta_fx,0);
            
            sum_Kb = zeros(78,78);
            for b = 1:size(K_b,3)
                sum_Kb = sum_Kb + alpha_bprev(b)*K_bnew(:,:,b);
            end
    
            [eig_vec, eig_val] = eig(sum_Kb);
            [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
            V = eig_vec(:,pos_sort(1:K-1));
            alpha_q_1 = [];
            for b = 1:size(K_b,3)
                K_btemp(1:78,1:78) = K_bnew(:,:,b);
                t = transpose(V)*K_btemp*V;
                alpha_q_1(b,1) = sum(diag(t));
            end
            delta_falpha_b = 2*trace(K_bnew(:,:,b).*K_bnew(:,:,b)).*alpha_bprev - 2*trace(K_bnew(:,:,b)*Y*transpose(X_prev)) + mu_2*alpha_q_1 - mu_2*alpha_q;
            alpha_b = Simplex_Projection(alpha_bprev-0.0001*delta_falpha_b);
            if norm(X_prev-X) < 0.1 && norm(alpha_bprev-alpha_b) < 0.1
                break;
            end
        end
        if norm(X_prev2-X_prev) < 0.1 && norm(alpha_bprev2-alpha_bprev) < 0.1
            break;
        end
    end
    
    sum_Kb = zeros(78,78);
    for b = 1:size(K_b,3)
        sum_Kb = sum_Kb + alpha_b(b)*K_bnew(:,:,b);
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
            variable Y(78,4)
            for i = size(Y,1)
                Y_norm = norm(Y(i,:), 1);
            end
            minimize(square_pos(norm(sum_Kb-X*transpose(Y), 'fro')) + v*sum(Y_norm) - v*sum(vec(transpose(dh_Y)).*vec(Y)))
            subject to
            vec(Y)>=0;
        cvx_end
        if norm(Y_prev-Y) < 0.1
            break;
        end
    end
    
    k
    alpha_b
    k = k + 1;
    
end
