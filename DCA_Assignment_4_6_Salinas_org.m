%DCA_Assignment_4_6_Salinas_org
tic
load('Salinas_gt.mat')
load('Salinas.mat')

v = 0.001;
mu_2 = 100;
beta = 0.9;
alpha_b = 1/17*ones(17,1);
K_bnew = zeros(100,100,17,300);
fbnorm_def = 0;
alphab_Kb = 0;
fbnorm_def1 = 0;

k = 1;
while k <= 300
    for experiment_num = 100
        rng(experiment_num,'twister')
        objects = randperm(16,4);
        sources(experiment_num,:) = objects;

        p = 100;
        num_sources = 4;
        fn = p/num_sources;
        
        tally=0;
        for i=1:num_sources
            [x,y] = find(salinas_gt==objects(i));
            rng(experiment_num,'twister')
            locations = randperm(length(x), fn);
            for j=1:length(locations)
                tally = tally + 1;
                for t = k
                    if(t<=100)
                        gnd(t,:) = [1*ones(1,fn) 2*ones(1,fn) 3*ones(1,fn) 4*ones(1,fn)];
                        data_x(tally,:) = salinas(x(locations(j)),y(locations(j)),:);
                        if tally == 100
                            tally = 0;
                        end
                    elseif(t<=200)
                        gnd(t,:) = [1*ones(1,fn) 2*ones(1,fn) 2*ones(1,fn) 4*ones(1,fn)];
                        if tally <=25
                            data_x(tally,:) = salinas(x(locations(j)),y(locations(j)),:);
                        elseif tally > 25 && tally <= 50
                            data_x(tally,:) = salinas(x(locations(j)),y(locations(j)),:);
                            locations_2 = locations;
                        elseif tally > 50 && tally <=75
                            locations_2 = locations;
                            data_x(tally,:) = beta^(t-100)*salinas(x(locations(j)),y(locations(j)),:) + (1-beta^(t-100))*salinas(x(locations_2(j)),y(locations_2(j)),:);
                        else
                            data_x(tally,:) = salinas(x(locations(j)),y(locations(j)),:);
                            if tally == 100
                                tally = 0;
                            end
                        end
                    else
                        gnd(t,:) = [1*ones(1,fn) 2*ones(1,fn) 3*ones(1,fn) 4*ones(1,fn)];
                        if tally <=25
                            data_x(tally,:) = salinas(x(locations(j)),y(locations(j)),:);
                        elseif tally > 25 && tally <= 50
                            data_x(tally,:) = salinas(x(locations(j)),y(locations(j)),:);
                            locations_2 = locations;
                        elseif tally > 50 && tally <= 75
                            locations_2 = locations;
                            data_x(tally,:) = (1-beta^(t-200))*salinas(x(locations(j)),y(locations(j)),:) + beta^(t-200)*salinas(x(locations_2(j)),y(locations_2(j)),:);
                        else
                            data_x(tally,:) = salinas(x(locations(j)),y(locations(j)),:);
                            if tally == 100
                                tally = 0;
                            end
                        end
                    end
                end
            end
        end
        
        x_in = data_x;
        %gnd = [objects(1)*ones(1,fn) objects(2)*ones(1,fn) objects(3)*ones(1,fn) objects(4)*ones(1,fn)];

        for i = 1:size(x_in,1)
            x_in(i,:) = x_in(i,:)/max(x_in(i,:),[],'all');
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
            K = K_b_u;
            K_b_u = K - one_mat*K - K*one_mat + one_mat*K*one_mat;
            K_b_gaussian(:,:,ii) = K_b_u; 
        end

        for ii = 1:length(delta_poly)
            for i = 1:p
                for j = i:p
                    x3 = x_in(i,:);
                    x4 = x_in(j,:);
                    K_b_u(i,j) = (sum(x3.*x4))^delta_poly(ii);
                    K_b_u(j,i) = K_b_u(i,j);
                end
            end

            one_mat = (1/size(K_b_u,1))*ones(size(K_b_u));
            K= K_b_u;
            K_b_u = K - one_mat*K - K*one_mat + one_mat*K*one_mat;
            K_b_poly(:,:,ii) = K_b_u;
            %variance_cxx(ii) = (var(reshape(cxx_temp,num_sensors*num_sensors,1)));
        end
        K_b(:,:,:,k) = cat(3,K_b_gaussian,K_b_poly);
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
            [X,Y] = nnmf(K_avg,4);
        elseif (t<=200)
            [X,Y] = nnmf(K_avg,3);
        else
            [X,Y] = nnmf(K_avg,4);
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
        sum_Kb = zeros(100,100,300);
        for b = 1:size(K_b,3)
            sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_b(b)*K_bnew(:,:,b,k);
        end
        
        [eig_vec, eig_val] = eig(sum_Kb(:,:,k));
        [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
        
        for t = k
            if (t<=100)
                V = eig_vec(:,pos_sort(1:4));
                alpha_q = [];
                for b = 1:size(K_b,3)
                    K_btemp(1:100,1:100) = K_bnew(:,:,b,k);
                    t1 = transpose(V)*K_btemp*V;
                    alpha_q(b,1) = sum(diag(t1));
                end
            elseif (t<=200)
                V = eig_vec(:,pos_sort(1:3));
                alpha_q = [];
                for b = 1:size(K_b,3)
                    K_btemp(1:100,1:100) = K_bnew(:,:,b,k);
                    t1 = transpose(V)*K_btemp*V;
                    alpha_q(b,1) = sum(diag(t1));
                end
            else
                V = eig_vec(:,pos_sort(1:4));
                alpha_q = [];
                for b = 1:size(K_b,3)
                    K_btemp(1:100,1:100) = K_bnew(:,:,b,k);
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
            X = max(X_prev - 0.0001*delta_fx,0);
            Y = max(Y_prev - 0.0001*delta_fy,0);

            sum_Kb = zeros(100,100,300);
            for b = 1:size(K_b,3)
                sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_bprev(b)*K_bnew(:,:,b,k);
            end

            [eig_vec1, eig_val1] = eig(sum_Kb(:,:,k));
            [v_sort1,pos_sort1] = sort(diag(eig_val1),'descend'); 
            
            for t = k
                if(t<=100)
                    V2 = eig_vec(:,pos_sort1(1:3));
                    alpha_q_1 = [];
                    for b = 1:size(K_b,3)
                        K_btemp(1:100,1:100) = K_bnew(:,:,b,k);
                        t2 = transpose(V2)*K_btemp*V2;
                        alpha_q_1(b,1) = sum(diag(t2));
                    end
                elseif (t<=200)
                    V2 = eig_vec(:,pos_sort1(1:2));
                    alpha_q_1 = [];
                    for b = 1:size(K_b,3)
                        K_btemp(1:100,1:100) = K_bnew(:,:,b,k);
                        t2 = transpose(V2)*K_btemp*V2;
                        alpha_q_1(b,1) = sum(diag(t2));
                    end
                else
                    V2 = eig_vec(:,pos_sort1(1:3));
                    alpha_q_1 = [];
                    for b = 1:size(K_b,3)
                        K_btemp(1:100,1:100) = K_bnew(:,:,b,k);
                        t2 = transpose(V2)*K_btemp*V2;
                        alpha_q_1(b,1) = sum(diag(t2));
                    end
                end
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
    
    %Method 1:
    %Matr = (X + Y)/2;
    %algo_acc = abs(Matr);
    
    %Method 2:
    %M_1 = 0.5*(X*Y'+Y*X');
    %[U,D] = eig(M_1);
    %[~,pos_sort2] = sort(diag(D),'descend');
    %Matr1 = U(:,pos_sort2(1:4));
    %[val_Y,idx_Y] = max(Y,[],2);
    [val_X,idx_X] = max((X+Y)/2,[],2);
    %result1 = ClusteringMeasure(gnd(k,:),idx_Y);
    result(k,:) = ClusteringMeasure(gnd(k,:),idx_X);
    %algo_acc1 = abs(Matr1);
    
    %Method 3
    %result1 = ClusteringMeasure(X,gnd)
    %Matr = (X + Y)/2;
    %k_means_out1 = kmeans(X,4);
    %result1 = ClusteringMeasure(gnd,max(X,[],2))
    %label1 = bestMap(gnd,k_means_out1);
    %acc_kmeans1(k)=length(find(gnd == label1'))/length(gnd)
    
    %Method 4
    %result2 = ClusteringMeasure(Y,gnd)
    %M_1 = 0.5*(X*Y'+Y*X');
    %[U,D] = eig(M_1);
    %[~,pos_sort2] = sort(diag(D),'descend');
    %Matr1 = U(:,pos_sort2(1:4));
    %k_means_out2 = kmeans(Y,4);
    %result2 = ClusteringMeasure(gnd,max(Y,[],2))
    %label2 = bestMap(gnd,k_means_out2);
    %acc_kmeans2(k)=length(find(gnd == label2'))/length(gnd)
    
    %[val, idx] = max(algo_acc,[],2);
    %[val1,idx1] = max(algo_acc1,[],2);
    
    %a11 = length(find(idx(1:fn)==objects(1)));
    %a12 = length(find(idx(fn+1:2*fn)==objects(2)));
    %a13 = length(find(idx(2*fn+1:3*fn)==objects(3)));
    %a14 = length(find(idx(3*fn+1:4*fn)==objects(4)));
    
    %a21 = length(find(idx(1:fn)==objects(1)));
    %a22 = length(find(idx(fn+1:2*fn)==objects(2)));
    %a23 = length(find(idx(2*fn+1:3*fn)==objects(3)));
    %a24 = length(find(idx(3*fn+1:4*fn)==objects(4)));
    
    %a31 = length(find(idx(1:fn)==objects(1)));
    %a32 = length(find(idx(fn+1:2*fn)==objects(2)));
    %a33 = length(find(idx(2*fn+1:3*fn)==objects(3)));
    %a34 = length(find(idx(3*fn+1:4*fn)==objects(4)));
    
    %a41 = length(find(idx(1:fn)==objects(1)));
    %a42 = length(find(idx(fn+1:2*fn)==objects(2)));
    %a43 = length(find(idx(2*fn+1:3*fn)==objects(3)));
    %a44 = length(find(idx(3*fn+1:4*fn)==objects(4)));
    
    %a_perf = [a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44];
    
    %amat = vec2mat(a_perf,4);
    %result1 = ClusteringMeasure(gnd,amat)
    %P_m = perms([1:4]);
    
    %Pc = zeros(1,8);
    %for ind_perm = 1:size(P_m,1)
        %Pc(ind_perm) = trace(amat(:,P_m(ind_perm,:)));
    %end
    
    %acc_max(k) = max(Pc)
    
    %a111 = length(find(idx1(1:fn)==objects(1)));
    %a121 = length(find(idx1(fn+1:2*fn)==objects(2)));
    %a131 = length(find(idx1(2*fn+1:3*fn)==objects(3)));
    %a141 = length(find(idx1(3*fn+1:4*fn)==objects(4)));
    
    %a212 = length(find(idx1(1:fn)==objects(1)));
    %a222 = length(find(idx1(fn+1:2*fn)==objects(2)));
    %a232 = length(find(idx1(2*fn+1:3*fn)==objects(3)));
    %a242 = length(find(idx1(3*fn+1:4*fn)==objects(4)));
    
    %a313 = length(find(idx1(1:fn)==objects(1)));
    %a323 = length(find(idx1(fn+1:2*fn)==objects(2)));
    %a333 = length(find(idx1(2*fn+1:3*fn)==objects(3)));
    %a343 = length(find(idx1(3*fn+1:4*fn)==objects(4)));
    
    %a414 = length(find(idx1(1:fn)==objects(1)));
    %a424 = length(find(idx1(fn+1:2*fn)==objects(2)));
    %a434 = length(find(idx1(2*fn+1:3*fn)==objects(3)));
    %a444 = length(find(idx1(3*fn+1:4*fn)==objects(4)));
    
    %a_perf1 = [a111,a121,a131,a141,a212,a222,a232,a242,a313,a323,a333,a343,a414,a424,a434,a444];
    
    %amat1 = vec2mat(a_perf1,4);
    %result2 = ClusteringMeasure(gnd,amat1)
    %P_m1 = perms([1:4]);
    
    %Pc1 = zeros(1,8);
    %for ind_perm1 = 1:size(P_m1,1)
        %Pc1(ind_perm1) = trace(amat1(:,P_m1(ind_perm1,:)));
    %end
    
    %acc_max1(k) = max(Pc1)
    
    %a1133 = length(find(idx2(1:fn)==objects(1)));
    %a1233 = length(find(idx2(fn+1:2*fn)==objects(2)));
    %a1333 = length(find(idx2(2*fn+1:3*fn)==objects(3)));
    %a1433 = length(find(idx2(3*fn+1:4*fn)==objects(4)));
    
    %a2133 = length(find(idx2(1:fn)==objects(1)));
    %a2233 = length(find(idx2(fn+1:2*fn)==objects(2)));
    %a2333 = length(find(idx2(2*fn+1:3*fn)==objects(3)));
    %a2433 = length(find(idx2(3*fn+1:4*fn)==objects(4)));
    
    %a3133 = length(find(idx2(1:fn)==objects(1)));
    %a3233 = length(find(idx2(fn+1:2*fn)==objects(2)));
    %a3333 = length(find(idx2(2*fn+1:3*fn)==objects(3)));
    %a3433 = length(find(idx2(3*fn+1:4*fn)==objects(4)));
    
    %a4143 = length(find(idx2(1:fn)==objects(1)));
    %a4243 = length(find(idx2(fn+1:2*fn)==objects(2)));
    %a4343 = length(find(idx2(2*fn+1:3*fn)==objects(3)));
    %a4443 = length(find(idx2(3*fn+1:4*fn)==objects(4)));
    
    %a_perf2 = [a1133,a1233,a1333,a1433,a2133,a2233,a2333,a2433,a3133,a3233,a3333,a3433,a4143,a4243,a4343,a4443];
    
    %amat2 = vec2mat(a_perf2,4);
    %P_m2 = perms([1:4]);
    
    %Pc2 = zeros(1,8);
    %for ind_perm2 = 1:size(P_m2,1)
        %Pc2(ind_perm2) = trace(amat2(:,P_m2(ind_perm2,:)));
    %end
    
    %acc_max2(k) = max(Pc2)
    %result1
    k = k + 1
end
%k = 1:300;
%subplot(3,1,1);
%plot(smooth(result(:,3)),smooth(result(:,2)))
%xlabel('Purity')
%ylabel('NMI')
%title('Proposed Method with Salinas')
%subplot(3,1,2);
%plot(k,smooth(result(:,2)))
%xlabel('no. of iterations (Time)')
%ylabel('NMI')
%subplot(3,1,3);
%plot(k,smooth(result(:,3)))
%xlabel('no. of iterations (Time)')
%ylabel('Purity')
toc