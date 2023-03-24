%DCA_Assignment_4_5_Salinas_org

load('Salinas_gt.mat')
load('Salinas.mat')

k = 1;
X = rand(100,4); 
Y = rand(100,4); 
alpha_b = 1/17*ones(17,1);
K_bnew = zeros(100,100,17,100);
v = 0.001; 
mu_2 = 100;

while k <= 100
    for experiment_num = 1:100
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
                data_x(tally,:) = salinas(x(locations(j)),y(locations(j)),:);
            end
        end
        %objects = [4 7 3 2];
        x_in = data_x;
        gnd = [objects(1)*ones(1,fn) objects(2)*ones(1,fn) objects(3)*ones(1,fn) objects(4)*ones(1,fn)];

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
            K = K_b_u;
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

            one_mat = (1/size(K_b_u,1))*ones(size(K_b_u));
            K= K_b_u;
            K_b_u = K - one_mat*K - K*one_mat + one_mat*K*one_mat;
            K_b_poly(:,:,ii) = K_b_u;
            %variance_cxx(ii) = (var(reshape(cxx_temp,num_sensors*num_sensors,1)));
        end
        K_b(:,:,:,k) = cat(3,K_b_gaussian,K_b_poly);

        for repeat_cntr = 1:1
            for b = 1:size(K_b,3)
                K_btemp(:,:) = K_b(:,:,b,k);
                K_btemp = K_btemp/(eps+trace(K_btemp));
                K_b(:,:,b,k) = K_btemp;
                K_bnew(:,:,b,k) = K_b(:,:,b,k);
            end
    
            while 1
                K = 4;
                X_prev2 = X;
                alpha_bprev2 = alpha_b;
                % subgradient calculations in terms of x and y
                dh_X = zeros(100,4);
                for i = 1:size(X,1)
                    for j = 1:size(X,2)
                        dh_X(i,j) = -3*X(i,j)./sqrt(sum(X(i,:).^2)+10^-8);
                    end
                end

                % calculating the subgradients for the eigenvalues
                sum_Kb = zeros(100,100,100);
                for b = 1:size(K_b,3)
                    sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_b(b)*K_bnew(:,:,b,k);
                end

                [eig_vec, eig_val] = eig(sum_Kb(:,:,k));
                [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 

                V = eig_vec(:,pos_sort(1:K));
                alpha_q = zeros(17,1);
                for b = 1:size(K_b,3)
                    K_btemp(1:100,1:100) = K_bnew(:,:,b,k);
                    t = transpose(V)*K_btemp*V;
                    alpha_q(b,1) = sum(diag(t));
                end

                % optimizing problem using projected subgradient descent
                while 1
                    X_prev = X;
                    alpha_bprev = alpha_b;
                    delta_fx = 2*Y*transpose(Y)*X - 2*sum_Kb(:,:,k)*Y + v*sign(X_prev) - v*dh_X;
                    X = max(X_prev - 0.0001*delta_fx,0);

                    sum_Kb = zeros(100,100,100);
                    for b = 1:size(K_b,3)
                        sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_bprev(b)*K_bnew(:,:,b,k);
                    end

                    [eig_vec1, eig_val1] = eig(sum_Kb(:,:,k));
                    [v_sort1,pos_sort1] = sort(diag(eig_val1),'descend'); 

                    V1 = eig_vec(:,pos_sort1(1:K-1));
                    alpha_q_1 = zeros(17,1);
                    for b = 1:size(K_b,3)
                        K_btemp(1:100,1:100) = K_bnew(:,:,b,k);
                        t1 = transpose(V1)*K_btemp*V1;
                        alpha_q_1(b,1) = sum(diag(t1));
                    end

                    M = zeros(17,17,100); m = zeros(17,100);
                    for b = 1:size(K_b,3)
                        M(b,b,k) = trace(K_bnew(:,:,b,k)*K_bnew(:,:,b,k));
                        m(b,k) = trace(K_bnew(:,:,b,k)*Y*transpose(X));
                    end

                    delta_falpha_b = 2*M(:,:,k)*alpha_bprev - 2*m(:,k) + mu_2*alpha_q_1 - mu_2*alpha_q;
                    alpha_b = Simplex_Projection(alpha_bprev-0.0001*delta_falpha_b);
                    if norm(X_prev-X,1) < 0.06 && norm(alpha_bprev-alpha_b,1) < 0.06
                        break;
                    end
                end
                if norm(X_prev2-X_prev,1) < 0.06 && norm(alpha_bprev2-alpha_bprev,1) < 0.06
                    break;
                end
            end

            sum_Kb = zeros(100,100,100);
            for b = 1:size(K_b,3)
                sum_Kb(:,:,k) = sum_Kb(:,:,k) + alpha_b(b)*K_bnew(:,:,b,k);
            end

            while 1
                dh_Y = zeros(100,4);
                for i = 1:size(Y,1)
                    for j = 1:size(Y,2)
                        dh_Y(i,j) = -3*Y(i,j)./sqrt(sum(Y(i,:).^2)+10^-8);
                    end
                end

                Y_prev = Y;
                cvx_begin quiet
                    variable Y(100,4)
                    for i = size(Y,1)
                        Y_norm = norm(Y(i,:), 1);
                    end
                    minimize(square_pos(norm(sum_Kb(:,:,k)-X*transpose(Y), 'fro')) + v*sum(Y_norm) - v*sum(vec(transpose(dh_Y)).*vec(Y)))
                    subject to
                    vec(Y)>=0;
                cvx_end
                if norm(Y_prev-Y,1) < 0.06
                    break;
                end
            end

            k = k+1;

            %Method 1:
            Matr = (X + Y)/2;
            algo_acc = abs(Matr);

            %Method 2:
            M_1 = 0.5*(X*Y'+Y*X');
            [U,D] = eig(M_1);
            [~,pos_sort2] = sort(diag(D),'descend');
            Matr1 = U(:,pos_sort2(1:4));
            algo_acc1 = abs(Matr1);

            [val, idx] = max(algo_acc,[],2);
            [val1,idx1] = max(algo_acc1,[],2);
            %[val2,idx2] = max(Y,[],2);

            a11 = length(find(idx(1:fn)==objects(1)));
            a12 = length(find(idx(fn+1:2*fn)==objects(2)));
            a13 = length(find(idx(2*fn+1:3*fn)==objects(3)));
            a14 = length(find(idx(3*fn+1:4*fn)==objects(4)));

            a21 = length(find(idx(1:fn)==objects(1)));
            a22 = length(find(idx(fn+1:2*fn)==objects(2)));
            a23 = length(find(idx(2*fn+1:3*fn)==objects(3)));
            a24 = length(find(idx(3*fn+1:4*fn)==objects(4)));

            a31 = length(find(idx(1:fn)==objects(1)));
            a32 = length(find(idx(fn+1:2*fn)==objects(2)));
            a33 = length(find(idx(2*fn+1:3*fn)==objects(3)));
            a34 = length(find(idx(3*fn+1:4*fn)==objects(4)));

            a41 = length(find(idx(1:fn)==objects(1)));
            a42 = length(find(idx(fn+1:2*fn)==objects(2)));
            a43 = length(find(idx(2*fn+1:3*fn)==objects(3)));
            a44 = length(find(idx(3*fn+1:4*fn)==objects(4)));

            a_perf = [a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44];

            amat = vec2mat(a_perf,4);
            P_m = perms([1:4]);

            Pc = zeros(1,8);
            for ind_perm = 1:size(P_m,1)
                Pc(ind_perm) = trace(amat(:,P_m(ind_perm,:)));
            end

            acc_max(k,experiment_num,repeat_cntr) = max(Pc)

            a111 = length(find(idx1(1:fn)==objects(1)));
            a121 = length(find(idx1(fn+1:2*fn)==objects(2)));
            a131 = length(find(idx1(2*fn+1:3*fn)==objects(3)));
            a141 = length(find(idx1(3*fn+1:4*fn)==objects(4)));

            a212 = length(find(idx1(1:fn)==objects(1)));
            a222 = length(find(idx1(fn+1:2*fn)==objects(2)));
            a232 = length(find(idx1(2*fn+1:3*fn)==objects(3)));
            a242 = length(find(idx1(3*fn+1:4*fn)==objects(4)));

            a313 = length(find(idx1(1:fn)==objects(1)));
            a323 = length(find(idx1(fn+1:2*fn)==objects(2)));
            a333 = length(find(idx1(2*fn+1:3*fn)==objects(3)));
            a343 = length(find(idx1(3*fn+1:4*fn)==objects(4)));

            a414 = length(find(idx1(1:fn)==objects(1)));
            a424 = length(find(idx1(fn+1:2*fn)==objects(2)));
            a434 = length(find(idx1(2*fn+1:3*fn)==objects(3)));
            a444 = length(find(idx1(3*fn+1:4*fn)==objects(4)));

            a_perf1 = [a111,a121,a131,a141,a212,a222,a232,a242,a313,a323,a333,a343,a414,a424,a434,a444];

            amat1 = vec2mat(a_perf1,4);
            P_m1 = perms([1:4]);

            Pc1 = zeros(1,8);
            for ind_perm1 = 1:size(P_m1,1)
                Pc1(ind_perm1) = trace(amat1(:,P_m1(ind_perm1,:)));
            end

            acc_max1(k,experiment_num,repeat_cntr) = max(Pc1)

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
        end
    end
end

k = 1:3000;
figure;
plot(k,acc_max(experiment_num,k),'b--','LineWidth',2);
hold on
plot(k,acc_max1(experiment_num,k),'m-','LineWidth',2);
%hold on
%plot(k,acc_max2(k),'r-','LineWidth',2);
xlabel('k');
ylabel('acc_max(k) vs acc_max1(k) vs acc_max2(k)');