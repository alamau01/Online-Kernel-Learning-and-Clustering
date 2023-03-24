% DCA_Assignment_4_2
clear all
clc
X = rand(60,4); Y = rand(60,4); alpha_b = 1/6*ones(6,1); v = 0.001; mu_2 = 20; K = 4; %beta = 0.1; T = 0;

k = 0;
while k <= 200
    
    K_b = zeros(60,60,6);
    for t = 1:15
        K_b(t,t,1) = 1 - 0.9999^k;
        K_b(t,t,2) = 0.9999^k;
    end
    
    K_b(31:45,31:45,2) = 1;
    K_b(16:30,16:30,3) = 1;
    K_b(46:60,46:60,4) = 1;
    K_b(:,:,5) = 1;
    K_b(:,:,6) = eye(60);
    
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
        sum_Kb = zeros(60,60);
        for b = 1:size(K_b,3)
            sum_Kb = sum_Kb + alpha_b(b)*K_bnew(:,:,b);
        end
        
        [eig_vec, eig_val] = eig(sum_Kb);
        [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
        V = eig_vec(:,pos_sort(1:K));
        alpha_q = [];
        for b = 1:size(K_b,3)
            K_btemp(1:60,1:60) = K_bnew(:,:,b);
            t = transpose(V)*K_btemp*V;
            alpha_q(b,1) = sum(diag(t));
        end
    
        % optimizing problem using projected subgradient descent
        while 1
            X_prev = X;
            alpha_bprev = alpha_b;
            %sum_Kbnorm = 0;
            %for t = 1:size(X,1)
                %sum_Kbnorm = sum(2*(beta^(t-T))*Y*transpose(Y)*X);
            %end
            delta_fx = 2*Y*transpose(Y)*X - 2*sum_Kb*Y + v*sign(X_prev) - v*dh_X;
            X = max(X_prev - 0.001*delta_fx,0);
            
            sum_Kb = zeros(60,60);
            for b = 1:size(K_b,3)
                sum_Kb = sum_Kb + alpha_bprev(b)*K_bnew(:,:,b);
            end
    
            [eig_vec, eig_val] = eig(sum_Kb);
            [v_sort,pos_sort] = sort(diag(eig_val),'descend'); 
    
            V = eig_vec(:,pos_sort(1:K-1));
            alpha_q_1 = [];
            for b = 1:size(K_b,3)
                K_btemp(1:60,1:60) = K_bnew(:,:,b);
                t = transpose(V)*K_btemp*V;
                alpha_q_1(b,1) = sum(diag(t));
            end
            %sum_Kbnorm1 = 0;
            %for t = size(X,1)
                %sum_Kbnorm1 = sum(2*beta^(t-T)*trace(K_bnew(:,:,b).*K_bnew(:,:,b)).*alpha_bprev - 2*beta^(t-T)*trace(K_bnew(:,:,b)*Y*transpose(X_prev)));
            %end
            M = []; m = [];
            for b = 1:size(K_b,3)
                for b1 = 1:size(K_b,3)
                    M(b,b1) = trace(K_bnew(:,:,b)*K_bnew(:,:,b1));
                end
                m(b) = trace(K_bnew(:,:,b)*Y*transpose(X_prev));
            end
            delta_falpha_b = 2*M*alpha_bprev - 2*m(:) + mu_2*alpha_q_1 - mu_2*alpha_q;
            alpha_b = Simplex_Projection(alpha_bprev-0.001*delta_falpha_b);
            if norm(X_prev-X) < 0.01 && norm(alpha_bprev-alpha_b) < 0.01
                break;
            end
        end
        if norm(X_prev2-X_prev) < 0.01 && norm(alpha_bprev2-alpha_bprev) < 0.01
            break;
        end
    end
    
    sum_Kb = zeros(60,60);
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
            variable Y(60,4)
            for i = size(Y,1)
                Y_norm = norm(Y(i,:), 1);
            end
            %for t = size(Y,1)
                %sum_Kbnorm2 = sum(beta^(t-T)*square_pos(norm(sum_Kb-X*transpose(Y), 'fro')));
            %end
            minimize(square_pos(norm(sum_Kb-X*transpose(Y), 'fro')) + v*sum(Y_norm) - v*sum(vec(transpose(dh_Y)).*vec(Y)))
            subject to
            vec(Y)>=0;
        cvx_end
        alpha_b
        if norm(Y_prev-Y) < 0.01
            break;
        end
    end
    
    k = k+1;
    
end






%%%if k <= 100
        %K_b1 = zeros(60,60,6);
        %K_b1(31:45,31:45,1) = 1;
        %K_b1(1:15,1:15,1) = 1;
        %K_b1(:,:,3) = 1;
        %K_b1(:,:,4) = eye(60);
        %K_b1(46:60,46:60,5) = 1;
        %K_b1(16:30,16:30,6) = 1;
        
        %K_bnew = [];
        %for b = 1:size(K_b,3)
            %K_btemp1(:,:) = K_b1(:,:,b);
            %K_btemp1 = K_btemp1/(eps+trace(K_btemp1));
            %K_b1(:,:,b) = K_btemp1;
            %K_bnew(:,:,b) = K_b1(:,:,b);
        %end
    %elseif k >= 101 && k <= 200
        %K_b2 = zeros(60,60,6);
        %K_b1(46:60,46:60,1) = 1;
        %K_b1(16:30,16:30,2) = 1;
        %K_b2(:,:,4) = eye(60);
        %K_b2(:,:,5) = 1;
        %K_b1(31:45,31:45,6) = 1;
        %K_b1(1:15,1:15,6) = 1;
        
        %K_bnew = [];
        %for b = 1:size(K_b,3)
            %K_btemp2(:,:) = K_b2(:,:,b);
            %K_btemp2 = K_btemp2/(eps+trace(K_btemp2));
            %K_b2(:,:,b) = K_btemp2;
            %K_bnew(:,:,b) = K_b2(:,:,b);
        %end
     %else
   %end