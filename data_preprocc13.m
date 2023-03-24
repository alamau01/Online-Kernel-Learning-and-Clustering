function data_preprocc13(t,experiment_num)
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
if(t<1000)
    data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);  acc_data(j_labels(range_3(1:range_overall)),51:102)];
    gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];

elseif(t<=2000)
    data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);...
           (beta^(1000-t)*acc_data(j_labels(range_3(1:range_overall)),51:102))+...
           ((1-beta^(1000-t))*acc_data(j_labels(range_2(range_overall+1:2*range_overall)),51:102))];
    gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 2*ones(1,range_overall)];
    %Try beta=0

else
    data = [acc_data(j_labels(range_1(1:range_overall)),51:102); acc_data(j_labels(range_2(1:range_overall)),51:102);...
           (1-beta^(2000-t)*acc_data(j_labels(range_3(1:range_overall)),51:102))+...
           ((beta^(2000-t))*acc_data(j_labels(range_2(range_overall+1:2*range_overall)),51:102))];
    gnd(t,:) = [ones(1,range_overall) 2*ones(1,range_overall) 3*ones(1,range_overall)];
    %Try beta=0
end     
   
x_in = data;
p = range_overall*3;
        
for i = 1:size(x_in,1)
    x_in(i,:) = x_in(i,:)/norm(x_in(i,:),2);
end

delta_gaussian = [10^8 10^6 10^4 10^3 1*10^2 1*10^0 1*10^-1 1*10^-2 1*10^-3 1*10^-4];
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
%Use an extra index t to store across time
K_b(:,:,:,t) = cat(3,K_b_gaussian,K_b_poly);
end
end