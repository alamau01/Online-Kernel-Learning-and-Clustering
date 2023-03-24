function [a_P]=Simplex_Projection(a)

a = real(a);

B=length(a);
a_asc=sort(a,'ascend');


for i=B-1:-1:1
    
   t_i=(sum(a_asc(i+1:B))-1)/(B-i); 
    
    
    
   if(t_i>=a_asc(i))
      t_hat=t_i;
      break;
   end
    
   if(i==1)
      t_hat=(sum(a)-1)/B; 
   end
    
end

a_P(:)=max(a(:)-t_hat*ones(B,1),0);
a_P=a_P(:);