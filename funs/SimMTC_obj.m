function [F,obj] = SimMTC_obj(B_init,k,lambda,p,mu)
[n,anchorNum]=size(B_init{1});
v=length(B_init);


B_tensor=time2frequency(B_init);
B_init_sum=0;
B_tensor_sum=0;
for i=1:v
    B_init_sum=B_init_sum+B_init{i};
    B_tensor_sum=B_tensor_sum+real(B_tensor{i});
end

[nn, ~, vv] = svd(B_init_sum, 'econ');
F=nn(:,1:k);


for iter=1:150
    iter
    % update B
    if iter==1
        H_tensor_sum=B_tensor_sum;
    end
    
    for i = 1:v
        if i==1
            temp1=(H_tensor_sum'*F)./v;
            MM_tensor{i} =(F*temp1'+mu*B_tensor{i})/(mu+1);
        else
            MM_tensor{i} = (F*temp1'+mu*B_tensor{i})/(mu+1);
        end
    end
    [MM] = frequency2time(MM_tensor);
    M = cat(3, MM{:,:});
    M_vector = M(:);
    sX = [n, anchorNum, v];
    [myj, objten] = wshrinkObj_weight_lp(M_vector, lambda*ones(v,1)/(mu+1), sX, 0, 3, p);
    temp_H = reshape(myj, sX);
    for i = 1:v
        H{i} = temp_H(:,:,i);
    end
    H_tensor = time2frequency(H);

    % update F
    H_tensor_sum=0;
    for i=1:v
        H_tensor_sum=H_tensor_sum+real(H_tensor{i});
    end
    [nn, ~, vv] = svd(H_tensor_sum, 'econ');
    F=nn(:,1:k);

    % calculate obj
    temp3=0;
    temp2=(H_tensor_sum'*F)./v;
    temp4=0;
    for i=1:v
        temp3=temp3+sum(sum((H_tensor{i}-F*temp2').^2));
        temp4=temp4+sum(sum((B_tensor{i}-H_tensor{i}).^2));
    end
    obj(iter)=0.5*temp3/v+0.5*mu*temp4/v+lambda*objten;
    if iter>3&&abs((obj(iter)-obj(iter-1))/obj(iter-1))<10^(-5)
        break
    end
end

end