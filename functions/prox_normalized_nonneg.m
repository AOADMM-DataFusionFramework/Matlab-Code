function [Y] = prox_normalized_nonneg(X)
% proximal operator for projecting onto the nonnegative unitsphere
    Y = project_box(X,0,inf); % non-negativity
    for r=1:size(Y,2)
        if norm(Y(:,r),2)==0
            [~,maxcoord] = max(X(:,r));
            Y(maxcoord,r) = 1; %set maximum coordinate to 1, leave rest 0
        else
            Y(:,r) = Y(:,r)./norm(Y(:,r),2); %normalize
        end
    end
end

