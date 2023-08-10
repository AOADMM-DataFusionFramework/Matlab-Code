function [X_TV] = prox_TV(X, lambda)
% returns the column-wise proximal operator for total variation (TV) regularization with regularization strength lambda of matrix X
% This function uses the function TV_Condat_v2.m by Laurent Condat, which can be
% found here: https://lcondat.github.io/software.html
X_TV = zeros(size(X));
for r=1:size(X,2)
    X_TV(:,r) = TV_Condat_v2(X(:,r), lambda);
end
end

