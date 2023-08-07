function [X_unimodal] = project_unimodal(X,non_negativity)
% returns unimodal L2 regression of columns in X, with or without additional projection
% onto non-negative orthant
% INPUT:
% X: matrix to project
% non_negativity: boolean, true for additional projection onto non-negative orthant, false otherwise
% OUTPUT:
% X_unimodal: matrix with unimodal (non-negative) columns

X_unimodal = zeros(size(X));
R = size(X,2);
for r=1:R
    X_unimodal(:,r) = project_unimodal_vector(X(:,r),non_negativity);
end

  

end

