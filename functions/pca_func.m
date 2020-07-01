function [f] = pca_func(Z,A,Znormsqr,weight)
% PCA_FG Function value of matrix factorization.
%
% Input:  Z: data matrix to be factorized using matrix factorization
%         A: a cell array of two factor matrices
%         Znormsqr: squared Frobenius norm of Z
%
% Output: f: function value, i.e., f = (1/2) ||Z - A{1}*A{2}'||^2
%         G: a cell array of two matrices corresponding to the gradients

% Parts are taken from the MATLAB CMTF Toolbox, 2013.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)

%%
U = A{1};%*diag(lambda);
V = A{2};
UTU = U'*U;
VTV = V'*V;

if exist('Znormsqr','var')
    f1 = Znormsqr;
else
    f1 = sum(Z(:).^2);
end

f2 = 0;
R = size(U,2);
for r = 1:R
    %f2 = f2 + U(:,r)'*Z*V(:,r);
    f2 = f2 + ttv(Z,{U(:,r),V(:,r)},[1 2]);
end

W = UTU .* VTV;
f3 = sum(W(:));

f =  f1 - 2*f2 +  f3;
f = weight *f;


