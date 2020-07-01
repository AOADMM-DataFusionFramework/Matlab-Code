function [f] = cp_func(Z,A,Znormsqr,weight)
%TT_CP_FG Computes function of the CP function.

% Parts are taken from the MATLAB CMTF Toolbox, 2013.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)

N = ndims(Z);

if ~iscell(A) && ~isa(A,'ktensor')
    error('A must be a cell array or ktensor');
end

if isa(A,'ktensor')
    A = tocell(A);
end
R = size(A{1},2);


%% Upsilon and Gamma
Upsilon = cell(N,1);
for n = 1:N
    Upsilon{n} = A{n}'*A{n};
end



W = ones(R,R);
for m = 1:N
    W = W .* Upsilon{m};
end



%% Calculation

%F1
if exist('Znormsqr','var')
    f_1 = Znormsqr;
else
    f_1 = norm(Z)^2;
end

%% Calculate  F2
U = mttkrp(Z,A,1);
V = A{1} .* U;
f_2 = sum(V(:));

%F3
f_3 = sum(W(:));

%SUM
f = f_1 - 2* f_2 + f_3;
f = weight *f;



