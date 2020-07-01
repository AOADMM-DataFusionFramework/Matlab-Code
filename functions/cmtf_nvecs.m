function U = cmtf_nvecs(Z,n,r)
% CMTF_NVECS Computes the first r left singular vectors of concatenated
% data sets sharing the mode n, e.g., Z.modes={[1 2 3], [1 4]}, if n=1,
% then U corresponds to the first r singular vectors of Z.object{1} unfolded 
% in the first mode and concatenated with Z.object{2}. If n=2, then U
% corresponds to the first r singular vectors of Z.object{2} unfolded in the
% second mode.
%
%   U = cmtf_nvecs(Z,n,r)
%
% Input:  Z: a struct with object, modes, size, miss fields storing the 
%            coupled data sets (See cmtf_check)
%         n: mode, e.g., if Z.modes={[1 2 3], [1 4]}, n is in {1,2,3,4}.
%         r: number of singular vectors to be computed
%
% Output: U: first r singular vectors
%
% See also CMTF_OPT, ACMTF_OPT, CMTF_CHECK
%
% This is the MATLAB CMTF Toolbox, 2013.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)
%    - (ACMTF)E. Acar, A. J. Lawaetz, M. A. Rasmussen,and R. Bro, Structure-Revealing Data 
%      Fusion Model with Applications in Metabolomics, IEEE EMBC, pages 6023-6026, 2013.
%    - (ACMTF)E. Acar,  E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, 
%      Structure-Revealing Data Fusion, BMC Bioinformatics, 15: 239, 2014.        
%

P = length(Z.object);


A = [];
for p = 1:P
    idx = find(Z.modes{p} == n);
    for i = idx
        if (isa(Z.object{p},'tensor') && (length(size(Z.object{p}))>=3))
            A = [A, double(tenmat(Z.object{p},i))];
        elseif (isa(Z.object{p}, 'sptensor')  && (length(size(Z.object{p}))>=3))
            A = [A, double(sptenmat(Z.object{p},i))];
        else
            if i == 1
                A = [A double(Z.object{p})];
            elseif i == 2
                A = [A double(Z.object{p})'];
            end
        end
        break
    end
end


Y = A*A';
           
[U,~] = eigs(Y, r, 'LM');
