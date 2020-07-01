function  [A] = init_coupled_AOADMM_CMTF(Z,varargin)
% initializes all fields of AOADMM solution for CMTF

% Parts are taken from the MATLAB CMTF Toolbox, 2013.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)

%% Parse inputs
params = inputParser;
params.addParameter('init_options', '', @isstruct);

params.parse(varargin{:});
sz         = Z.size;    %size of data sets
lambdas    = params.Results.init_options.lambdas_init; % norms of components in each data set
distr      = params.Results.init_options.distr; % function handle for generating factor matrices (distributions)
normalize  = params.Results.init_options.normalize;
nvecs      = params.Results.init_options.nvecs;
modes      = Z.modes;   % how the data sets are grouped
coupling   = Z.coupling;   % how the modes sets are coupled
constrained_modes = Z.constrained_modes; % which factors are constrained
prox_operators = Z.prox_operators;

coupled_modes = coupling.lin_coupled_modes;
coupling_type = coupling.coupling_type;
coupl_trafo_matrices = coupling.coupl_trafo_matrices;

max_modeid = max(cellfun(@(x) max(x), modes));
if max_modeid ~= length(sz)
    error('Mismatch between size and modes inputs')
end

%% Generate factor matrices
P = length(modes);
nb_modes  = length(sz);
nb_couplings = max(coupled_modes);
A.fac     = cell(nb_modes,1);
A.coupling_fac = cell(nb_couplings,1);
A.constraint_fac = cell(nb_modes,1);
A.coupling_dual_fac = cell(nb_modes,1);
A.constraint_dual_fac = cell(nb_modes,1);

% generate factor matrices
for p=1:P
    for n = modes{p}
        if nvecs % use singular vectors for initialization for each mode separately, even if the coupling is exact!
            A.fac{n} = cmtf_nvecs(Z,n,length(lambdas{p}));
        else % use random initilaization given by distr
            A.fac{n} = feval(distr{n},sz(n),length(lambdas{p}));
            if normalize
                for r=1:length(lambdas{p})
                    A.fac{n}(:,r)=A.fac{n}(:,r)/norm(A.fac{n}(:,r));
                end
            end
        end
    end 
end

for n = 1:nb_modes    
    if constrained_modes(n)
        A.constraint_fac{n} = feval(distr{n},size(A.fac{n},1),size(A.fac{n},2));
        if isempty(prox_operators{n})
            error('No proximal operator provided for mode %s.',num2str(n));
        end
        A.constraint_fac{n} = feval(prox_operators{n},A.constraint_fac{n},1);
        A.constraint_dual_fac{n} = rand(size(A.fac{n}));
    end 
end


%initialize Delta
for n = 1:nb_couplings
     cmodes = find(coupled_modes==n);
     mode1 = cmodes(1);
     ctype = coupling_type(n);
     switch ctype
         case 0
             A.coupling_fac{n} = rand(size(A.fac{mode1}));
             for m=cmodes
                 A.coupling_dual_fac{m} = rand(size(A.coupling_fac{n})); % mu_Deltas have same size as Delta!
             end
         case 1
             A.coupling_fac{n} = rand(size(coupl_trafo_matrices{mode1},1),size(A.fac{mode1},2));
             for m=cmodes
                 A.coupling_dual_fac{m} = rand(size(A.coupling_fac{n}));
             end
         case 2
             A.coupling_fac{n} = rand(size(A.fac{mode1},1),size(coupl_trafo_matrices{mode1},2));
             for m=cmodes
                 A.coupling_dual_fac{m} = rand(size(A.coupling_fac{n}));
             end
         case 3
             A.coupling_fac{n} = rand(size(coupl_trafo_matrices{mode1},2),size(A.fac{mode1},2));
             for m=cmodes
                 A.coupling_dual_fac{m} = rand(size(A.fac{m}));
             end
         case 4
             A.coupling_fac{n} = rand(size(A.fac{mode1},1),size(coupl_trafo_matrices{mode1},1));
             for m=cmodes
                 A.coupling_dual_fac{m} = rand(size(A.fac{m}));
             end
     end              
end

end



