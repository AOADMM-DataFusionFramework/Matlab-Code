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
model      = Z.model;
constrained_modes = Z.constrained_modes; % which factors are constrained
constraints = Z.constraints;

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
            if strcmp(model{p},'CP')
                A.fac{n} = cmtf_nvecs(Z,n,length(lambdas{p}));
            elseif strcmp(model{p},'PAR2')
                if 1 == find(modes{p}==n)
                    M = [];
                    for k=1:length(sz{modes{p}(2)})
                        M = [M,Z.object{p}{k}];
                    end
                    Y = M*M';
                    [A.fac{n},~] = eigs(Y, length(lambdas{p}), 'LM');
                elseif 2 == find(modes{p}==n)
                    A.DeltaB{p} = rand(length(lambdas{p}),length(lambdas{p}));
                    for k=1:length(sz{n})
                        M = Z.object{p}{k}';
                        Y = M*M';
                        [A.fac{n}{k},~] = eigs(Y, length(lambdas{p}), 'LM');
                        A.P{p}{k} = eye(sz{n}(k),length(lambdas{p}));
                        A.mu_DeltaB{p}{k} = rand(sz{n}(k),length(lambdas{p}));
                    end
                else % mode C
                    A.fac{n} = ones(sz{n},length(lambdas{p}));
                end
            end
        else % use random initilaization given by distr
            if (strcmp(model{p},'PAR2') && 2 == find(modes{p}==n))
                A.DeltaB{p} = rand(length(lambdas{p}),length(lambdas{p}));
                for k=1:length(sz{n})
                    A.fac{n}{k} = feval(distr{n},sz{n}(k),length(lambdas{p}));
                    A.P{p}{k} = eye(sz{n}(k),length(lambdas{p}));
                    A.mu_DeltaB{p}{k} = rand(sz{n}(k),length(lambdas{p}));
                    if normalize
                        for r=1:length(lambdas{p})
                            A.fac{n}{k}(:,r)=A.fac{n}{k}(:,r)/norm(A.fac{n}{k}(:,r));
                        end
                    end
                end
            else %CP model
                A.fac{n} = feval(distr{n},sz{n},length(lambdas{p}));
                if normalize
                    for r=1:length(lambdas{p})
                        A.fac{n}(:,r)=A.fac{n}(:,r)/norm(A.fac{n}(:,r));
                    end
                end
            end
        end
    end 
end

if any(constrained_modes)
    [prox_operators,~] = constraints_to_prox(constrained_modes,constraints,sz);
    for p=1:P
        for n = modes{p}    
            if constrained_modes(n)
                if (strcmp(model{p},'PAR2') && 2 == find(modes{p}==n))
                    for k=1:length(sz{n})
                        A.constraint_fac{n}{k} = feval(distr{n},size(A.fac{n}{k},1),size(A.fac{n}{k},2));
                        if isempty(constraints{n})
                            error('No constraint provided for mode %s.',num2str(n));
                        end
                        A.constraint_fac{n}{k} = feval(prox_operators{n},A.constraint_fac{n}{k},1);
                        A.constraint_dual_fac{n}{k} = rand(size(A.fac{n}{k}));
                    end
               else
                    A.constraint_fac{n} = feval(distr{n},size(A.fac{n},1),size(A.fac{n},2));
                    if isempty(prox_operators{n})
                        error('No proximal operator provided for mode %s.',num2str(n));
                    end
                    A.constraint_fac{n} = feval(prox_operators{n},A.constraint_fac{n},1);
                    A.constraint_dual_fac{n} = rand(size(A.fac{n}));
                end 
            end
        end
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



