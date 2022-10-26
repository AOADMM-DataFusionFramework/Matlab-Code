function  [X, A, Delta,sigma] = create_coupled_data(varargin)
% CREATE_COUPLED_DATA generates coupled higher-order tensors and matrices -
% and returns the generated data as a cell array, X, as well as the factors 
% used to generate these data sets as a cell array, A.
% 
%
% Parts are taken from MATLAB CMTF Toolbox, 2013.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)
%    - (ACMTF)E. Acar, A. J. Lawaetz, M. A. Rasmussen,and R. Bro, Structure-Revealing Data 
%      Fusion Model with Applications in Metabolomics, IEEE EMBC, pages 6023-6026, 2013.
%    - (ACMTF)E. Acar,  E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, 
%      Structure-Revealing Data Fusion, BMC Bioinformatics, 15: 239, 2014.        
%

%% Parse inputs
params = inputParser;
params.addParameter('size', [50 30 40 20 10], @iscell);
params.addParameter('modes', {[1 2 3], [1 4], [1 5]}, @iscell);
params.addOptional('noise', 0.1, @(x) x >= 0);
params.addParameter('lambdas', {[1 1 1], [1 1 1], [1 1 0]}, @iscell);
params.addParameter('distr_data',[],@iscell);
params.addParameter('model',{'Frobenius','Frobenius','Frobenius'},@iscell);
params.addParameter('loss_function',[],@iscell);
params.addParameter('loss_function_param',[],@iscell);
params.addParameter('coupling',[],@isstruct);
params.addParameter('normalize_columns',0,@isnumeric);
params.parse(varargin{:});
sz         = params.Results.size;
lambdas    = params.Results.lambdas; % norms of components in each data set
modes      = params.Results.modes;   % how the data sets are coupled
nlevel     = params.Results.noise;
loss_function = params.Results.loss_function;
loss_function_param = params.Results.loss_function_param;
distr_data = params.Results.distr_data;
normalize_columns = params.Results.normalize_columns;
coupling   = params.Results.coupling;
model   = params.Results.model;

lin_coupled_modes = coupling.lin_coupled_modes;
coupling_type = coupling.coupling_type;
coupl_trafo_matrices = coupling.coupl_trafo_matrices;

check_data_input(sz,modes,lambdas,coupling,loss_function,model);

%% Generate factor matrices
P = length(modes);
nb_modes = length(sz);

A = cell(nb_modes,1);
Delta = cell(nb_modes,1);
% generate factor matrices
for p=1:P
    for n = modes{p}
        if lin_coupled_modes(n) == 0 % this mode is not coupled
            A{n} = feval(distr_data{n},sz{n}(1),length(lambdas{p}));
            if normalize_columns
                for r=1:length(lambdas{p})
                    A{n}(:,r)=A{n}(:,r)/norm(A{n}(:,r));
                end
            end
            if (strcmp(model{p},'PAR2') && 2 == find(modes{p}==n))
                %SHIFT PARAFAC
                AA = A{n};
                A{n} = cell(length(sz{n}),1);
                A{n}{1} = AA;
                for k=2:length(sz{n})
                    A{n}{k} = circshift(AA,k-1);
                end
            end
            if n == 6 %generating a simplex
                for r=1:length(lambdas{p})
                    A{n}(:,r)=A{n}(:,r)./sum(A{n}(:,r));
                end
            end
        end
    end
end


nb_c_modes = max(lin_coupled_modes);
for i=1:nb_c_modes
    ctype = coupling_type(i);
    cp_modes = find(lin_coupled_modes==i);
    mode1 = cp_modes(1);
    p_mode1 = find(cellfun(@(x) any(ismember(x,mode1)),modes));
    switch ctype
        case 0 % exactly coupled
            A{mode1} = feval(distr_data{mode1},sz{mode1},length(lambdas{p_mode1}));
            if normalize_columns
                for r=1:size(A{mode1},2)
                    A{mode1}(:,r)=A{mode1}(:,r)/norm(A{mode1}(:,r)); %normalizing columns
                end
            end
            for j=cp_modes(2:end)%initialize as same
                A{j} = A{mode1};
            end
        case 1
            A{mode1} = feval(distr_data{mode1},sz{mode1},length(lambdas{p_mode1}));
            if normalize_columns
                for r=1:size(A{mode1},2)
                    A{mode1}(:,r)=A{mode1}(:,r)/norm(A{mode1}(:,r)); %normalizing columns
                end
            end
            Delta{i} = coupl_trafo_matrices{mode1}*A{mode1};
            for j=cp_modes(2:end)
                A{j} = pinv(coupl_trafo_matrices{j})* Delta{i};
            end
           
        case 2             
            Delta{i} = feval(distr_data{mode1},sz{mode1},size(coupl_trafo_matrices{mode1},2));
            if normalize_columns
                for r=1:size(Delta{i},2)
                    Delta{i}(:,r)=Delta{i}(:,r)/norm(Delta{i}(:,r)); %normalizing columns of DELTA!
                end
            end
            for j=cp_modes %initialize as Delta_i *pinv(H_j)
                A{j} = Delta{i}*pinv(coupl_trafo_matrices{j});
                A{j} = lsqminnorm(coupl_trafo_matrices{j}',Delta{i}'); 
                A{j} = A{j}';
                zerocolumns = find(sum(abs(A{j}))==0);
                A{j}(:,zerocolumns) = feval(distr_data{j},sz{j},length(zerocolumns)); % fill all-zero columns
                if normalize_columns
                    for r=zerocolumns
                        A{j}(:,r)=A{j}(:,r)/norm(A{j}(:,r)); %normalizing columns 
                    end
                end
            end
        case 3
            Delta{i} = feval(distr_data{mode1},size(coupl_trafo_matrices{mode1},2),length(lambdas{cellfun(@(x) any(ismember(x,mode1)), modes)}));
            if normalize_columns
                for r=1:size(Delta{i},2)
                    Delta{i}(:,r)=Delta{i}(:,r)/norm(Delta{i}(:,r)); %normalizing columns of DELTA!
                end
            end
            for j=cp_modes %initialize as H_j*Delta_i
                A{j} = coupl_trafo_matrices{j}*Delta{i};
            end
        case 4
            Delta{i} = feval(distr_data{mode1},sz{mode1},size(coupl_trafo_matrices{mode1},1));
            if normalize_columns
                for r=1:size(Delta{i},2)
                    Delta{i}(:,r)=Delta{i}(:,r)/norm(Delta{i}(:,r)); %normalizing columns of DELTA!
                end
            end
            for j=cp_modes %initialize as Delta_i *(H_j)
                A{j} = Delta{i}*coupl_trafo_matrices{j};
            end
    end
   
end

%% Generate data blocks
P  = length(modes);
X  = cell(P,1);
sigma = zeros(P,1);
for p = 1:P
    if strcmp(model{p},'CP')
        X{p} = full(ktensor(lambdas{p}',A(modes{p})));
        if strcmp(loss_function{p},'Frobenius')
            N    = tensor(randn(size(X{p})));
            sigma(p) = nlevel*norm(X{p})/norm(N); %true standard deviation
            X{p} = X{p} + sigma(p)*N; 
        elseif strcmp(loss_function{p},'KL')
            X{p} = tensor(poissrnd(double(X{p})));
        elseif strcmp(loss_function{p},'IS')
            X{p} = tensor(gamrnd(loss_function_param{p},double(X{p})./loss_function_param{p}));
        end
    elseif strcmp(model{p},'PAR2') % only Frobenius norm implemented for Parafac2 models
        %construct tensor slices
        C = A{modes{p}(3)};
        for k=1:size(C,1)
            X{p}{k} = A{modes{p}(1)}*diag(lambdas{p})*diag(C(k,:))*A{modes{p}(2)}{k}';
        end
        % add noise 
        for k=1:size(C,1)
            normXk = norm(X{p}{k},'fro');
            Nk = randn(size(X{p}{k}));
            normNk = norm(Nk,'fro');
            sigma(p,k) = nlevel*normXk/normNk;
            X{p}{k} = X{p}{k} + sigma(p,k)*Nk;
        end
    end
end

