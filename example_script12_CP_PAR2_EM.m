%% Example script: coupled CP-PAR2 with missing data in both blocks
%
% A rank-3 CP tensor and a rank-3 regular PARAFAC2 tensor share an exact
% coupling in their first mode (the A factor).
%
%   Tensor 1 (CP, incomplete):   size [20 x 30 x 40], modes [1 2 3]
%                                 ~20% of entries missing at random
%   Tensor 2 (PAR2, incomplete): K=30 slices of size 20 x 25, modes [4 5 6]
%                                 mode 4 (A) exactly coupled to mode 1
%                                 ~20% of entries missing at random per slice
%
% Both tensors are fitted with squared Frobenius loss. Missing entries are
% imputed at each outer iteration via the EM loop.
%
% The example shows:
%   - How to build Z.miss for a CP block (3-D logical array) and a PAR2
%     block (cell array of 2-D logical masks)
%   - Convergence of the joint fit alongside the EM imputation metric
%   - Factor recovery via FMS

%%
close all
clear all
%%
rng("default")
%% add AO-ADMM solver functions to path
addpath(genpath('.\functions'))
%% add other packages to your path!
addpath(genpath('..\tensor_toolbox-v3.1')) %Tensor toolbox is needed!  MATLAB Tensor Toolbox. Copyright 2017, Sandia Corporation, http://www.tensortoolbox.org/
addpath(genpath('..\L-BFGS-B-C-master')) % LBFGS-B implementation only needed when other loss than Frobenius is used, download here: https://github.com/stephenbeckr/L-BFGS-B-C
addpath(genpath('..\proximal_operators\code\matlab')) % Proximal operator repository needed! download here: http://proximity-operator.net/proximityoperator.html
addpath(genpath('.\functions_for_example_scripts'))
%% specify synthetic data
sz     = {20, 30, 40, 20, 25*ones(1,30), 30}; %size of each mode
P      = 2; %number of tensors
lambdas_data= {[1 1 1], [1 1 1]}; % norms of components in each data set (length of each array specifies the number of components in each dataset)
modes  = {[1 2 3], [4 5 6]}; % which modes belong to which dataset: every mode should have its unique number d, sz(d) corresponds to size of that mode
noise = 0.05; %level of noise, for gaussian noise only!
distr_data = {@(x,y) randn(x,y), @(x,y) randn(x,y), @(x,y) randn(x,y), @(x,y) randn(x,y), @(x,y) randn(x,y), @(x,y) rand(x,y)+0.1}; % function handle of distribution of data within each factor matrix /or Delta if linearly coupled, x,y are the size inputs %coupled modes need to have same distribution! If not, just the first one will be considered
normalize_columns = 0; %wether or not to normalize columns of the created factor matrices, this might destroy the distribution!
%% specify tensor model
model{1} = 'CP';
model{2} = 'PAR2';
%% specify couplings
coupling.lin_coupled_modes = [1 0 0 1 0 0]; % which modes are coupled, coupled modes get the same number (0: uncoupled)
coupling.coupling_type = [0]; % for each coupling number in the array lin_coupled_modes, set the coupling type: 0 exact coupling, 1: HC=Delta, 2: CH=Delta, 3: C=HDelta, 4: C=DeltaH, 5: H1C=DeltaH2
coupling.coupl_trafo_matrices = cell(6,1); % cell array with coupling transformation matrices for each mode (if any, otherwise keep empty)
%% set the fitting function for each dataset: 'Frobenius' for squared
% Frobenius norm, 'KL' for KL divergence, IS for Itakura-Saito, 'beta' for other beta divergences (give beta in loss_function_param),...more todo
loss_function{1} = 'Frobenius';
loss_function{2} = 'Frobenius';
loss_function_param{1} = [];
loss_function_param{2} = [];
%% check model
check_data_input(sz,modes,lambdas_data,coupling,loss_function,model);

%% set initialization options
init_options.lambdas_init = {[1 1 1], [1 1 1]}; %norms of components in each data set for initialization
init_options.nvecs = 0; % wether or not to use cmtf_nvecs.m funcion for initialization of factor matrices Ci (if true, distr_data and normalize are ignored for Ci, not for Zi)
init_options.distr = distr_data; % distribution of the initial factor matrices and their auxiliary variables
init_options.normalize = 1; % wether or not to normalize the columns of the initial factor matrices (might destroy the distribution)

%% set constraints
constrained_modes = [0 0 0 0 0 0]; % 1 if the mode is constrained in some way, 0 otherwise, put the same for coupled modes!

constraints = cell(length(constrained_modes),1); % cell array of length number of modes containing the type of constraint or regularization for each mode, empty if no constraint
%specify constraints-regularizations for each mode, find the options in the file "List of constraints and regularizations.txt"

%% add optional ridge regularization performed via primal variable updates, not proximal operators (for no ridge leave field empty), will automatically be added to function value computation
%Z.ridge = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]; % penalties for each mode
%% set weights
weights = [1/2 1/2]; %weight w_i for each data set

%% set lbfgsb options (only needed for loss functions other than Frobenius)
% lbfgsb_options.m = 5;
% lbfgsb_options.printEvery = -1;
% lbfgsb_options.maxIts = 100;
% lbfgsb_options.maxTotalIts = 1000;
% lbfgsb_options.factr = 1e-6/eps;
% lbfgsb_options.pgtol = 1e-4;

%% build model
Z.loss_function = loss_function;
Z.loss_function_param = loss_function_param;
Z.model = model;
Z.modes = modes;
Z.size  = sz;
Z.coupling = coupling;
Z.constrained_modes = constrained_modes;
Z.constraints = constraints;
Z.weights = weights;

%% create data
[X, Atrue, Deltatrue,sigmatrue] = create_coupled_data('model', model, 'size', sz, 'modes', modes, 'lambdas', lambdas_data, 'noise', noise,'coupling',coupling,'normalize_columns',normalize_columns,'distr_data',distr_data,'loss_function',Z.loss_function); %create data

%% create Z.object and normalize
normZ = cell(P,1);
for p = 1:P
    Z.object{p} = X{p};
    if strcmp(model{p},'CP')
        normZ{p} = norm(Z.object{p});
        Z.object{p} = Z.object{p}/normZ{p};
    elseif strcmp(model{p},'PAR2')
        normZ{p} = 0;
        for k = 1:length(Z.object{p})
            normZ{p} = normZ{p} + norm(Z.object{p}{k},'fro')^2;
        end
        normZ{p} = sqrt(normZ{p});
        for k = 1:length(Z.object{p})
            Z.object{p}{k} = Z.object{p}{k}/normZ{p};
        end
    end
end

%% Build missing-data masks (~20% missing at random in each block)
miss_frac = 0.20;

% Tensor 1 (CP): Z.miss{1} is an sptensor (sparse tensor) matching the tensor size;
%                true = observed, false = missing.
sz1          = [sz{1}, sz{2}, sz{3}];   % [20 30 40]
n_entries    = prod(sz1);
miss_mask_CP = true(sz1);
miss_mask_CP(randperm(n_entries, round(miss_frac * n_entries))) = false;
Z.miss{1}    = sptensor(miss_mask_CP);

% Tensor 2 (PAR2): Z.miss{2} must be a cell array of K logical matrices,
%                  one per slice, where true = observed and false = missing.
%                  Each slice X_k = A*diag(C(k,:))*Bk' has size sz{4} x sz{5}(k).
K2             = length(sz{5});   % number of PAR2 slices
miss_cell_PAR2 = cell(K2,1);
for k = 1:K2
    n_rows = sz{4};
    n_cols = sz{5}(k);
    n_k    = n_rows * n_cols;
    mask_k = true(n_rows, n_cols);
    mask_k(randperm(n_k, round(miss_frac * n_k))) = false;
    miss_cell_PAR2{k} = mask_k;
end
Z.miss{2} = miss_cell_PAR2;

%% initialize missing entries with entires of your choice

Z.object{1}(find(~Z.miss{1})) = 0; % initialize missing entries with 0

for k=1:K2
    Z.object{2}{k}(~Z.miss{2}{k}) = 0; % initialize missing entries with 0
end

%% Create random initialization
init_fac = init_coupled_AOADMM_CMTF(Z,'init_options', init_options);

%% set options
options.Display ='iter'; %  set to 'iter' or 'final' or 'no'
options.DisplayIters = 10;
options.MaxOuterIters = 4000;
options.MaxInnerIters = 5;
options.AbsFuncTol   = 1e-7;
options.OuterRelTol = 1e-8;
options.innerRelPrTol_coupl = 1e-5;
options.innerRelPrTol_constr = 1e-5;
options.innerRelDualTol_coupl = 1e-5;
options.innerRelDualTol_constr = 1e-5;
options.bsum = 0; % wether or not to use AO with BSUM regularization
%options.bsum_weight = 1e-3; %set the penalty parameter (mu) for BSUM regularization
options.eps_log = 1e-10; % for KL divergence log(x+eps) for numerical stability
%options.lbfgsb_options = lbfgsb_options;

%% run algorithm
fprintf('AOADMM cmtf \n')
tic
[Zhat,Fac,FacInit,out] = cmtf_AOADMM(Z,'alg_options',options,'init',init_fac,'init_options',init_options);
toc

%% FIT
Fit1 = 100*(1-norm(Z.object{1}-full(Zhat{1}))^2/norm(Z.object{1})^2);
Fit2 = 0;
Fitx = 0;
for k=1:length(sz{5})
    Fit2 = Fit2 + norm(Z.object{2}{k}-Zhat{2}.A*diag(Zhat{2}.C(k,:))*Zhat{2}.Bk{k}','fro')^2;
    Fitx    = Fitx    + norm(Z.object{2}{k},'fro')^2;
end
Fit2 = 100*(1-Fit2/Fitx);
  
%% FMS 
true_ktensor{1} =(ktensor(lambdas_data{1}'./normZ{1},Atrue(modes{1})));
FMS1 = score(Zhat{1},true_ktensor{1},'lambda_penalty',false);

FMS2_A = score(ktensor(ones(3,1),Zhat{2}.A),ktensor(ones(3,1),Atrue{4}),'lambda_penalty',false);
FMS2_C = score(ktensor(ones(3,1),Zhat{2}.C),ktensor(ones(3,1),Atrue{6}),'lambda_penalty',false);
SollargeB = [];
largeB = [];
for k=1:length(sz{5})
    SollargeB = [SollargeB;Zhat{2}.Bk{k}];
    largeB = [largeB;Atrue{5}{k}];
end
FMS2_B = score(ktensor(ones(3,1),SollargeB),ktensor(ones(3,1),largeB),'lambda_penalty',false);
%% convergence
figure()
subplot(1,3,1)
semilogy([0:out.OuterIterations],out.func_val_conv)
hold on
semilogy([0:out.OuterIterations],out.func_coupl_conv,'--')
hold on
semilogy([0:out.OuterIterations],out.func_constr_conv,':')
hold on
semilogy([0:out.OuterIterations],out.func_PAR2_coupl,'+')
hold on
semilogy([0:out.OuterIterations],out.func_rel_missing,'*')
xlabel('iterations')
ylabel('function value')
legend('function value','difference coupling','difference constraints','difference PAR2 coupling','difference missing')


subplot(1,3,2)
semilogy(out.time_at_it,out.func_val_conv)
hold on
semilogy(out.time_at_it,out.func_coupl_conv,'--')
hold on
semilogy(out.time_at_it,out.func_constr_conv,':')
hold on
semilogy(out.time_at_it,out.func_PAR2_coupl,'+')
hold on
semilogy(out.time_at_it,out.func_rel_missing,'*')
xlabel('time in seconds')
ylabel('function value')
legend('function value','difference coupling','difference constraints','difference PAR2 coupling','difference missing')

markers = {'+','o','*','x','^','v','s','d','>','<','p','h'};
subplot(1,3,3)
for i=1:6
    plot(out.innerIters(i,:),markers{i})
    hold on
end
xlabel('outer iteration')
ylabel('inner iterations')
legend('mode 1', 'mode 2','mode 3','mode 4','mode 5','mode 6')
sgtitle('convergence AO-ADMM')

%% Simulated vs imputed values at missing positions
figure()

% --- Block 1: CP ---
cp_true_arr = double(X{1}) / normZ{1};   % ground-truth, normalized
cp_fit_arr  = double(full(Zhat{1}));      % model fit, normalized
miss_idx_CP = ~double(Z.miss{1});                 % true = missing

subplot(1,2,1)
scatter(cp_true_arr(miss_idx_CP), cp_fit_arr(miss_idx_CP), 5, 'filled', 'MarkerFaceAlpha', 0.3)
hold on
lims_cp = [min([cp_true_arr(miss_idx_CP); cp_fit_arr(miss_idx_CP)]), ...
           max([cp_true_arr(miss_idx_CP); cp_fit_arr(miss_idx_CP)])];
plot(lims_cp, lims_cp, 'r--', 'LineWidth', 1.5)
xlabel('Simulated (true) value')
ylabel('Imputed (fitted) value')
r_cp = corr(cp_true_arr(miss_idx_CP), cp_fit_arr(miss_idx_CP));
title(sprintf('Block 1: CP  (r = %.3f)', r_cp))
axis equal tight

% --- Block 2: PAR2 ---
n_miss_par2   = sum(cellfun(@(m) sum(~m(:)), Z.miss{2}));
par2_true_vec = zeros(n_miss_par2, 1);
par2_fit_vec  = zeros(n_miss_par2, 1);
idx = 1;
for k = 1:K2
    slice_true = X{2}{k} / normZ{2};
    slice_fit  = Zhat{2}.A * diag(Zhat{2}.C(k,:)) * Zhat{2}.Bk{k}';
    miss_k     = ~Z.miss{2}{k};
    n_k        = sum(miss_k(:));
    par2_true_vec(idx:idx+n_k-1) = slice_true(miss_k);
    par2_fit_vec(idx:idx+n_k-1)  = slice_fit(miss_k);
    idx = idx + n_k;
end

subplot(1,2,2)
scatter(par2_true_vec, par2_fit_vec, 5, 'filled', 'MarkerFaceAlpha', 0.3)
hold on
lims_p2 = [min([par2_true_vec; par2_fit_vec]), max([par2_true_vec; par2_fit_vec])];
plot(lims_p2, lims_p2, 'r--', 'LineWidth', 1.5)
xlabel('Simulated (true) value')
ylabel('Imputed (fitted) value')
r_p2 = corr(par2_true_vec, par2_fit_vec);
title(sprintf('Block 2: PAR2  (r = %.3f)', r_p2))
axis equal tight

sgtitle('Simulated vs imputed values at missing entries')