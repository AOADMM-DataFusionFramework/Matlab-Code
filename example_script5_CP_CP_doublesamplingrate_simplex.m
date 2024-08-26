%%  example script 5 AOADMM for CMTF 
% In this example, we create a synthetic dataset consisting of two CP
% tensors of order 3 (modes 1,2,3 and modes 4,5,6) and sizes [50,30,40] and [100,70,80], 
% both with 4 components. They are coupled in the first mode, i.e. mode 1 and 4, 
% where it is assumed that tensor 2 was sampled at double sampling rate of that of tensor 1,
% so only every second row of factor matrices 1 and 4 are matched (coupled).
% Therefore, modes 1 and 4 are coupled using coupling type 1.
% Furthermore, the columns of the factor matrix in mode 6 are constructed
% to form a simplex and a column-wise simplex constraint is employed when
% fitting the model.
% Some Gaussian noise is added to the datasets.
% The loss functions is set to be squared Frobenius norm.
%%
close all
clear all
%% add AO-ADMM solver functions to path
addpath(genpath('.\functions'))
%% add other apckages to your path!
addpath(genpath('...\tensor_toolbox-v3.1')) %Tensor toolbox is needed!  MATLAB Tensor Toolbox. Copyright 2017, Sandia Corporation, http://www.tensortoolbox.org/
addpath(genpath('...\L-BFGS-B-C-master')) % LBFGS-B implementation only needed when other loss than Frobenius is used, download here: https://github.com/stephenbeckr/L-BFGS-B-C
addpath(genpath('...\proximal_operators\code\matlab')) % Proximal operator repository needed! download here: http://proximity-operator.net/proximityoperator.html
addpath(genpath('.\functions_for_example_scripts')) 
%% specify synthetic data
sz     = {50,30,40,100,70,80}; %size of each mode
P      = 2; %number of tensors
lambdas_data= {[1 1 1 1], [1 1 1 1]}; % norms of components in each data set (length of each array specifies the number of components in each dataset)
modes  = {[1 2 3], [4 5 6]}; % which modes belong to which dataset: every mode should have its unique number d, sz(d) corresponds to size of that mode
noise = 0.2; %level of noise, for gaussian noise only!
distr_data = {@(x,y) randn(x,y),@(x,y) randn(x,y), @(x,y) randn(x,y),@(x,y) randn(x,y),@(x,y) randn(x,y),@(x,y) rand(x,y)}; % function handle of distribution of data within each factor matrix /or Delta if linearly coupled, x,y are the size inputs %coupled modes need to have same distribution! If not, just the first one will be considered
normalize_columns = 0; %wether or not to normalize columns of the created factor matrices, this might destroy the distribution!
%% specify tensor model
model{1} = 'CP';
model{2} = 'CP';
%% specify couplings
coupling.lin_coupled_modes = [1 0 0 1 0 0]; % which modes are coupled, coupled modes get the same number (0: uncoupled)
coupling.coupling_type = [1]; % for each coupling number in the array lin_coupled_modes, set the coupling type: 0 exact coupling, 1: HC=Delta, 2: CH=Delta, 3: C=HDelta, 4: C=DeltaH
coupling.coupl_trafo_matrices = cell(6,1); % cell array with coupling transformation matrices for each mode (if any, otherwise keep empty)

coupling.coupl_trafo_matrices{1} = eye(50,50);
coupling.coupl_trafo_matrices{4} = zeros(50,100);
for i=1:50
  coupling.coupl_trafo_matrices{4}(i,i+(i-1)) = 1; %take every second entry
end

%% set the fitting function for each dataset: 'Frobenius' for squared
% Frobenius norm, 'KL' for KL divergence, IS for Itakura-Saito, 'beta' for other beta divergences (give beta in loss_function_param),...more todo
loss_function{1} = 'Frobenius';
loss_function{2} = 'Frobenius';
loss_function_param{1} = [];
loss_function_param{2} = [];
%% check model
check_data_input(sz,modes,lambdas_data,coupling,loss_function,model);

%% set initialization options
init_options.lambdas_init = {[1 1 1 1], [1 1 1 1]}; %norms of components in each data set for initialization
init_options.nvecs = 0; % wether or not to use cmtf_nvecs.m funcion for initialization of factor matrices Ci (if true, distr_data and normalize are ignored for Ci, not for Zi)
init_options.distr = distr_data; % distribution of the initial factor matrices and their auxiliary variables
init_options.normalize = 1; % wether or not to normalize the columns of the initial factor matrices (might destroy the distribution)

%% set constraints
constrained_modes = [0 0 0 0 0 1]; % 1 if the mode is constrained in some way, 0 otherwise, put the same for coupled modes!

constraints = cell(length(constrained_modes),1); % cell array of length number of modes containing the type of constraint or regularization for each mode, empty if no constraint
%specify constraints-regularizations for each mode, find the options in the file "List of constraints and regularizations.txt"
constraints{6} = {'simplex column-wise',1}; %  column-wise simplex constraint

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
[X, Atrue, Deltatrue,sigmatrue] = create_coupled_data_example5simplex('model', model, 'size', sz, 'modes', modes, 'lambdas', lambdas_data, 'noise', noise,'coupling',coupling,'normalize_columns',normalize_columns,'distr_data',distr_data,'loss_function',Z.loss_function); %create data
%% create Z.object and normalize
normZ=cell(P,1);
for p=1:P
    Z.object{p} = X{p};
    if strcmp(model{p},'CP')
        normZ{p} = norm(Z.object{p});
        Z.object{p} = Z.object{p}/normZ{p};
    elseif strcmp(model{p},'PAR2')
        normZ{p} = 0;
        for k=1:length(Z.object{p})
            normZ{p} = normZ{p} + norm(Z.object{p}{k},'fro')^2;
        end
        normZ{p} = sqrt(normZ{p});
        for k=1:length(Z.object{p})
            Z.object{p}{k} = Z.object{p}{k}/normZ{p};
        end
    end
end

%% Create random initialization
init_fac = init_coupled_AOADMM_CMTF(Z,'init_options', init_options);

%% set options 

options.Display ='iter'; %  set to 'iter' or 'final' or 'no'
options.DisplayIters = 10;
options.MaxOuterIters = 4000;
options.MaxInnerIters = 5;
options.AbsFuncTol   = 1e-4;
options.OuterRelTol = 1e-8;
options.innerRelPrTol_coupl = 1e-3;
options.innerRelPrTol_constr = 1e-3;
options.innerRelDualTol_coupl = 1e-3;
options.innerRelDualTol_constr = 1e-3;
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
Fit2 = 100*(1-norm(Z.object{2}-full(Zhat{2}))^2/norm(Z.object{2})^2);
%% FMS 
true_ktensor{1} =(ktensor(lambdas_data{1}'./normZ{1},Atrue(modes{1})));
FMS1 = score(Zhat{1},true_ktensor{1});
true_ktensor{2} =(ktensor(lambdas_data{2}'./normZ{2},Atrue(modes{2})));
FMS2 = score(Zhat{2},true_ktensor{2});

%% convergence
figure()
subplot(1,3,1)
semilogy([0:out.OuterIterations],out.func_val_conv)
hold on
semilogy([0:out.OuterIterations],out.func_coupl_conv,'--')
xlabel('iterations')
ylabel('function value')
legend('function value','difference coupling')


subplot(1,3,2)
semilogy(out.time_at_it,out.func_val_conv)
hold on
semilogy(out.time_at_it,out.func_coupl_conv,'--')
xlabel('time in seconds')
ylabel('function value')
legend('function value','difference coupling')

markers = {'+','o','*','x','^','v','s','d','>','<','p','h'};
subplot(1,3,3)
for i=1:5
    plot(out.innerIters(i,:),markers{i})
    hold on
end
xlabel('outer iteration')
ylabel('inner iterations')
legend('mode 1', 'mode 2','mode 3','mode 4','mode 5')
sgtitle('convergence AO-ADMM')




