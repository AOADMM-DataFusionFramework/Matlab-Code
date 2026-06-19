%%  example script 15 AOADMM for CMTF 
% 

%%
close all
clear all
%%
rng("default")
%% add AO-ADMM solver functions to path
addpath(genpath('.\functions'))
%% add other apckages to your path!
addpath(genpath('...\tensor_toolbox-v3.1')) %Tensor toolbox is needed!  MATLAB Tensor Toolbox. Copyright 2017, Sandia Corporation, http://www.tensortoolbox.org/
addpath(genpath('...\L-BFGS-B-C-master')) % LBFGS-B implementation only needed when other loss than Frobenius is used, download here: https://github.com/stephenbeckr/L-BFGS-B-C
addpath(genpath('...\proximal_operators\code\matlab')) % Proximal operator repository needed! download here: http://proximity-operator.net/proximityoperator.html
%% load and preprocess the data
load('data_for_example15\EEM_NMR_LCMS.mat')
% replace the missing in X with 0
X.data(isnan(X.data))=0;
Z.data = Z.data*diag(1./std(Z.data)); %scaling LCMS features
XX{1} = tensor(X.data); %eem
XX{2} = tensor(Y.data(:,1:10:end,:)); % nmr
XX{3} = tensor(Z.data); %lcms
clear X Z Y
X = XX; clear XX
%% data parameters
sz         = {size(X{1},1), size(X{1},2), size(X{1},3), size(X{2},1), size(X{2},2), size(X{2},3), size(X{3},1), size(X{3},2)}; 
P          = 3;
modes      = {[1 2 3], [4 5 6], [7 8]}; % put different numbers for every mode
%% specify tensor model
model{1} = 'CP';
model{2} = 'CP';
model{3} = 'CP';
%% specify couplings
coupling.lin_coupled_modes = [1 0 0 1 0 0 1 0]; % which modes are coupled, coupled modes get the same number (0: uncoupled)
coupling.coupling_type = [4]; % for each coupling number in the array lin_coupled_modes, set the coupling type: 0 exact coupling, 1: HC=Delta, 2: CH=Delta, 3: C=HDelta, 4: C=DeltaH, 5: H1C=DeltaH2
coupling.coupl_trafo_matrices = cell(8,1); % cell array with coupling transformation matrices for each mode (if any, otherwise keep empty)

coupling.coupl_trafo_matrices{1} = [diag(ones(3,1));zeros(3,3)];
coupling.coupl_trafo_matrices{4} = [diag(ones(5,1));zeros(1,5)];
T = [diag(ones(4,1)) zeros(4,1)];
T = [T; zeros(1,5); 0 0 0 0 1];
coupling.coupl_trafo_matrices{7} = T;
%% set the fitting function for each dataset: 'Frobenius' for squared
% Frobenius norm, 'KL' for KL divergence, IS for Itakura-Saito, 'beta' for other beta divergences (give beta in loss_function_param),...more todo
loss_function{1} = 'Frobenius';
loss_function{2} = 'Frobenius';
loss_function{3} = 'Frobenius';
loss_function_param{1} = [];
loss_function_param{2} = [];
loss_function_param{3} = [];

%% set initialization options
init_options.lambdas_init = {[1 1 1], [1 1 1 1 1],[1 1 1 1 1]}; %for initialization
init_options.nvecs     = 0; % use cmtf_nvecs.m funcion for initialization of factor matrices Ci (if 1, distr_data and normalize are ignored for Ci, not for Zi)
init_options.distr     = {@(x,y) rand(x,y), @(x,y) rand(x,y),@(x,y) rand(x,y),@(x,y) rand(x,y),@(x,y) rand(x,y), @(x,y) rand(x,y),@(x,y) rand(x,y),@(x,y) rand(x,y),@(x,y) rand(x,y)};% distribution of the initial factor matrices and their auxiliary variables
init_options.normalize = 0; % wether or not to normalize the columns of the initial factor matrices (might destroy the distribution)

%% set constraints
constrained_modes = ones(8,1); % 1 if the mode is constrained in some way, 0 otherwise, put the same for coupled modes!

constraints = cell(length(constrained_modes),1); % cell array of length number of modes containing the type of constraint or regularization for each mode, empty if no constraint
%specify constraints-regularizations for each mode, find the options in the file "List of constraints and regularizations.txt"
for i =1:8
    constraints{i} = {'non-negativity'};
end

%% set weights
weights = [1/3 1/3 1/3]; %weight w_i for each data set

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

%% create Z.object and normalize
normZ=cell(P,1);
for p=1:P
    Z.object{p} = X{p};
    normZ{p} = norm(Z.object{p});
    Z.object{p} = Z.object{p}/normZ{p};
end

%% set options 
options.Display ='iter'; %  set to 'iter' or 'final' or 'no'
options.DisplayIters = 1000;
options.MaxOuterIters = 20000;
options.MaxInnerIters = 5;
options.AbsFuncTol   = 1e-4;
options.OuterRelTol = 1e-10;
options.innerRelPrTol_coupl = 1e-3;
options.innerRelPrTol_constr = 1e-3;
options.innerRelDualTol_coupl = 1e-3;
options.innerRelDualTol_constr = 1e-3;
options.bsum = 0; % wether or not to use AO with BSUM regularization
options.eps_log = 1e-10; % for KL divergence log(x+eps) for numerical stability

%% compute
rand_runs     = 10;
AOADMM_maxit_reached = 0;
for n=1:rand_runs
    %Create random initialization
    if n==1
        init_options.nvecs = 1;
        init_fac = init_coupled_AOADMM_CMTF(Z,'init_options', init_options);
    else
        init_options.nvecs = 0;
        init_fac = init_coupled_AOADMM_CMTF(Z,'init_options', init_options);
   end
    % AOADMM
    [AOADMM_result.Zhat{n},AOADMM_result.Fac{n},AOADMM_result.FacInit{n},AOADMM_result.out{n}] = cmtf_AOADMM(Z,'alg_options', options,'init',init_fac);
    if AOADMM_result.out{n}.OuterIterations >= options.MaxOuterIters
        AOADMM_maxit_reached = AOADMM_maxit_reached + 1;
    end

end
%AOADMM
for n=1:rand_runs
    AOADMM_func_vals(n) = AOADMM_result.out{n}.f_tensors;
end
[~,AOADMM_best_run] = min(AOADMM_func_vals);

%% plotting weights
load('data_for_example15\TrueDesign.mat')
A = A([1:26 28:end],:);
names = {'Val-Tyr-Val','Trp-Gly','Phe.','Malto.','Propanol','Noise'};
%% Estimated factors
Zhat = AOADMM_result.Fac{AOADMM_best_run};

%% True design vs. Estimated relative concentrations
figure
for i=1:5
    TT(:,i) = A(:,i)/norm(A(:,i)); % true design
end
for i=1:5
    subplot(3,2,i)
    plot(TT(:,i),'b*-');hold on;
    title(names{i});xlabel('Mixtures');
    ylabel(strcat('C_{1,1}(:,',num2str(i),')'), 'Interpreter','tex');
end
% what is captured by the A matrix
K_EEM   = fixsigns(normalize(ktensor(Zhat.fac(1:3))));  
ord_eem = [2 1 3]; %check the permutation and change accordingly
K_EEM.U{1} = K_EEM.U{1}(:,ord_eem); 
for r=1:size(K_EEM.U{1},2)
    subplot(3,2,r); plot(K_EEM.U{1}(:,r),'r*-'); hold on;
end
%%
figure
for i=1:5
    subplot(3,2,i)
    plot(TT(:,i),'b*-');hold on;
    title(names{i});xlabel('Mixtures');
    ylabel(strcat('C_{2,1}(:,',num2str(i),')'), 'Interpreter','tex');
end
% what is captured by the A matrix
K_NMR   = fixsigns(normalize(ktensor(Zhat.fac(4:6))));    
ord_nmr = [2 1 3 5 4];
K_NMR.U{1} = K_NMR.U{1}(:,ord_nmr); 
for r=1:size(K_NMR.U{1},2)
    subplot(3,2,r); plot(K_NMR.U{1}(:,r),'g*-'); hold on;
end
%%
figure
for i=1:5
    subplot(3,2,i)
    plot(TT(:,i),'b*-');hold on;
    title(names{i});xlabel('Mixtures');
    ylabel(strcat('C_{3,1}(:,',num2str(i),')'), 'Interpreter','tex');
end
% what is captured by the A matrix
K_LCMS   = fixsigns(normalize(ktensor(Zhat.fac(7:8))));   
ord_lcms = [2 1 3 4];
Temp = K_LCMS.U{1}(:,ord_lcms); 
for r=1:size(Temp,2)
    subplot(3,2,r); plot(Temp(:,r),'c*-'); hold on;
end
