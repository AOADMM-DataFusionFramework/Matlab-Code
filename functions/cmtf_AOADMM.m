function [Zhat,Fac, G,out] = cmtf_AOADMM(Z,varargin)
% takes care of initialization, runs AOADMM and normalizes the output
% Fac constains factor matrices, auxiliary variables and dual variables
% Zhat constains the normalized factor matrices
% G contains all initial variables
% out contains information like number of iterations, number of inner
% iterations, stopping criteria responsible for termination and the
% function value

%% Error checking
%TODO

%% Set parameters
params = inputParser;
params.addParameter('init', 'random', @(x) (isstruct(x) || ismember(x,{'random'})));
params.addParameter('alg_options', '', @isstruct);
params.addOptional('init_options', '', @isstruct);
params.parse(varargin{:}); 
options = params.Results.alg_options;
init = params.Results.init_options;

%%
P = numel(Z.object);
%% Constraints
[prox_operators,reg_func] = constraints_to_prox(Z.constrained_modes,Z.constraints,Z.size);
Z.prox_operators = prox_operators;
Z.reg_func = reg_func;

% Z.prox_operators = cell(length(Z.constrained_modes),1);
% Z.reg_func = cell(length(Z.constrained_modes),1);
% for m=1:length(Z.constrained_modes)
%     if Z.constrained_modes(m)
%         if strcmp(Z.constraints{m}{1},'non-negativity')
%             Z.prox_operators{m} = @(x,rho) project_box(x,0,inf); % non-negativity
%         elseif strcmp(Z.constraints{m}{1},'box')
%             l = Z.constraints{m}{2};
%             u = Z.constraints{m}{3};
%             Z.prox_operators{m} =  @(x,rho) project_box(x,l,u); %box constraints with lower bound l and upper bound u
%         elseif strcmp(Z.constraints{m}{1},'simplex column-wise')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) project_simplex(x, eta, 1); %simplex constraint column-wise
%         elseif strcmp(Z.constraints{m}{1},'simplex row-wise')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) project_simplex(x, eta, 2); %simplex constraint column-wise
%         elseif strcmp(Z.constraints{m}{1},'non-decreasing')
%             Z.prox_operators{m} = @(x,rho) project_monotone(x, 1); % monotonicity column-wise 
%         elseif strcmp(Z.constraints{m}{1},'non-increasing')
%             Z.prox_operators{m} = @(x,rho) -project_monotone(-x, 1); % monotonicity column-wise 
%         elseif strcmp(Z.constraints{m}{1},'unimodality')
%             nn = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) project_unimodal(x,nn); % unimodality column-wise
%         elseif strcmp(Z.constraints{m}{1},'l1-ball')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) project_L1(x, eta, 1); %(hard) l1 sparsity column-wise (||x||_1<=eta):
%         elseif strcmp(Z.constraints{m}{1},'l2-ball')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) project_L2(x, eta, 1); %(hard) l2 normlization column-wise (||x||_2<=eta):
%         elseif strcmp(Z.constraints{m}{1},'non-negative l2-ball')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) project_L2(project_box(x,0,inf),eta,1); %(hard) l2 normlization column-wise (||x||_2<=eta)
%         elseif strcmp(Z.constraints{m}{1},'non-negative l2-sphere')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) prox_normalized_nonneg(x); %(hard) l2 normlization column-wise (||x||_2<=eta) and non-negativity (not convex!)
%         elseif strcmp(Z.constraints{m}{1},'orthonormal')
%             Z.prox_operators{m} = @(x,rho) project_ortho(x); %orthonormal columns
%         elseif strcmp(Z.constraints{m}{1},'l1 regularization')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) prox_abs(x,eta/rho*ones(size(x))); % l1 sparsity regularization
%             Z.reg_func{m} = @(x) eta*sum(vecnorm(x,1));
%         elseif strcmp(Z.constraints{m}{1},'l0 regularization')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) prox_zero(x,eta/rho*ones(size(x))); % l0 sparsity regularization (not convex!)
%             Z.reg_func{m} = @(x) eta*nnz(x);
%         elseif strcmp(Z.constraints{m}{1},'l2 regularization')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) prox_L2(x, eta/rho, 1); % l2 regularization
%             Z.reg_func{m} = @(x) eta*sum(vecnorm(x,2));
%         elseif strcmp(Z.constraints{m}{1},'ridge')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} = @(x,rho) 1/(2*(eta/rho)+1).*x; % ridge
%             Z.reg_func{m} = @(x) eta*norm(x,'fro')^2;
%         elseif strcmp(Z.constraints{m}{1},'quadratic regularization')
%             eta = Z.constraints{m}{2};
%             L = Z.constraints{m}{3};
%             I = speye(size(L));
%             Z.prox_operators{m} = @(x,rho) (2*eta/rho.*L+I)\x; 
%             Z.reg_func{m} = @(x) eta*trace(x'*L*x);
%          elseif strcmp(Z.constraints{m}{1},'GL smoothness')
%             eta = Z.constraints{m}{2};
%             szm = Z.size{m}(1); % DOES NOT WORK FOR THE Bk MODE OF PARAFAC2 WHEN Bk's HAVE DIFFERENT SIZES!!!! TODO: WRITE OWN FUNCTION!
%             L = diag(2*ones(szm,1)) + diag(-1*ones(szm-1,1),1) + diag(-1*ones(szm-1,1),-1);
%             L(1,1) = 1;
%             L(end,end) = 1;
%             L = sparse(L);
%             I = speye(size(L));
%             Z.prox_operators{m} = @(x,rho) (2*eta/rho.*L+I)\x; 
%             Z.reg_func{m} = @(x) eta*trace(x'*L*x);
%          elseif strcmp(Z.constraints{m}{1},'TV regularization')
%             eta = Z.constraints{m}{2};
%             Z.prox_operators{m} =  @(x,rho) prox_TV(x,eta/rho); 
%             Z.reg_func{m} = @(x) eta*sum(sum(x(2:end,:)-x(1:end-1,:)));
%          elseif strcmp(Z.constraints{m}{1},'custom')
%              Z.prox_operators{m} = Z.constraints{m}{2};
%              if length(Z.constraints{m})>2
%                  Z.reg_func{m} = Z.constraints{m}{3};
%              end
%          end
%     end
% end
%% Initialization
if isstruct(params.Results.init)
    G = params.Results.init;
elseif strcmpi(params.Results.init,'random')
    if isempty(init)
        error('init_options are missing as input in cmtf_AOADMM.');
    end
    G = init_coupled_AOADMM_CMTF(Z, 'init_options',init);
else
    error('Initialization type not supported')
end
%% Loss function
Znorm_const = cell(P,1);
fh = cell(P,1); % function handles for lossfunction 
gh = cell(P,1); % function handles for gradient 
lscalar = cell(P,1); %lower bound for m
uscalar = cell(P,1); % upper bound for m
for p = 1:P
    if strcmp(Z.loss_function{p},'Frobenius')
        if strcmp(Z.model{p},'CP')
            if isa(Z.object{p},'tensor') || isa(Z.object{p},'sptensor')
                Znorm_const{p} = norm(Z.object{p})^2;
            else
                Znorm_const{p} = norm(Z.object{p},'fro')^2;
            end
        elseif strcmp(Z.model{p},'PAR2')
            Znorm_const{p} = 0;
            for k=1:length(Z.object{p})
                Znorm_const{p} = Znorm_const{p} + norm(Z.object{p}{k},'fro')^2;
            end
        end
        
        fh{p} = [];
        gh{p} = [];
        lscalar{p} = [];
        uscalar{p} = [];
    elseif strcmp(Z.loss_function{p},'KL') % f(x,m)=xlogx - xlogm -x+m, m>=0
        if ~valid_natural(Z.object{p})
            warning('Using ''%s'' but tensor is not count', Z.loss_function{p});
        end
        log_ten = double(tenfun(@(x) reallog(x+options.eps_log),Z.object{p})); % add epsilon to avoid problems at x=0
        Znorm_const{p} = collapse(minus(times(Z.object{p},log_ten),Z.object{p}));
        fh{p} = @(X,M) tenfun(@(x,m) m - x.*log(m + options.eps_log),X,M); % X:data, M:model(in tensor format)
        gh{p} = @(X,M) tenfun(@(x,m) 1 - x./(m + options.eps_log),X,M); % (in tensor format)
        lscalar{p} = 0; %lower bound for m
        uscalar{p} = Inf; % upper bound for m
    elseif strcmp(Z.loss_function{p},'IS') % f(x,m)=x/m+log(m/x)-1, x>0,m>0 (Itakura-Saito)
        if ~valid_nonneg(Z.object{p})
            warning('Using ''%s'' but tensor is not positive', Z.loss_function{p});
        end
        log_ten = (tenfun(@(x) -reallog(x+options.eps_log),Z.object{p})); % add epsilon to avoid problems at x=0
        Znorm_const{p} = collapse(minus(log_ten,tensor(ones(size(Z.object{p})))));
        fh{p} = @(X,M) tenfun(@(x,m) x./(m + options.eps_log) + log(m + options.eps_log),X,M); % X:data, M:model(in tensor format)
        gh{p} = @(X,M) tenfun(@(x,m) - x./((m + options.eps_log).^2) + 1./(m + options.eps_log),X,M); % (in tensor format)
        lscalar{p} = 0; %lower bound for m
        uscalar{p} = Inf; % upper bound for m
    elseif strcmp(Z.loss_function{p},'beta') % f(x,m)=1/beta *m^beta -1/(beta-1)*x*m^(beta-1) +1/(beta(beta-1))*x^beta (Beta divergence, beta!=0,1)
        beta = Z.loss_function_param{p};
        Znorm_const{p} = collapse(Z.object{p}.^beta)*1/(beta*(beta-1));
        fh{p} = @(X,M) tenfun(@(x,m) 1/beta.*m.^beta -1/(beta-1).*x.*m.^(beta-1),X,M); % X:data, M:model(in tensor format)
        gh{p} = @(X,M) tenfun(@(x,m) m.^(beta-1) -x.*m.^(beta-2),X,M); % (in tensor format)
        lscalar{p} = 0; %lower bound for m
        uscalar{p} = Inf; % upper bound for m
    end
end
%% Run algorithm

[Fac,out] = cmtf_fun_AOADMM(Z,Znorm_const, G,fh,gh,lscalar,uscalar,options);

%% Compute factors 

Zhat = cell(P,1);
for p=1:P
    if strcmp(Z.model{p},'CP')
        Zhat{p} = ktensor(Fac.fac(Z.modes{p}));
        Zhat{p} = normalize(Zhat{p});
    elseif strcmp(Z.model{p},'PAR2')
        Zhat{p}.A = Fac.fac{Z.modes{p}(1)};
        Zhat{p}.Bk = Fac.fac{Z.modes{p}(2)};
        Zhat{p}.C = Fac.fac{Z.modes{p}(3)};
        % normalize columns of A and B and put norms into C
        for r=1:size(Zhat{p}.A,2)
            normAr = norm(Zhat{p}.A(:,r),2);
            Zhat{p}.A(:,r) = Zhat{p}.A(:,r)/normAr;
            Zhat{p}.C(:,r) = Zhat{p}.C(:,r).*normAr;
            for k=1:length(Zhat{p}.Bk)
                normBrk = norm(Zhat{p}.Bk{k}(:,r),2);
                Zhat{p}.Bk{k}(:,r) = Zhat{p}.Bk{k}(:,r)/normBrk;
                Zhat{p}.C(k,r) = Zhat{p}.C(k,r).*normBrk;   
            end
        end
    end
end

%% nested functions
function tf = valid_nonneg(X)
    if isa(X,'sptensor')
        tf = all(X.vals > 0);
    else
        tf = all(X(:) > 0);
    end


function tf = valid_binary(X)
    if isa(X,'sptensor')
        tf = all(X.vals == 1);
    else
        tf = isequal(unique(X(:)),[0;1]);
    end


function tf = valid_natural(X)
    if isa(X, 'sptensor')
        vals = X.vals;
    else
        vals = X(:);
    end
    tf = all(vals >= 0) && all(vals == round(vals));




