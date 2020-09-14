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

%%
P = numel(Z.object);
Znorm_const = cell(P,1);
fh = cell(P,1); % function handles for lossfunction 
gh = cell(P,1); % function handles for gradient 
lscalar = cell(P,1); %lower bound for m
uscalar = cell(P,1); % upper bound for m
for p = 1:P
    if strcmp(Z.loss_function{p},'Frobenius')
        if isa(Z.object{p},'tensor') || isa(Z.object{p},'sptensor')
            Znorm_const{p} = norm(Z.object{p})^2;
        else
            Znorm_const{p} = norm(Z.object{p},'fro')^2;
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

[Fac,out] = cmtf_fun_AOADMM(Z,Znorm_const, G,fh,gh,lscalar,uscalar,options);

%% Compute factors 

Zhat = cell(P,1);
for p=1:P
    Zhat{p} = ktensor(Fac.fac(Z.modes{p}));
    Zhat{p} = normalize(Zhat{p});
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




