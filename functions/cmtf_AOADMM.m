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
nb_modes  = length(Z.size);
which_p = zeros(1,nb_modes);
for i=1:nb_modes
    which_p(i) = find(cellfun(@(x) any(ismember(x,i)),Z.modes));
end
%% Constraints
[prox_operators,reg_func] = constraints_to_prox(Z.constrained_modes,Z.constraints,Z.size);
Z.prox_operators = prox_operators;
Z.reg_func = reg_func;
for m=1:length(Z.constraints)
    if Z.constrained_modes(m)
        if strcmp(Z.constraints{m}{1},'tPARAFAC2')
            if ~strcmp(Z.model{which_p(m)},'PAR2') || 2 ~= find(Z.modes{which_p(m)}==m)
                error('The tPARAFAC2 constraint can only be impsed on the second mode of a PARAFAC2 model')
            end
        end
    end
end

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
%% PARAFAC2 rank and dimensions
for p=1:P
    if strcmp(Z.model{p},'PAR2')
        R = size(G.fac{Z.modes{p}(1)},2);
        Bkmode = Z.modes{p}(2);
        for k=1:length(Z.size{Bkmode})
            if Z.size{Bkmode}(k)<R
                error('Number of components for PARAFAC2 is larger than size of slice %d of data tensor %d.',k,p)
            end
        end
    end
end
%% Missing data preprocessing
% If Z.miss is provided, validate masks 
has_missing = isfield(Z, 'miss') && any(~cellfun(@isempty, Z.miss));
if has_missing
    for p = 1:P
        if isempty(Z.miss{p}), continue; end
        if ~strcmp(Z.loss_function{p}, 'Frobenius')
            error('cmtf:missingData:nonFrobenius', ...
                'Missing data (Z.miss) is only supported for Frobenius loss functions.');
        end
        if strcmp(Z.model{p}, 'CP')
            if isa(Z.object{p}, 'sptensor')
                error('cmtf:missingData:sptensor', ...
                    'Missing data (Z.miss) not supported for sptensor objects. Convert to tensor first.');
            end
            if isa(Z.object{p}, 'tensor')
                sz_obj = Z.object{p}.size;
            else
                sz_obj = size(Z.object{p});
            end
            if ~isequal(sz_obj, size(Z.miss{p}))
                error('cmtf:missingData:maskSizeMismatch', ...
                    'Z.miss{%d} size does not match Z.object{%d}.', p, p);
            end
            if ~isa(Z.miss{p},'sptensor')
                try 
                    Z.miss{p} = sptensor(tensor(Z.miss{p}));
                catch
                    error('cmtf:missingData:maskTypeError',...
                        'Z.miss{%d} is not a sptensor and cannot be converted to one.',p);
                end

            end
        elseif strcmp(Z.model{p}, 'PAR2')
            K = length(Z.object{p});
            if ~iscell(Z.miss{p}) || length(Z.miss{p}) ~= K
                error('cmtf:missingData:PAR2maskNotCell', ...
                    'Z.miss{%d} must be a cell array of length %d for PAR2.', p, K);
            end
            for k = 1:K
                if ~islogical(Z.miss{p}{k})
                    if isnumeric(Z.miss{p}{k}) && all(ismember(Z.miss{p}{k}(:), [0 1]))
                        Z.miss{p}{k} = logical(Z.miss{p}{k});
                    else
                        error('cmtf:missingData:PAR2maskSliceNotLogical', ...
                            'Z.miss{%d}{%d} must be a logical or binary (0/1) array.', p, k);
                    end
                end
                if ~isequal(size(Z.object{p}{k}), size(Z.miss{p}{k}))
                    error('cmtf:missingData:PAR2maskSliceSizeMismatch', ...
                        'Z.miss{%d}{%d} size does not match Z.object{%d}{%d}.', p, k, p, k);
                end
            end
        end
    end
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
                if has_missing && ~isempty(Z.miss{p})
                    Znorm_const{p} = norm(Z.miss{p}.*Z.object{p})^2;
                else
                    Znorm_const{p} = norm(Z.object{p})^2;
                end
            else
                if has_missing
                    Znorm_const{p} = norm(Z.miss{p}.*Z.object{p},'fro')^2;
                else
                    Znorm_const{p} = norm(Z.object{p},'fro')^2;
                end
            end
        elseif strcmp(Z.model{p},'PAR2')
            Znorm_const{p} = 0;
            if has_missing  && ~isempty(Z.miss{p})
                for k=1:length(Z.object{p})
                    Znorm_const{p} = Znorm_const{p} + norm(Z.miss{p}{k}.*Z.object{p}{k},'fro')^2;
                end
            else
                for k=1:length(Z.object{p})
                    Znorm_const{p} = Znorm_const{p} + norm(Z.object{p}{k},'fro')^2;
                end
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
    elseif strcmp(Z.model{p},'PAR2')
        Zhat{p}.A = Fac.fac{Z.modes{p}(1)};
        Zhat{p}.Bk = Fac.fac{Z.modes{p}(2)};
        Zhat{p}.C = Fac.fac{Z.modes{p}(3)};
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




