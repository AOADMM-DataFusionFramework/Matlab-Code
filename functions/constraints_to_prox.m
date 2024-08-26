function [prox_operators,reg_func] = constraints_to_prox(constrained_modes,constraints,sz)
% This function converts the given constraints in the cell constraints into
% proximal operators and corresponding regularization functions that are
% used in the algorithm

    prox_operators = cell(length(constrained_modes),1);
    reg_func = cell(length(constrained_modes),1);
    for m=1:length(constrained_modes)
        if constrained_modes(m)
            if isempty(constraints{m})
               error('No constraint provided for mode %s.',num2str(m));
            end
            if strcmp(constraints{m}{1},'non-negativity')
                prox_operators{m} = @(x,rho) project_box(x,0,inf); % non-negativity
            elseif strcmp(constraints{m}{1},'box')
                l = constraints{m}{2};
                u = constraints{m}{3};
                prox_operators{m} =  @(x,rho) project_box(x,l,u); %box constraints with lower bound l and upper bound u
            elseif strcmp(constraints{m}{1},'simplex column-wise')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) project_simplex(x, eta, 1); %simplex constraint column-wise
            elseif strcmp(constraints{m}{1},'simplex row-wise')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) project_simplex(x, eta, 2); %simplex constraint column-wise
            elseif strcmp(constraints{m}{1},'non-decreasing')
                prox_operators{m} = @(x,rho) project_monotone(x, 1); % monotonicity column-wise 
            elseif strcmp(constraints{m}{1},'non-increasing')
                prox_operators{m} = @(x,rho) -project_monotone(-x, 1); % monotonicity column-wise 
            elseif strcmp(constraints{m}{1},'unimodality')
                nn = constraints{m}{2};
                prox_operators{m} = @(x,rho) project_unimodal(x,nn); % unimodality column-wise
            elseif strcmp(constraints{m}{1},'l1-ball')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) project_L1(x, eta, 1); %(hard) l1 sparsity column-wise (||x||_1<=eta):
            elseif strcmp(constraints{m}{1},'l2-ball')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) project_L2(x, eta, 1); %(hard) l2 normlization column-wise (||x||_2<=eta):
            elseif strcmp(constraints{m}{1},'non-negative l2-ball')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) project_L2(project_box(x,0,inf),eta,1); %(hard) l2 normlization column-wise (||x||_2<=eta)
            elseif strcmp(constraints{m}{1},'non-negative l2-sphere')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) prox_normalized_nonneg(x); %(hard) l2 normlization column-wise (||x||_2<=eta) and non-negativity (not convex!)
            elseif strcmp(constraints{m}{1},'orthonormal')
                prox_operators{m} = @(x,rho) project_ortho(x); %orthonormal columns
            elseif strcmp(constraints{m}{1},'l1 regularization')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) prox_abs(x,eta/rho*ones(size(x))); % l1 sparsity regularization
                reg_func{m} = @(x) eta*sum(vecnorm(x,1));
            elseif strcmp(constraints{m}{1},'l0 regularization')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) prox_zero(x,eta/rho*ones(size(x))); % l0 sparsity regularization (not convex!)
                reg_func{m} = @(x) eta*nnz(x);
            elseif strcmp(constraints{m}{1},'l2 regularization')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) prox_L2(x, eta/rho, 1); % l2 regularization
                reg_func{m} = @(x) eta*sum(vecnorm(x,2));
            elseif strcmp(constraints{m}{1},'ridge')
                eta = constraints{m}{2};
                prox_operators{m} = @(x,rho) 1/(2*(eta/rho)+1).*x; % ridge
                reg_func{m} = @(x) eta*norm(x,'fro')^2;
            elseif strcmp(constraints{m}{1},'quadratic regularization')
                eta = constraints{m}{2};
                L = constraints{m}{3};
                I = speye(size(L));
                prox_operators{m} = @(x,rho) (2*eta/rho.*L+I)\x; 
                reg_func{m} = @(x) eta*trace(x'*L*x);
             elseif strcmp(constraints{m}{1},'GL smoothness')
                eta = constraints{m}{2};
                szm = sz{m}(1); % DOES NOT WORK FOR THE Bk MODE OF PARAFAC2 WHEN Bk's HAVE DIFFERENT SIZES!!!! TODO: WRITE OWN FUNCTION!
                L = diag(2*ones(szm,1)) + diag(-1*ones(szm-1,1),1) + diag(-1*ones(szm-1,1),-1);
                L(1,1) = 1;
                L(end,end) = 1;
                L = sparse(L);
                I = speye(size(L));
                prox_operators{m} = @(x,rho) (2*eta/rho.*L+I)\x; 
                reg_func{m} = @(x) eta*trace(x'*L*x);
             elseif strcmp(constraints{m}{1},'TV regularization')
                eta = constraints{m}{2};
                prox_operators{m} =  @(x,rho) prox_TV(x,eta/rho); 
                reg_func{m} = @(x) eta*sum(sum(x(2:end,:)-x(1:end-1,:)));
             elseif strcmp(constraints{m}{1},'custom')
                 prox_operators{m} = constraints{m}{2};
                 if length(constraints{m})>2
                     reg_func{m} = constraints{m}{3};
                 end
             end
        end
    end
end