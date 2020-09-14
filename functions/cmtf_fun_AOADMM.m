function [G,out] = cmtf_fun_AOADMM(Z,Znorm_const, G,fh,gh,lscalar,uscalar,options)
% performs AOADMM for CMTF 
    
    lbfgsb_options = options.lbfgsb_options;
    couplings = unique(Z.coupling.lin_coupled_modes);
    nb_modes  = length(Z.size);
    which_p = zeros(1,nb_modes);
    for i=1:nb_modes
        which_p(i) = find(cellfun(@(x) any(ismember(x,i)),Z.modes));
    end
    P = numel(Z.object);
    G_transp_G = cell(nb_modes,1);
    sum_column_norms_sqr = zeros(nb_modes,1);
    A = cell(nb_modes,1);
    C = cell(nb_modes,1);
    B = cell(nb_modes,1);
    B2 = cell(nb_modes,1);
    L = cell(nb_modes,1);
    rho = ones(nb_modes,1);
    last_m = zeros(P,1);
    last_mttkrp = cell(P,1); %for efficient function value computation
    last_had = cell(P,1); %for efficient function value computation
    out.innerIters = zeros(nb_modes,1);

    [f_tensors,f_couplings,f_constraints] = CMTF_AOADMM_func_eval(Znorm_const,[],[],[]);
    f_total = f_tensors+f_couplings+f_constraints;
    func_val(1) = f_tensors;
    func_coupl(1) = f_couplings;
    func_constr(1) = f_constraints;
    tstart = tic;
    time_at_it(1) = 0;
    %display first iteration
    if strcmp(options.Display,'iter') || strcmp(options.Display,'final')
        fprintf(1,' Iter  f total      f tensors      f couplings    f constraints   \n');
        fprintf(1,'------ ------------ -------------  -------------- ----------------\n');
    end

    if strcmp(options.Display,'iter')
        fprintf(1,'%6d %12f %12f %12f %12f\n', 0, f_total, f_tensors, f_couplings,f_constraints);
    end
    iter = 1;
    
    for m=1:nb_modes
        p = which_p(m);
        if strcmp(Z.loss_function{p},'Frobenius')
            G_transp_G{m} = G.fac{m}'*G.fac{m}; %precompute G'*G;
        else
            for r=1:size(G.fac{m},2)
                sum_column_norms_sqr(m,1) = sum_column_norms_sqr(m,1)+norm(G.fac{m}(:,r))^2; % need this to compute rho in the non-Frobenius case
            end
        end
    end
    
    stop = false;
    while(iter<=options.MaxOuterIters && ~stop)    
        
        for coupl_id=couplings %loop over all couplings (and non-coupled modes, if couplings=0)
            coupled_modes = find(Z.coupling.lin_coupled_modes==coupl_id); % all modes with this coupling_id
            for p=unique(which_p(coupled_modes)) % loop over all coupled tensors for this coupling_id (can be done in parallel!)
                for m=coupled_modes(which_p(coupled_modes)==p) %loop over all modes in tensor p with this coupling_id (can NOT be done in parallel)
                    if strcmp(Z.loss_function{p},'Frobenius')
                        % precomputations
                        if length(size(Z.object{p}))>=3  % Tensor
                            A{m} = Z.weights(p) *mttkrp(Z.object{p},G.fac(Z.modes{p}),find(Z.modes{p}==m)); %efficient calculation of matricized tensor with kathrirao product of all factor matrices, but the mth
                            C{m} = ones(size(G_transp_G{m}));
                            for j=Z.modes{p} 
                                if(j~=m)
                                    C{m} = C{m} .* G_transp_G{j}; % efficient calculation of the product of khatrirao products
                                end
                            end

                        else % Matrix
                            matrix_mode = find(Z.modes{p}==m);
                            if matrix_mode == 1 %first mode in matrix
                                A{m} = Z.weights(p)* double(Z.object{p})*G.fac{Z.modes{p}(2)};
                                C{m} = G_transp_G{Z.modes{p}(2)};
                            else %second mode in matrix
                                A{m} = Z.weights(p)* double(Z.object{p})'*G.fac{Z.modes{p}(1)}; %transposed M!
                                C{m} = G_transp_G{Z.modes{p}(1)};
                            end
                        end
                        rho(m) = trace(C{m})/size(C{m},1);
                        B{m} = Z.weights(p)* C{m}; 

                        last_mttkrp{p} = A{m}*1/Z.weights(p);
                        last_had{p} = C{m};
                        last_m(p) = m;
                        if options.bsum
                            A{m} = A{m} + options.bsum_weight/2*G.fac{m};
                            B{m} = B{m} + options.bsum_weight/2*eye(size(B{m}));
                        end
                    else % other loss than Frobenius
                        rho(m) = sum(sum_column_norms_sqr([1:m-1,m+1:end])); 
                    end
                    if coupl_id==0 %modes are not coupled
                        if (Z.constrained_modes(m)==0) % mode is not constrained
                            if strcmp(Z.loss_function{p},'Frobenius')
                                G.fac{m} = A{m}/B{m}; % Update factor matrices (Least squares update)
                            else
                                [lbfgsb_iterations] = lbfgsb_update(p,m,false,-1,rho(m)); %updates G.fac{m} with lbfgsb
                            end
                            inner_iters = 1;
                        else % mode is constrained, use ADMM
                            if strcmp(Z.loss_function{p},'Frobenius')
                                B{m} = B{m}+ rho(m)/2*eye(size(B{m})); % for constraint
                                L{m} = chol(B{m}','lower'); %precompute Cholesky decomposition of B (only works in the chase when rho does not change between inner iterations)
                            end
                            [inner_iters,lbfgsb_iterations] = ADMM_constrained_only(A{m},L{m},m,p,rho,options);
                        end
                        out.innerIters(m,iter)= inner_iters;
                        if strcmp(Z.loss_function{p},'Frobenius')
                            G_transp_G{m} = G.fac{m}'*G.fac{m}; % update G transposed G for mth mode
                        else
                            out.lbfgsb_iterations{m,iter} = lbfgsb_iterations;
                            for r=1:size(G.fac{m},2)
                                sum_column_norms_sqr(m,1) = norm(G.fac{m}(:,r))^2;
                            end
                        end
                    end
                end
            end

            if coupl_id~=0 %modes are coupled: use "coupled ADMM"
                ctype = Z.coupling.coupling_type(coupl_id); %type of linear coupling
                switch ctype
                    case 0 %exact coupling
                        for m=coupled_modes
                            p = which_p(m);
                            if strcmp(Z.loss_function{p},'Frobenius')
                                B{m} = B{m} + rho(m)/2* eye(size(B{m})); % for the coupling
                                if Z.constrained_modes(m) %mode is constrained
                                    B{m} = B{m} + rho(m)/2*eye(size(B{m}));
                                end
                                L{m} = chol(B{m}','lower'); %precompute Cholesky decomposition of B (only works in the chase when rho does not change between inner iterations)
                            end
                        end
                        [inner_iters,lbfgsb_iterations] = ADMM_coupled_case0(A,L,coupled_modes,coupl_id,rho,options);
                    case 1 % mode is linear coupled with trafo matrix from left, use ADMM with silvester equation
                        for m=coupled_modes
                            p = which_p(m);
                            if strcmp(Z.loss_function{p},'Frobenius')
                                B2{m} = rho(m)/2* Z.coupling.coupl_trafo_matrices{m}'*Z.coupling.coupl_trafo_matrices{m}; % precompute????
                                if Z.constrained_modes(m) %mode is constrained 
                                    B2{m} = B2{m} + rho(m)/2*eye(size(B2{m}));
                                end
                            end
                        end
                        [inner_iters,lbfgsb_iterations] = ADMM_coupled_case1(A,B,B2,coupled_modes,coupl_id,rho,options);
                    case 2
                        for m=coupled_modes
                            p = which_p(m);
                            if strcmp(Z.loss_function{p},'Frobenius')
                                B{m} = B{m} + rho(m)/2* Z.coupling.coupl_trafo_matrices{m}*Z.coupling.coupl_trafo_matrices{m}'; % precompute????; % for the coupling
                                if Z.constrained_modes(m) %mode is constrained
                                    B{m} = B{m} + rho(m)/2*eye(size(B{m}));
                                end
                                L{m} = chol(B{m}','lower'); %precompute Cholesky decomposition of B (only works in the chase when rho does not change between inner iterations)
                            end
                        end
                        [inner_iters,lbfgsb_iterations] = ADMM_coupled_case2(A,L,coupled_modes,coupl_id,rho,options);
                    case 3
                        for m=coupled_modes
                            p = which_p(m);
                            if strcmp(Z.loss_function{p},'Frobenius')
                                B{m} = B{m} + rho(m)/2* eye(size(B{m})); % for the coupling
                                if Z.constrained_modes(m) %mode is constrained
                                    B{m} = B{m} + rho(m)/2*eye(size(B{m}));
                                end
                                L{m} = chol(B{m}','lower'); %precompute Cholesky decomposition of B (only works in the chase when rho does not change between inner iterations)
                            end
                        end
                        [inner_iters,lbfgsb_iterations] = ADMM_coupled_case3(A,L,coupled_modes,coupl_id,rho,options);
                    case 4
                        for m=coupled_modes
                            p = which_p(m);
                            if strcmp(Z.loss_function{p},'Frobenius')
                                B{m} = B{m} + rho(m)/2* eye(size(B{m})); % for the coupling
                                if Z.constrained_modes(m) %mode is constrained
                                    B{m} = B{m} + rho(m)/2*eye(size(B{m}));
                                end
                                L{m} = chol(B{m}','lower'); %precompute Cholesky decomposition of B (only works in the chase when rho does not change between inner iterations)
                            end
                        end
                        [inner_iters,lbfgsb_iterations] = ADMM_coupled_case4(A,L,coupled_modes,coupl_id,rho,options);
                end
                out.innerIters(coupled_modes,iter)= inner_iters;
                for m=coupled_modes
                    p = which_p(m);
                    if strcmp(Z.loss_function{p},'Frobenius')
                        G_transp_G{m} = G.fac{m}'*G.fac{m}; % update G transposed G for mth mode
                    else
                        out.lbfgsb_iterations{m,iter} = lbfgsb_iterations{m};
                        for r=1:size(G.fac{m},2)
                            sum_column_norms_sqr(m,1) = norm(G.fac{m}(:,r))^2;
                        end
                    end
                end
            end
        end
        
        
       f_tensors_old = f_tensors;
       f_couplings_old = f_couplings;
       f_constraints_old = f_constraints;
       [f_tensors,f_couplings,f_constraints] = CMTF_AOADMM_func_eval(Znorm_const,last_mttkrp,last_had,last_m);
       f_total = f_tensors+f_couplings+f_constraints;
       func_val(iter+1) = f_tensors;
       func_coupl(iter+1) = f_couplings;
       func_constr(iter+1) = f_constraints;
       time_at_it(iter+1) = toc(tstart);
       stop = evaluate_stopping_conditions(f_tensors,f_couplings,f_constraints,f_tensors_old,f_couplings_old,f_constraints_old,options);

        %display
        if strcmp(options.Display,'iter') && mod(iter,options.DisplayIters)==0
            fprintf(1,'%6d %12f %12f %12f %12f\n', iter, f_total, f_tensors, f_couplings,f_constraints);
        end
        iter = iter+1;
    end
    % which condition caused stop?
    exit_flag = make_exit_flag(iter,f_tensors,f_couplings,f_constraints,options);
    %save output
    out.f_tensors = f_tensors;
    out.f_couplings = f_couplings;
    out.f_constraints = f_constraints;
    out.exit_flag = exit_flag;
    out.OuterIterations = iter-1;
    out.func_val_conv = func_val;
    out.func_coupl_conv = func_coupl;
    out.func_constr_conv = func_constr;
    out.time_at_it = time_at_it;

    %display final
    if strcmp(options.Display,'iter') || strcmp(options.Display,'final')
        fprintf(1,'%6d %12f %12f %12f %12f\n', iter-1, f_total, f_tensors, f_couplings,f_constraints);
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% NESTED FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    function [inner_iter,lbfgsb_iterations] = ADMM_constrained_only(A,L,m,p,rho,options)
    %ADMM loop for mode m, where mode m is constrained, but not coupled!
    % changes the global variable G (only fields related to mode m)

        inner_iter = 1;
        rel_primal_res_constr = inf;
        rel_dual_res_constr = inf;
        oldZ = cell(nb_modes,1);
        % ADMM loop
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_constr>options.innerRelDualTol_constr))
            if strcmp(Z.loss_function{p},'Frobenius')
                A_inner = A + rho(m)/2*(G.constraint_fac{m} - G.constraint_dual_fac{m});
                G.fac{m} = (A_inner/L')/L; % forward-backward substitution
                lbfgsb_iterations{m} = [];
            else % other loss function, use lbfgsb
                [lbfgsb_iterations{m}(inner_iter)] = lbfgsb_update(p,m,true,-1,rho(m)); %updates G.fac{m}
            end

            % Update constraint factor (Z) and its dual (mu_Z)
            oldZ{m} = update_constraint(m,rho(m)); %updates G.constraint_fac{mm} and G.constraint_dual_fac{mm}

            inner_iter = inner_iter + 1; 
            [rel_primal_res_constr,rel_dual_res_constr] = eval_res_ADMM_constr(m,rho,oldZ); 
        end
        inner_iter = inner_iter-1;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [inner_iter,lbfgsb_iterations] = ADMM_coupled_case0(A,L,coupled_modes,coupl_id,rho,options)
        inner_iter = 1;
        rel_primal_res_coupling = inf;
        rel_primal_res_constr = inf;
        rel_dual_res_coupling = inf;
        rel_dual_res_constr = inf;
        A_inner = cell(nb_modes,1);
        oldZ = cell(nb_modes,1);
        lbfgsb_iterations = cell(nb_modes,1);
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_coupling>options.innerRelPrTol_coupl||rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_coupling>options.innerRelDualTol_coupl||rel_dual_res_constr>options.innerRelDualTol_constr))
            %exact coupling
            for mm=coupled_modes %update all factor matrices (can be done in parallel!)
                pp = which_p(mm);
                if strcmp(Z.loss_function{pp},'Frobenius')
                    A_inner{mm} = A{mm} + rho(mm)/2*( G.coupling_fac{Z.coupling.lin_coupled_modes(mm)} - G.coupling_dual_fac{mm});
                    if Z.constrained_modes(mm) %in case the mode is also constrained
                         A_inner{mm} = A_inner{mm} + rho(mm)/2*(G.constraint_fac{mm} - G.constraint_dual_fac{mm});
                    end
                    G.fac{mm} = (A_inner{mm}/L{mm}')/L{mm}; % forward-backward substitution
                    lbfgsb_iterations{m} = [];
                else
                    [lbfgsb_iters(inner_iter)] = lbfgsb_update(pp,mm,Z.constrained_modes(mm),0,rho(mm)); %updates G.fac{m}
                    lbfgsb_iterations{mm} = lbfgsb_iters;
                end
            end
            
            % Update coupling factor (Delta) 
            oldDelta = G.coupling_fac{coupl_id};
            G.coupling_fac{coupl_id} = zeros(size(G.coupling_fac{coupl_id})); 
            for jj = coupled_modes
                G.coupling_fac{coupl_id} = G.coupling_fac{coupl_id} + G.fac{jj} + G.coupling_dual_fac{jj};
            end
            G.coupling_fac{coupl_id} = 1/length(coupled_modes).*G.coupling_fac{coupl_id};
            
            % Update constraint factor (Z) and its dual (mu_Z) and mu_Delta
            for mm=coupled_modes % (can be done in parallel!)
                G.coupling_dual_fac{mm} = G.coupling_dual_fac{mm} + G.fac{mm} - G.coupling_fac{Z.coupling.lin_coupled_modes(mm)}; % Update (mu_Delta)
                if Z.constrained_modes(mm)  
                    oldZ{mm} = update_constraint(mm,rho(mm)); %updates G.constraint_fac{mm} and G.constraint_dual_fac{mm}
                end
            end
            inner_iter = inner_iter + 1; 
            [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case0(coupled_modes,coupl_id,rho,oldDelta);
            constrained_modes = coupled_modes(logical(Z.constrained_modes(coupled_modes)));
            if ~isempty(constrained_modes)
                [rel_primal_res_constr,rel_dual_res_constr] = eval_res_ADMM_constr(constrained_modes,rho,oldZ); % does this work? integrate in loop instead? (no nested function)
            else
               rel_primal_res_constr = 0;
               rel_dual_res_constr =0;
            end
        end
        inner_iter = inner_iter-1;
    end  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [inner_iter,lbfgsb_iterations] = ADMM_coupled_case1(A,B,B2,coupled_modes,coupl_id,rho,options)
        inner_iter = 1;
        rel_primal_res_coupling = inf;
        rel_primal_res_constr = inf;
        rel_dual_res_coupling = inf;
        rel_dual_res_constr = inf;
        A_inner = cell(nb_modes,1);
        oldZ = cell(nb_modes,1);
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_coupling>options.innerRelPrTol_coupl||rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_coupling>options.innerRelDualTol_coupl||rel_dual_res_constr>options.innerRelDualTol_constr))
            %exact coupling
            for mm=coupled_modes %update all factor matrices (can be done in parallel!)
                pp = which_p(mm);
                if strcmp(Z.loss_function{pp},'Frobenius')
                    A_inner{mm} = A{mm} + rho(mm)/2*Z.coupling.coupl_trafo_matrices{mm}'*( G.coupling_fac{Z.coupling.lin_coupled_modes(mm)} - G.coupling_dual_fac{mm});
                    if Z.constrained_modes(mm) %in case the mode is also constrained
                        A_inner{mm} = A_inner{mm} + rho(mm)/2*(G.constraint_fac{mm} - G.constraint_dual_fac{mm});
                    end
                    G.fac{mm} = sylvester(B2{mm},B{mm},A_inner{mm}); % solve Sylvester equation
                    lbfgsb_iterations{m} = [];
                else
                    [lbfgsb_iters(inner_iter)] = lbfgsb_update(pp,mm,Z.constrained_modes(mm),1,rho(mm)); %updates G.fac{m} with lbfgsb
                    lbfgsb_iterations{mm} = lbfgsb_iters;
                end
            end
            
            % Update coupling factor (Delta) 
            oldDelta = G.coupling_fac{coupl_id};
            G.coupling_fac{coupl_id} = zeros(size(G.coupling_fac{coupl_id})); 
            for jj = coupled_modes
                G.coupling_fac{coupl_id} = G.coupling_fac{coupl_id} + Z.coupling.coupl_trafo_matrices{jj}*G.fac{jj} + G.coupling_dual_fac{jj};
            end
            G.coupling_fac{coupl_id} = 1/length(coupled_modes).*G.coupling_fac{coupl_id};
            
            % Update constraint factor (Z) and its dual (mu_Z) and mu_Delta
            for mm=coupled_modes % (can be done in parallel!)
                G.coupling_dual_fac{mm} = G.coupling_dual_fac{mm} + Z.coupling.coupl_trafo_matrices{mm}*G.fac{mm} - G.coupling_fac{Z.coupling.lin_coupled_modes(mm)}; % Update (mu_Delta)
                if Z.constrained_modes(mm)  
                    oldZ{mm} = update_constraint(mm,rho(mm)); %updates G.constraint_fac{mm} and G.constraint_dual_fac{mm}
                end
            end
            inner_iter = inner_iter + 1; 
            [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case1(coupled_modes,coupl_id,rho,oldDelta);
            constrained_modes = coupled_modes(logical(Z.constrained_modes(coupled_modes)));
            if ~isempty(constrained_modes)
                [rel_primal_res_constr,rel_dual_res_constr] = eval_res_ADMM_constr(constrained_modes,rho,oldZ); % does this work? integrate in loop instead? (no nested function)
            else
               rel_primal_res_constr = 0;
               rel_dual_res_constr =0;
            end
        end
        inner_iter = inner_iter-1;
    end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [inner_iter,lbfgsb_iterations] = ADMM_coupled_case2(A,L,coupled_modes,coupl_id,rho,options)
        inner_iter = 1;
        rel_primal_res_coupling = inf;
        rel_primal_res_constr = inf;
        rel_dual_res_coupling = inf;
        rel_dual_res_constr = inf;
        A_inner = cell(nb_modes,1);
        oldZ = cell(nb_modes,1);
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_coupling>options.innerRelPrTol_coupl||rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_coupling>options.innerRelDualTol_coupl||rel_dual_res_constr>options.innerRelDualTol_constr))
            %exact coupling
            for mm=coupled_modes %update all factor matrices (can be done in parallel!)
                pp = which_p(mm);
                if strcmp(Z.loss_function{pp},'Frobenius')
                    A_inner{mm} = A{mm} + rho(mm)/2*(G.coupling_fac{Z.coupling.lin_coupled_modes(mm)} - G.coupling_dual_fac{mm})*Z.coupling.coupl_trafo_matrices{mm}';
                    if Z.constrained_modes(mm) %in case the mode is also constrained
                        A_inner{mm} = A_inner{mm} + rho(mm)/2*(G.constraint_fac{mm} - G.constraint_dual_fac{mm});
                    end
                    G.fac{mm} = (A_inner{mm}/L{mm}')/L{mm}; % forward-backward substitution
                    lbfgsb_iterations{m} = [];
                else
                   [lbfgsb_iters(inner_iter)] = lbfgsb_update(pp,mm,Z.constrained_modes(mm),2,rho(mm)); %updates G.fac{m} with lbfgsb
                    lbfgsb_iterations{mm} = lbfgsb_iters;
                end
            end
            
            % Update coupling factor (Delta) 
            oldDelta = G.coupling_fac{coupl_id};
            G.coupling_fac{coupl_id} = zeros(size(G.coupling_fac{coupl_id})); 
            for jj = coupled_modes
                G.coupling_fac{coupl_id} = G.coupling_fac{coupl_id} + G.fac{jj}*Z.coupling.coupl_trafo_matrices{jj} + G.coupling_dual_fac{jj};
            end
            G.coupling_fac{coupl_id} = 1/length(coupled_modes).*G.coupling_fac{coupl_id};
            
            % Update constraint factor (Z) and its dual (mu_Z) and mu_Delta
            for mm=coupled_modes % (can be done in parallel!)
                G.coupling_dual_fac{mm} = G.coupling_dual_fac{mm} + G.fac{mm}*Z.coupling.coupl_trafo_matrices{mm}  - G.coupling_fac{Z.coupling.lin_coupled_modes(mm)}; % Update (mu_Delta)
                if Z.constrained_modes(mm)  
                    oldZ{mm} = update_constraint(mm,rho(mm)); %updates G.constraint_fac{mm} and G.constraint_dual_fac{mm}
                end
            end
            inner_iter = inner_iter + 1; 
            [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case2(coupled_modes,coupl_id,rho,oldDelta);
            constrained_modes = coupled_modes(logical(Z.constrained_modes(coupled_modes)));
            if ~isempty(constrained_modes)
                [rel_primal_res_constr,rel_dual_res_constr] = eval_res_ADMM_constr(constrained_modes,rho,oldZ); % does this work? integrate in loop instead? (no nested function)
            else
               rel_primal_res_constr = 0;
               rel_dual_res_constr =0;
            end
        end
        inner_iter = inner_iter-1;
    end  
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [inner_iter,lbfgsb_iterations] = ADMM_coupled_case3(A,L,coupled_modes,coupl_id,rho,options)
        inner_iter = 1;
        rel_primal_res_coupling = inf;
        rel_primal_res_constr = inf;
        rel_dual_res_coupling = inf;
        rel_dual_res_constr = inf;
        A_inner = cell(nb_modes,1);
        oldZ = cell(nb_modes,1);
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_coupling>options.innerRelPrTol_coupl||rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_coupling>options.innerRelDualTol_coupl||rel_dual_res_constr>options.innerRelDualTol_constr))
            %exact coupling
            for mm=coupled_modes %update all factor matrices (can be done in parallel!)
                pp = which_p(mm);
                if strcmp(Z.loss_function{pp},'Frobenius')
                    A_inner{mm} = A{mm} + rho(mm)/2*(Z.coupling.coupl_trafo_matrices{mm}* G.coupling_fac{Z.coupling.lin_coupled_modes(mm)} - G.coupling_dual_fac{mm});
                    if Z.constrained_modes(mm) %in case the mode is also constrained
                        A_inner{mm} = A_inner{mm} + rho(mm)/2*(G.constraint_fac{mm} - G.constraint_dual_fac{mm});
                    end
                    G.fac{mm} = (A_inner{mm}/L{mm}')/L{mm}; % forward-backward substitution
                    lbfgsb_iterations{m} = [];
                else
                    [lbfgsb_iters(inner_iter)] = lbfgsb_update(pp,mm,Z.constrained_modes(mm),3,rho(mm)); %updates G.fac{m} with lbfgsb
                    lbfgsb_iterations{mm} = lbfgsb_iters;
                end
            end
            
            % Update coupling factor (Delta) 
            oldDelta = G.coupling_fac{coupl_id};
            AA = zeros(size(Z.coupling.coupl_trafo_matrices{coupled_modes(1)},2));
            BB = zeros(size(Z.coupling.coupl_trafo_matrices{coupled_modes(1)},2),size(G.fac{coupled_modes(1)},2));
            for jj = coupled_modes
                AA = AA + Z.coupling.coupl_trafo_matrices{jj}'*Z.coupling.coupl_trafo_matrices{jj};
                BB = BB + Z.coupling.coupl_trafo_matrices{jj}'*(G.fac{jj} + G.coupling_dual_fac{jj});
            end
            G.coupling_fac{coupl_id} = AA\BB;
            
            % Update constraint factor (Z) and its dual (mu_Z) and mu_Delta
            for mm=coupled_modes % (can be done in parallel!)
                G.coupling_dual_fac{mm} = G.coupling_dual_fac{mm} + G.fac{mm} - Z.coupling.coupl_trafo_matrices{mm}*G.coupling_fac{Z.coupling.lin_coupled_modes(mm)}; % Update (mu_Delta)
                if Z.constrained_modes(mm)  
                    oldZ{mm} = update_constraint(mm,rho(mm)); %updates G.constraint_fac{mm} and G.constraint_dual_fac{mm}
                end
            end
            inner_iter = inner_iter + 1; 
            [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case3(coupled_modes,coupl_id,rho,oldDelta);
            constrained_modes = coupled_modes(logical(Z.constrained_modes(coupled_modes)));
            if ~isempty(constrained_modes)
                [rel_primal_res_constr,rel_dual_res_constr] = eval_res_ADMM_constr(constrained_modes,rho,oldZ); % does this work? integrate in loop instead? (no nested function)
            else
               rel_primal_res_constr = 0;
               rel_dual_res_constr =0;
            end
        end
        inner_iter = inner_iter-1;
    end  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [inner_iter,lbfgsb_iterations] = ADMM_coupled_case4(A,L,coupled_modes,coupl_id,rho,options)
        inner_iter = 1;
        rel_primal_res_coupling = inf;
        rel_primal_res_constr = inf;
        rel_dual_res_coupling = inf;
        rel_dual_res_constr = inf;
        A_inner = cell(nb_modes,1);
        oldZ = cell(nb_modes,1);
        while (inner_iter<=options.MaxInnerIters &&(rel_primal_res_coupling>options.innerRelPrTol_coupl||rel_primal_res_constr>options.innerRelPrTol_constr||rel_dual_res_coupling>options.innerRelDualTol_coupl||rel_dual_res_constr>options.innerRelDualTol_constr))
            %exact coupling
            for mm=coupled_modes %update all factor matrices (can be done in parallel!)
                pp = which_p(mm);
                if strcmp(Z.loss_function{pp},'Frobenius')
                    A_inner{mm} = A{mm} + rho(mm)/2*(G.coupling_fac{Z.coupling.lin_coupled_modes(mm)}*Z.coupling.coupl_trafo_matrices{mm} - G.coupling_dual_fac{mm});
                    if Z.constrained_modes(mm) %in case the mode is also constrained
                        A_inner{mm} = A_inner{mm} + rho(mm)/2*(G.constraint_fac{mm} - G.constraint_dual_fac{mm});
                    end
                    G.fac{mm} = (A_inner{mm}/L{mm}')/L{mm}; % forward-backward substitution
                    lbfgsb_iterations{m} = [];
                else
                    [lbfgsb_iters(inner_iter)] = lbfgsb_update(pp,mm,Z.constrained_modes(mm),4,rho(mm)); %updates G.fac{m} with lbfgsb
                    lbfgsb_iterations{mm} = lbfgsb_iters;
                end
            end
            
            % Update coupling factor (Delta) 
            oldDelta = G.coupling_fac{coupl_id};
            AA = zeros(size(Z.coupling.coupl_trafo_matrices{coupled_modes(1)},1));
            BB = zeros(size(G.fac{coupled_modes(1)},1),size(Z.coupling.coupl_trafo_matrices{coupled_modes(1)},1));
            for jj = coupled_modes
                AA = AA + Z.coupling.coupl_trafo_matrices{jj}*Z.coupling.coupl_trafo_matrices{jj}';
                BB = BB + (G.fac{jj} + G.coupling_dual_fac{jj})*Z.coupling.coupl_trafo_matrices{jj}';
            end
            G.coupling_fac{coupl_id} = BB/AA;
            
            % Update constraint factor (Z) and its dual (mu_Z) and mu_Delta
            for mm=coupled_modes % (can be done in parallel!)
                G.coupling_dual_fac{mm} = G.coupling_dual_fac{mm} + G.fac{mm} - G.coupling_fac{Z.coupling.lin_coupled_modes(mm)}*Z.coupling.coupl_trafo_matrices{mm}; % Update (mu_Delta)
                if Z.constrained_modes(mm) 
                    oldZ{mm} = update_constraint(mm,rho(mm)); %updates G.constraint_fac{mm} and G.constraint_dual_fac{mm}
                end
            end
            inner_iter = inner_iter + 1; 
            [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case4(coupled_modes,coupl_id,rho,oldDelta);
            constrained_modes = coupled_modes(logical(Z.constrained_modes(coupled_modes)));
            if ~isempty(constrained_modes)
                [rel_primal_res_constr,rel_dual_res_constr] = eval_res_ADMM_constr(constrained_modes,rho,oldZ); % does this work? integrate in loop instead? (no nested function)
            else
               rel_primal_res_constr = 0;
               rel_dual_res_constr =0;
            end
        end
        inner_iter = inner_iter-1;
    end  
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [rel_primal_res_constr,rel_dual_res_constr] = eval_res_ADMM_constr(modes,rho,oldZ)
    % computes relative primal and dual residuals of ADMM iteration for factor
    % matrix m connected to the constraint only!
        
        rel_primal_res_constr = 0;
        rel_dual_res_constr = 0;
        scaling1 = 0;
        scaling3 = 0;
        for mm=modes
            rel_primal_res_constr = rel_primal_res_constr + norm(G.fac{mm}-G.constraint_fac{mm},'fro');
            scaling1 = scaling1 + norm(G.fac{mm},'fro');
            rel_dual_res_constr = rel_dual_res_constr + norm(G.constraint_fac{mm}-oldZ{mm},'fro');
            scaling3 = scaling3 + norm(G.constraint_dual_fac{mm},'fro');
        end
        rel_primal_res_constr = rel_primal_res_constr/scaling1;
        if scaling3>0
            rel_dual_res_constr = rel_dual_res_constr/scaling3;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    function [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case0(modes,coupl_id,rho,oldDelta)
    % computes relative primal and dual residuals of ADMM iteration for factor
    % matrices in in coupled modes connected (coupling case 0 only!)
        rel_primal_res_coupling = 0;
        rel_dual_res_coupling = 0;
        scaling1 = 0;
        scaling3 = 0;
        for mm=modes
            rel_primal_res_coupling = rel_primal_res_coupling + norm(G.fac{mm}-G.coupling_fac{coupl_id},'fro');
            scaling1 = scaling1 + norm(G.fac{mm},'fro');
            rel_dual_res_coupling = rel_dual_res_coupling + norm(G.coupling_fac{coupl_id}-oldDelta,'fro');
            scaling3 = scaling3 + norm(G.coupling_dual_fac{mm},'fro');
        end
        rel_primal_res_coupling = rel_primal_res_coupling/scaling1;
        if scaling3>0
            rel_dual_res_coupling = rel_dual_res_coupling/scaling3;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    function [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case1(modes,coupl_id,rho,oldDelta)
    % computes relative primal and dual residuals of ADMM iteration for factor
    % matrices in in coupled modes connected (coupling case 1 only!)
        rel_primal_res_coupling = 0;
        rel_dual_res_coupling = 0;
        scaling1 = 0;
        scaling3 = 0;
        for mm=modes
            rel_primal_res_coupling = rel_primal_res_coupling + norm(Z.coupling.coupl_trafo_matrices{mm}*G.fac{mm}-G.coupling_fac{coupl_id},'fro');
            scaling1 = scaling1 + norm(Z.coupling.coupl_trafo_matrices{mm}*G.fac{mm},'fro');
            rel_dual_res_coupling = rel_dual_res_coupling + norm(G.coupling_fac{coupl_id}-oldDelta,'fro');
            scaling3 = scaling3 + norm(G.coupling_dual_fac{mm},'fro');
        end
        rel_primal_res_coupling = rel_primal_res_coupling/scaling1;
        if scaling3>0
            rel_dual_res_coupling = rel_dual_res_coupling/scaling3;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case2(modes,coupl_id,rho,oldDelta)
    % computes relative primal and dual residuals of ADMM iteration for factor
    % matrices in in coupled modes connected (coupling case 2 only!)
        rel_primal_res_coupling = 0;
        rel_dual_res_coupling = 0;
        scaling1 = 0;
        scaling3 = 0;
        for mm=modes
            rel_primal_res_coupling = rel_primal_res_coupling + norm(G.fac{mm}*Z.coupling.coupl_trafo_matrices{mm}-G.coupling_fac{coupl_id},'fro');
            scaling1 = scaling1 + norm(G.fac{mm}*Z.coupling.coupl_trafo_matrices{mm},'fro');
            rel_dual_res_coupling = rel_dual_res_coupling + norm((G.coupling_fac{coupl_id}-oldDelta),'fro');
            scaling3 = scaling3 + norm(G.coupling_dual_fac{mm},'fro');
        end
        rel_primal_res_coupling = rel_primal_res_coupling/scaling1;
        if scaling3>0
            rel_dual_res_coupling = rel_dual_res_coupling/scaling3;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case3(modes,coupl_id,rho,oldDelta)
    % computes relative primal and dual residuals of ADMM iteration for factor
    % matrices in in coupled modes connected (coupling case 2 only!)
        rel_primal_res_coupling = 0;
        rel_dual_res_coupling = 0;
        scaling1 = 0;
        scaling3 = 0;
        for mm=modes
            rel_primal_res_coupling = rel_primal_res_coupling + norm(G.fac{mm}-Z.coupling.coupl_trafo_matrices{mm}*G.coupling_fac{coupl_id},'fro');
            scaling1 = scaling1 + norm(G.fac{mm},'fro');
            rel_dual_res_coupling = rel_dual_res_coupling + norm(Z.coupling.coupl_trafo_matrices{mm}*(G.coupling_fac{coupl_id}-oldDelta),'fro');
            scaling3 = scaling3 + norm(G.coupling_dual_fac{mm},'fro');
        end
        rel_primal_res_coupling = rel_primal_res_coupling/scaling1;
        if scaling3>0
            rel_dual_res_coupling = rel_dual_res_coupling/scaling3;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [rel_primal_res_coupling,rel_dual_res_coupling] = eval_res_ADMM_coupl_case4(modes,coupl_id,rho,oldDelta)
    % computes relative primal and dual residuals of ADMM iteration for factor
    % matrices in in coupled modes connected (coupling case 2 only!)
        rel_primal_res_coupling = 0;
        rel_dual_res_coupling = 0;
        scaling1 = 0;
        scaling3 = 0;
        for mm=modes
            rel_primal_res_coupling = rel_primal_res_coupling + norm(G.fac{mm}-G.coupling_fac{coupl_id}*Z.coupling.coupl_trafo_matrices{mm},'fro');
            scaling1 = scaling1 + norm(G.fac{mm},'fro');
            rel_dual_res_coupling = rel_dual_res_coupling + norm((G.coupling_fac{coupl_id}-oldDelta)*Z.coupling.coupl_trafo_matrices{mm},'fro');
            scaling3 = scaling3 + norm(G.coupling_dual_fac{mm},'fro');
        end
        rel_primal_res_coupling = rel_primal_res_coupling/scaling1;
        if scaling3>0
            rel_dual_res_coupling = rel_dual_res_coupling/scaling3;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [f_tensors,f_couplings,f_constraints] = CMTF_AOADMM_func_eval( Znorm_const,last_mttkrp,last_had,last_m)
    % evaluates the 'residuals' of the objective function
    % ftensors = sum_i w_i ||T_i-[|C_i,1,C_i,2,C_i,3|]||_F^2 (or corresponding function for matrices)
    % f_couplings = sum_i ||C_i-Delta_i||_F^2
    % f_constraints = sum_i ||C_i-Z_i||_F^2


        fp = zeros(P,1);
        for pp = 1:P
            if strcmp(Z.loss_function{pp},'Frobenius')
                if length(size(Z.object{pp}))>=3
                    % Tensor 
                    if isempty(last_mttkrp)
                       fp(pp) = cp_func(Z.object{pp}, G.fac(Z.modes{pp}),  Znorm_const{pp},Z.weights(pp)); 
                    else
                        f_1 =  Znorm_const{pp};
                        V = last_mttkrp{pp}.*G.fac{last_m(pp)};
                        f_2 = sum(V(:));
                        W = last_had{pp}.*G_transp_G{last_m(pp)};
                        f_3 = sum(W(:));
                        f = f_1 - 2* f_2 + f_3;
                        fp(pp) = Z.weights(pp) *f;
                    end
                elseif length(size(Z.object{pp}))==2
                    % Matrix   
                   if isempty(last_mttkrp)
                        fp(pp) = pca_func(Z.object{pp}, G.fac(Z.modes{pp}),  Znorm_const{pp},Z.weights(pp));   
                    else
                        f_1 =  Znorm_const{pp};
                        V = last_mttkrp{pp}.*G.fac{last_m(pp)};
                        f_2 = sum(V(:));
                        W = last_had{pp}.*G_transp_G{last_m(pp)};
                        f_3 = sum(W(:));
                        f = f_1 - 2* f_2 + f_3;
                        fp(pp) = Z.weights(pp) *f;
                    end
                end
            else
                fp(pp) = Z.weights(pp)*(Znorm_const{pp} + collapse(fh{pp}(Z.object{pp},full(ktensor(G.fac(Z.modes{pp}))))));   % can we avoid this???????????????    
            end
        end 
        f_tensors = sum(fp);

        % residuals for coupling
        nb_couplings = max(Z.coupling.lin_coupled_modes);
        coupling_p = zeros(nb_couplings,1);
        for n = 1:nb_couplings
            ctype_n = Z.coupling.coupling_type(n);
            cmodes = find(Z.coupling.lin_coupled_modes==n);
            for jj = 1:length(cmodes)
                switch ctype_n
                    case 0 %exact coupling
                        coupling_p(n) = coupling_p(n) + norm(G.fac{cmodes(jj)} - G.coupling_fac{n},'fro')/norm(G.fac{cmodes(jj)},'fro'); %no rho!
                    case 1
                        coupling_p(n) = coupling_p(n) + norm(Z.coupling.coupl_trafo_matrices{cmodes(jj)}*G.fac{cmodes(jj)} - G.coupling_fac{n},'fro')/norm(Z.coupling.coupl_trafo_matrices{cmodes(jj)}*G.fac{cmodes(jj)},'fro'); %no rho!
                    case 2
                        coupling_p(n) = coupling_p(n) + norm(G.fac{cmodes(jj)}*Z.coupling.coupl_trafo_matrices{cmodes(jj)} - G.coupling_fac{n},'fro')/norm(G.fac{cmodes(jj)}*Z.coupling.coupl_trafo_matrices{cmodes(jj)},'fro'); %no rho!
                    case 3
                        coupling_p(n) = coupling_p(n) + norm(G.fac{cmodes(jj)} - Z.coupling.coupl_trafo_matrices{cmodes(jj)}*G.coupling_fac{n},'fro')/norm(G.fac{cmodes(jj)},'fro'); %no rho!
                    case 4
                        coupling_p(n) = coupling_p(n) + norm(G.fac{cmodes(jj)} - G.coupling_fac{n}*Z.coupling.coupl_trafo_matrices{cmodes(jj)},'fro')/norm(G.fac{cmodes(jj)},'fro'); %no rho!
                end
            end
        end
        f_couplings = sum(coupling_p);

        %residuals for constraints
        f_constraint_p = zeros(nb_modes,1);
        for n = 1:nb_modes 
            if ~isempty(G.constraint_fac{n})
               f_constraint_p(n) = norm(G.fac{n} - G.constraint_fac{n},'fro')/norm(G.fac{n},'fro');%no rho!
            end
        end
        f_constraints = sum(f_constraint_p);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [function_value,gradient_vector] = compute_gen_f_g(x,p,m,ff,gg,constrained,coupling_type,rho)
        % constrained: boolean
        % coupling_type: 0,...,4 or something else for not coupled
        MM = update(ktensor(G.fac(Z.modes{p})),find(Z.modes{p}==m),x);
        Mfull = full(MM);
        function_value = Z.weights(p)*collapse(ff(Z.object{p},Mfull));
        Y = gg(Z.object{p},Mfull);
        Gmatrix = mttkrp(Y,MM.U,find(Z.modes{p}==m));
        gradient_vector = Z.weights(p).*Gmatrix(:); %vectorize
        if constrained
            function_value = function_value + rho/2*sum((x-G.constraint_fac{m}(:)+G.constraint_dual_fac{m}(:)).^2);
            gradient_vector = gradient_vector + rho*(x-G.constraint_fac{m}(:)+G.constraint_dual_fac{m}(:));
        end
        n = Z.coupling.lin_coupled_modes(m);
        switch coupling_type
            case 0
                function_value = function_value + rho/2*sum((x-G.coupling_fac{n}(:)+G.coupling_dual_fac{m}(:)).^2);
                gradient_vector = gradient_vector + rho*(x-G.coupling_fac{n}(:)+G.coupling_dual_fac{m}(:));
            case 1
                function_value = function_value + rho/2*sum(sum(Z.coupling.coupl_trafo_matrices{m}*reshape(x,size(G.fac{m}))-G.coupling_fac{n}+G.coupling_dual_fac{m}).^2); 
                gradient_vector = gradient_vector + rho*reshape(Z.coupling.coupl_trafo_matrices{m}'*(Z.coupling.coupl_trafo_matrices{m}*reshape(x,size(G.fac{m}))-G.coupling_fac{n}+G.coupling_dual_fac{m}),[],1);
            case 2
                function_value = function_value + rho/2*sum(sum(reshape(x,size(G.fac{m}))*Z.coupling.coupl_trafo_matrices{m}-G.coupling_fac{n}+G.coupling_dual_fac{m}).^2);
                gradient_vector = gradient_vector + rho*reshape((reshape(x,size(G.fac{m}))*Z.coupling.coupl_trafo_matrices{m}-G.coupling_fac{n}+G.coupling_dual_fac{m})*Z.coupling.coupl_trafo_matrices{m}',[],1);
            case 3
                function_value = function_value + rho/2*sum((x-reshape(Z.coupling.coupl_trafo_matrices{m}*G.coupling_fac{n},[],1)+G.coupling_dual_fac{m}(:)).^2);
                gradient_vector = gradient_vector + rho*(x-reshape(Z.coupling.coupl_trafo_matrices{m}*G.coupling_fac{n},[],1)+G.coupling_dual_fac{m}(:));
            case 4
                function_value = function_value + rho/2*sum((x-reshape(G.coupling_fac{n}*Z.coupling.coupl_trafo_matrices{m},[],1)+G.coupling_dual_fac{m}(:)).^2);
                gradient_vector = gradient_vector + rho*(x-reshape(G.coupling_fac{n}*Z.coupling.coupl_trafo_matrices{m},[],1)+G.coupling_dual_fac{m}(:));
        end
        if options.bsum
            function_value = function_value + options.bsum_weight/2*sum((x-G.fac{m}(:)).^2);
            gradient_vector = gradient_vector + options.bsum_weight*(x-G.fac{m}(:));
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [lbfgsb_iterations] = lbfgsb_update(p,m,constrained,coupling_type,rho)
        % updates G.fac{m} using lbfgsb
        ll = lscalar{p}*ones(numel(G.fac{m}),1);
        uu = uscalar{p}*ones(numel(G.fac{m}),1);
        lbfgsb_loss_func_inner = @(x) compute_gen_f_g(x,p,m,fh{p},gh{p},constrained,coupling_type,rho);
        lbfgsb_options.x0 = G.fac{m}(:); %vectorize starting point
        [xfacm_,~,lbfgsb_info] = lbfgsb(lbfgsb_loss_func_inner,ll,uu,lbfgsb_options);
        lbfgsb_iterations = lbfgsb_info.iterations;
        G.fac{m} = reshape(xfacm_,size(G.fac{m}));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [oldZ] = update_constraint(m,rho)
        %updates G.constraint_fac{mm} and G.constraint_dual_fac{mm}
        oldZ = G.constraint_fac{m};
        G.constraint_fac{m} = feval(Z.prox_operators{m},(G.fac{m} + G.constraint_dual_fac{m}),rho);
        G.constraint_dual_fac{m} = G.constraint_dual_fac{m} + G.fac{m} - G.constraint_fac{m};
    end

end

