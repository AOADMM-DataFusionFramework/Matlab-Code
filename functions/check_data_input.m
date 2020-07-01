function [] = check_data_input(sz,modes,lambdas,coupling)
%checks if all dimensions for coupling match each other

lin_coupled_modes = coupling.lin_coupled_modes;
coupling_type = coupling.coupling_type;
coupl_trafo_matrices = coupling.coupl_trafo_matrices;

max_modeid = max(cellfun(@(x) max(x), modes));
if max_modeid ~= length(sz)
    error('Mismatch between size and modes inputs')
end

nb_couplings = max(lin_coupled_modes);
if nb_couplings ~= length(coupling_type)
    error('Mismatch between number of coulings and coupling types')
end

for p = 1:length(lambdas)
    for m = modes{p}
        if lin_coupled_modes(m) == 0 % this mode is not coupled
            if ~isempty(coupl_trafo_matrices{m})
                warning('Coupling matrix for mode %s will not be considered, because the mode is not coupled.', num2str(m))
            end
        else %this mode is coupled
            coupl_id = lin_coupled_modes(m);
            mode_ids = find(lin_coupled_modes==coupl_id);
            ctype = coupling_type(coupl_id);
            switch ctype
                case 0
                    if ~isempty(coupl_trafo_matrices{m})
                        warning('Coupling matrix for mode %s will not be considered, because the mode is exactly coupled.', num2str(m))
                    end
                    for n=mode_ids
                        if (n~=m && sz(m)~=sz(n))
                            error('Coupled factor matrices of mode %s and mode %s need to have same number of rows.', num2str(m),num2str(n))
                        end
                        if (n~=m && length(lambdas{p}) ~= length(lambdas{cellfun(@(x) any(ismember(x,n)), modes)}))
                            error('Coupled factor matrices of mode %s and mode %s need to have same number of components/columns.', num2str(m),num2str(n))
                        end
                    end
                case 1
                    if isempty(coupl_trafo_matrices{m})
                        error('Coupling matrix for mode %s is missing.', num2str(m))
                    end
                    if rank(coupl_trafo_matrices{m})<size(coupl_trafo_matrices{m},1)
                        error('Coupling matrix for mode %s is not right-invertible.', num2str(m))
                    end
                    if size(coupl_trafo_matrices{m},2) ~= sz(m) 
                        error('Mismatch between sz and number of columns of coupling matrix for mode %s.', num2str(m))
                    end
                    for n=mode_ids
                        if (n~=m && size(coupl_trafo_matrices{m},1) ~=size(coupl_trafo_matrices{n},1))
                            error('Coupling transformation matrices need to have same number of rows for mode %s and mode %s.', num2str(m),num2str(n))
                        end
                        if (n~=m && length(lambdas{p}) ~= length(lambdas{cellfun(@(x) any(ismember(x,n)), modes)}))
                            error('Coupled factor matrices of mode %s and mode %s need to have same number of components/columns.', num2str(m),num2str(n))
                        end
                    end
                case 2
                    if isempty(coupl_trafo_matrices{m})
                        error('Coupling matrix for mode %s is missing.', num2str(m))
                    end
                    if size(coupl_trafo_matrices{m},1) ~= length(lambdas{cellfun(@(x) any(ismember(x,m)), modes)}) 
                        error('Mismatch between lambdas and number of rows of coupling matrix for mode %s.', num2str(m))
                    end
                    for n=mode_ids
                        if (n~=m && sz(m)~=sz(n))
                            error('Coupled factor matrices of mode %s and mode %s need to have same number of rows.', num2str(m),num2str(n))
                        end
                        if (n~=m && size(coupl_trafo_matrices{m},2) ~=size(coupl_trafo_matrices{n},2))
                            error('Coupling transformation matrices need to have same number of columns for mode %s and mode %s.', num2str(m),num2str(n))
                        end
                    end
                    if(size(coupl_trafo_matrices{m},2)>length(lambdas{p}))
                        error('Couplig matrix for mode %s can not have more columns than rows.',num2str(m))
                    end
                case 3
                    if isempty(coupl_trafo_matrices{m})
                        error('Coupling matrix for mode %s is missing.', num2str(m))
                    end
                    if size(coupl_trafo_matrices{m},1) ~= sz(m) 
                        error('Mismatch between sz and number of rows of coupling matrix for mode %s.', num2str(m))
                    end
                    for n=mode_ids
                        if (n~=m && size(coupl_trafo_matrices{m},2) ~=size(coupl_trafo_matrices{n},2))
                            error('Coupling transformation matrices need to have same number of columns for mode %s and mode %s.', num2str(m),num2str(n))
                        end
                        if (n~=m && length(lambdas{p}) ~= length(lambdas{cellfun(@(x) any(ismember(x,n)), modes)}))
                            error('Coupled factor matrices of mode %s and mode %s need to have same number of components/columns.', num2str(m),num2str(n))
                        end
                    end

                case 4 
                    if isempty(coupl_trafo_matrices{m})
                        error('Coupling matrix for mode %s is missing.', num2str(m))
                    end
                    if size(coupl_trafo_matrices{m},2) ~= length(lambdas{cellfun(@(x) any(ismember(x,m)), modes)}) 
                        error('Mismatch between lambdas and number of columns of coupling matrix for mode %s.', num2str(m))
                    end
                    for n=mode_ids
                        if (n~=m && sz(m)~=sz(n))
                            error('Coupled factor matrices of mode %s and mode %s need to have same number of rows.', num2str(m),num2str(n))
                        end
                        if (n~=m && size(coupl_trafo_matrices{m},1) ~=size(coupl_trafo_matrices{n},1))
                            error('Coupling transformation matrices need to have same number of rows for mode %s and mode %s.', num2str(m),num2str(n))
                        end
                    end
            end
        end
    end
end



end

