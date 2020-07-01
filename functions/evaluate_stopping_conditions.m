function [stop] = evaluate_stopping_conditions(f_tensors,f_couplings,f_constraints,f_tensors_old,f_couplings_old,f_constraints_old,options)
% wether or not to stop
stop_tensors = false;
stop_couplings = false;
stop_constraints = false;

if f_tensors_old>0
    f_tensors_rel_change = abs(f_tensors_old-f_tensors)/f_tensors_old;
else
    f_tensors_rel_change = abs(f_tensors_old-f_tensors);
end
if f_tensors < options.AbsFuncTol || f_tensors_rel_change < options.OuterRelTol
    stop_tensors = true;
end

if f_couplings_old>0
    f_couplings_rel_change = abs(f_couplings_old-f_couplings)/f_couplings_old;
else
    f_couplings_rel_change = abs(f_couplings_old-f_couplings);
end
if f_couplings < options.AbsFuncTol || f_couplings_rel_change < options.OuterRelTol
    stop_couplings = true;
end

if f_constraints_old>0
    f_constraints_rel_change = abs(f_constraints_old-f_constraints)/f_constraints_old;
else
    f_constraints_rel_change = abs(f_constraints_old-f_constraints);
end
if f_constraints < options.AbsFuncTol || f_constraints_rel_change < options.OuterRelTol
    stop_constraints = true;
end

stop = stop_tensors & stop_couplings & stop_constraints;


end

