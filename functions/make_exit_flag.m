function [exit_flag] = make_exit_flag(iter,f_tensors,f_couplings,f_constraints,f_PAR2_couplings,options,illconditioned)
% which condition caused stop?

if iter>options.MaxOuterIters
    exit_flag = 'maxIterations';
elseif illconditioned 
    exit_flag = "illconditioned lin system";
else
    if f_tensors < options.AbsFuncTol 
        exit_flag.f_tensors = "AbsFuncTol";
    else
        exit_flag.f_tensors = "RelFuncTol";
    end
    if f_couplings < options.AbsFuncTol 
        exit_flag.f_couplings = "AbsFuncTol";
    else
        exit_flag.f_couplings = "RelFuncTol";
    end
    if f_constraints < options.AbsFuncTol 
        exit_flag.f_constraints = "AbsFuncTol";
    else
        exit_flag.f_constraints = "RelFuncTol";
    end
    if f_PAR2_couplings < options.AbsFuncTol
        exit_flag.f_PAR2_couplings = "AbsFuncTol";
    else
        exit_flag.f_PAR2_couplings = "RelFuncTol";
    end
end
   
end

