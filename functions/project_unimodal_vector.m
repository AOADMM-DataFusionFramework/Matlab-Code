 function [x_unimodal] = project_unimodal_vector(x,non_negativity)
 % This function is called in the proximal operator function
 % project_unimodal.m
 % It returns the unimodal (non-negative) projection of column vector x, non-negativity is a
 % boolean
 % This function implements the algorithm described in "Unimodal Regression
 % via Prefix Isotonic Regression", In Computational Statistics and Data
 % Analysis 53 (2008), pp. 289-297 by Quentin F. Stout
 
    num_samples = size(x,1);
    [iso_left,error_left] = prefix_isotonic_regression(x);
    [iso_right,error_right] = prefix_isotonic_regression(flip(x));
    
    best_idx = get_best_unimodality_index(error_left,error_right);
    x_iso_left = compute_isotonic_from_index(best_idx,iso_left(:,1),iso_left(:,2));
    x_iso_right = compute_isotonic_from_index(num_samples - best_idx,iso_right(:,1),iso_right(:,2));
    
    x_unimodal = [x_iso_left;flip(x_iso_right)];
        
    
    function [best_idx] = get_best_unimodality_index(error_left,error_right)
        best_error = error_right(end);
        best_idx = 1;
        for i=2:num_samples
            error = error_left(i) + error_right(num_samples-(i-1));
            if error < best_error
                best_error = error;
                best_idx = i;
            end
        end
        
    end

    function [y_iso] = compute_isotonic_from_index(mode_idx,level_set,index_range)
        y_iso = NaN * ones(mode_idx,1);
        idx = mode_idx;
        while idx >=1
            y_iso(index_range(idx):idx,1) = level_set(idx);
            idx = index_range(idx) - 1;
        end
    end
    
    function [iso, error] = prefix_isotonic_regression(y)
        % all weights are assumed to be 1
        sumwy = [0;y];
        sumwy2 = [0;y.^2];
        sumw = [0;ones(size(y))];
        
        level_set = zeros(size(y,1)+1,1); %mean in paper
        index_range = zeros(size(y,1)+1,1); % left in paper
        error = zeros(size(y,1)+1,1); %+1 since error(1) is error of empty set
        
        level_set(1) = -Inf;
        
        if non_negativity
            cumsumwy2 = cumsum(sumwy2);
            threshold = logical(zeros(size(level_set)));
        end
        
        for i = 2:num_samples+1
            level_set(i) = y(i-1);
            index_range(i) = i;
            while level_set(i) <= level_set(index_range(i)-1)
                merge_intervals_inplace(i, index_range(i)-1)
                index_range(i) = index_range(index_range(i) -1);
            end
            levelerror = sumwy2(i) - (sumwy(i)^2/sumw(i));
            if non_negativity && level_set(i)<0
                threshold(i) = 1;
                error(i) = cumsumwy2(i-1);
            else
                error(i) = levelerror + error(index_range(i)-1);
            end
        end
        if non_negativity
            level_set(threshold) = 0;
        end
        
        iso = [level_set(2:end),index_range(2:end)-1];
        error = error(2:end);
        
        function merge_intervals_inplace(merge_target, merger)
            sumwy(merge_target) = sumwy(merge_target) + sumwy(merger);
            sumwy2(merge_target) = sumwy2(merge_target) + sumwy2(merger);
            sumw(merge_target) = sumw(merge_target) + sumw(merger);
            level_set(merge_target) = sumwy(merge_target)/sumw(merge_target);
        end
    end
        
        
end

