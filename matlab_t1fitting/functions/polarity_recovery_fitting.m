function [p, null_index, sorted_err] = polarity_recovery_fitting(t_vec, data_vec, p0, solver)
%% polarity_recovery_fitting.m 
%   This function fits a 3-parameter T1 model s(t) = c * ( 1- k*e(-t/T1*)
%   by Levernburg-Marquardt or grid-search. The polarity of the first n
%   observed data points are inverted iteratively for n = 1, 2, ..., (#data
%   points). 
        
        % ensure t_vec and d_vec are column vectors
        t_vec = t_vec(:);
        data_vec = data_vec(:);
        dlength = length(data_vec);
        
        [~, minD_idx] = min(data_vec);
        inds = 0:1:minD_idx+2;
%         inds = inds(inds > 0);
        inds = inds(inds < 8);
        
        % iteratively revert the polarity
        fitting_err_vals = zeros(length(inds), 1);
        fitting_results = cell(length(inds), 1);
        
        for test_num = 1:1:length(inds)
            data_vec_temp = data_vec;
            data_vec_temp(1:1:inds(test_num)) =  data_vec_temp(1:1:inds(test_num) ) * (-1);
            t1_3param = @(p, t) p(1) * (1 - p(2) * exp(-t/p(3)));
            if solver == 1
                % LM solver
                options = optimoptions('lsqcurvefit','Display','off', 'MaxFunctionEvaluations', 5000);
                [p_iter ,resnorm, ~, ~, ~] =  ...
                    lsqcurvefit(t1_3param, p0, t_vec, data_vec_temp, [1, 0.01, 1], [1e4, 1000, 1e4], options);
                fitting_err_vals(test_num) = resnorm;
                fitting_results{test_num} = p_iter;
            elseif solver == 2
                % Grid searche
                rms_3param_fitting  = @(p) mean( ( ...
                    t1_3param(p, t_vec) ...
                    - data_vec_temp) .^2, 'all') ;
                options = optimset('Display','off', 'MaxFunEvals', 2000);
                [p_iter, fval, ~, ~] = fminsearch(rms_3param_fitting, p0, options);
                fitting_err_vals(test_num) = fval;
                fitting_results{test_num} = p_iter;
            else
                % not implemented yet
            end
        end
        
        % find the best null point
        [sorted_err, sorted_index] = sort(fitting_err_vals);
        for j = 1:1:length(inds)
           	p = fitting_results{sorted_index(j)};
            if p(2) < 1
                null_index = 3;
                continue;
            else
                null_index = inds( sorted_index(j) );
                break;
            end
        end
end


