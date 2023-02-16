function [p, null_index, sorted_err] = SAPPHIRE_polarity_recovery_fitting(t_vec, TD,  d_vec, p0, solver)
        t_vec = t_vec(:);
        data_vec = d_vec(:);
        dlength = length(data_vec);
        
        
        [~, minD_idx] = min(data_vec);
        inds = minD_idx + [-1, 0, 1];
        inds = min(inds, dlength);
        
        % iteratively revert the polarity
        fitting_err_vals = zeros(length(inds), 1);
        fitting_results = cell(length(inds), 1);
        
         for test_num = 1:1:length(inds)
            data_vec_temp = data_vec;
            data_vec_temp(1:1:inds(test_num)) =  data_vec_temp(1:1:inds(test_num) ) * (-1);
            if solver == 1
                % LM solver
                options = optimoptions('lsqcurvefit','Display','off', 'MaxFunctionEvaluations', 2000);
                [p_iter ,resnorm, ~, ~, ~] =  ...
                    lsqcurvefit(@(px, t) SAPPHIRE2param(px, t, TD), p0, t_vec, data_vec_temp, [1, 1], [1e4, 1e4], options);
                fitting_err_vals(test_num) = resnorm;
                fitting_results{test_num} = p_iter;
            elseif solver == 2
                % Grid search
                rms_fitting  = @(p) mean( ( ...
                    SAPPHIRE2param(p, t_vec, TD) ...
                    - data_vec_temp) .^2, 'all') ;
                options = optimset('Display','off', 'MaxFunEvals', 2000);
                [p_iter, fval, ~, ~] = fminsearch(rms_fitting, p0, options);
                fitting_err_vals(test_num) = fval;
                fitting_results{test_num} = p_iter;
            else
                % not implemented yet
            end
         end
        
        % find the best null point
        [sorted_err, sorted_index] = sort(fitting_err_vals);
        null_index = inds( sorted_index(1) );
        p = fitting_results{sorted_index(1)};

end

