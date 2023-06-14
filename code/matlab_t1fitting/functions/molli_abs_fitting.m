function [p, res, dvec_pred] = molli_abs_fitting(t_vec, data_vec, p0, solver)
%% abs_fitting.m 
%   This function fits a 3-parameter T1 model |s(t) = c * ( 1- k*e(-t/T1*)|
%   by Levernburg-Marquardt or grid-search. 
        
        % ensure t_vec and d_vec are column vectors
        t_vec = t_vec(:);
        data_vec = data_vec(:);
        
        % iteratively revert the polarity
        t1_3param = @(p, t) abs(p(1) * (1 - p(2) * exp(-t/p(3))));
        if solver == 1
            % LM solver
            options = optimoptions('lsqcurvefit','Display','off');
            [p, res, ~, ~, ~] =  ...
                lsqcurvefit(t1_3param, p0, t_vec, data_vec, [1, 0.01, 1], [1e4, 1000, 1e4], options);
        elseif solver == 2
            % Grid searcher
            rms_3param_fitting  = @(p) mean( ( ...
                t1_3param(p, t_vec) ...
                - data_vec) .^2, 'all') ;
            options = optimset('Display','off');
            [p, res, ~, ~] = fminsearch(rms_3param_fitting, p0, options);
        else
            % not implemented yet
        end
        dvec_pred = t1_3param(p, t_vec);
end