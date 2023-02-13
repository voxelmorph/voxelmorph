function [pmap, sdmap, null_index, S] = mestimation_reweight(data_struct, configs, pmap_mse, null_index, S)

    do_mse = false;
    if nargin < 3
        do_mse = true;      % no initial MSE is given, do MSE first
        if nargin < 2
            configs.type = 'Huber';
            configs.stype = 'MOLLI';
        end
    
    end
    
    if strcmpi(configs.type, 'Gaussian')
        % Gaussian is basically MSE estimator
        do_mse = true;
    end
    
    frames = data_struct.frames;        % we assue that the frames has shape [H, W, L], where L is the number of points. For MOLLI we have L = 11
    [H, W, ~] = size(frames);
    
    % initialize results
    sdmap = zeros(H, W);
    if do_mse
        S = zeros(H, W);
        null_index = zeros(H, W);
    end
    
    if strcmpi(  configs.stype , 'MOLLI' ) % MOLLI fitting
        areamask = false(H, W);
        upleftcorner = max( configs.center - configs.alpha * configs.diameter, 1);
        bottomrightcorner = min( configs.center + configs.alpha * configs.diameter, [W, H] );
        areamask( round(upleftcorner(2) : bottomrightcorner(2)), ...
            round( upleftcorner(1) : bottomrightcorner(1) ) ) = true;
        
        % sort time vector
        tvec = data_struct.tvec;
        [tvec_sorted, tvec_index] = sort(tvec); 
        frames_sorted = frames(:, :, tvec_index);
        
        % initialize parameter map
        pmap = zeros(H, W, 3);
        
        % pixel-wise fitting
%         f = waitbar(0, 'Starting...');
        for tx = 1:1:H
%             waitbar( tx / H, f, sprintf( "Processing %s ... %.1f %%", configs.type, tx/H*100),...
%                 'Name', 'Bar');
            for ty = 1:1:W
                dvec = squeeze( frames_sorted(tx, ty, :) ); 
                if (max(dvec) < 30) || ( ~areamask(tx, ty) ) % mask air voxels
                    continue;
                end
                
                % MSE fitting if necessary
                if do_mse
                    [p_mse, nl] = polarity_recovery_fitting(tvec_sorted(:), ...
                        dvec(:), [150, 2, 1000], 2);
                    dvec(1:nl) = dvec(1:nl) * (-1.0); % polarity recovery
                    pred_mse = MOLLI3param(p_mse, tvec_sorted(:));
                    resid_mse = pred_mse(:) - dvec(:) ;
                    SD = median( abs(resid_mse) ) * 1.48;
                    null_index(tx, ty) = nl;
                    S(tx, ty) = SD;
                else
                    SD = S(tx, ty);
                    nl = null_index(tx, ty);
                    p_mse = squeeze( pmap_mse(tx, ty, :) );
                    dvec(1:nl) = dvec(1:nl) * (-1.0); % polarity recovery
                end
                
                % reweighting
                approxd = gradweight(dvec);
                dweight = ones(size(approxd));
                dweight( approxd < 0 ) = max( - approxd( approxd < 0), 1.0) ;
                
                
                % M-estimators
                options = optimset('MaxFunEvals',5000, 'Display', 'off');
                if strcmpi( configs.type, 'gaussian')
                    pres = p_mse;
                elseif strcmpi( configs.type, 'fair')
                    pres = fminsearch( @(px) sum( ...
                        fairloss((dvec(:) - MOLLI3param(px, tvec_sorted(:) )) ./ dweight,  SD .^ 2) , ...
                        'all'), p_mse, options);
                elseif strcmpi(configs.type, 'huber')
                    pres = fminsearch( @(px) sum( ...
                        huberloss((dvec(:) - MOLLI3param(px, tvec_sorted(:) )) ./ dweight,  SD .^ 2) , ...
                        'all'), p_mse, options);
                    
                elseif strcmpi(configs.type, 'cauchy')
                    pres = fminsearch( @(px) sum( ...
                        cauchyloss( (dvec(:) - MOLLI3param(px, tvec_sorted(:) )) ./ dweight,  SD .^ 2) , ...
                        'all'), p_mse, options);                
                elseif strcmpi(configs.type, 'geman')
                    pres = fminsearch( @(px) sum( ...
                        gemanloss( (dvec(:) - MOLLI3param(px, tvec_sorted(:) )) ./ dweight,  SD .^ 2) , ...
                        'all'), p_mse, options);
                    
                elseif strcmpi(configs.type, 'welsch')
                    pres = fminsearch( @(px) sum( ...
                        welschloss( (dvec(:) - MOLLI3param(px, tvec_sorted(:) )) ./ dweight,  SD .^ 2) , ...
                        'all'), p_mse, options);
                else
                    error("Not implemented!");
                end
                pmap(tx, ty, :) = pres;
                sdmap(tx, ty) = MOLLIComputeSD(pres, tvec_sorted, dvec);
            end
        end
%         close(f);
        
    elseif strcmpi(  configs.stype , 'SAPPHIRE' )
        % initialize results
        pmap = zeros(H, W, 2);        
        tvec = data_struct.tvec;
        [tvec_sorted, tvec_index] = sort(tvec);
        TD = max( tvec(tvec < 1e4), [], 'all');
        frames_sorted = frames(:, :, tvec_index);
        
        % pixel-wise fitting
        f = waitbar(0, 'Starting...');
        for tx = 1:1:H
            waitbar( tx / H, f, sprintf( "Processing %s ... %.1f %%", configs.type, tx/H*100),...
                'Name', 'Bar');
            for ty = 1:1:W
                dvec = squeeze( frames_sorted(tx, ty, :) );
                if max(dvec) < 30 % mask air voxels
                    continue;
                end

                % MSE fitting
                if do_mse
                    [p_mse, nl] = SAPPHIRE_polarity_recovery_fitting(tvec_sorted(:), ...
                        TD, dvec(:), [100, 1000], 2);
                    pmap_mse(tx, ty, :) = p_mse;
                    dvec(1:nl) = dvec(1:nl) * (-1.0); % polarity recovery! Very important!
                    pred_mse = SAPPHIRE1param(p_mse(2), tvec_sorted(:), TD, dvec(end) );
                    resid_mse = pred_mse(:) - dvec(:) ;
                    SD = median( abs(resid_mse) ) * 1.48;
                    S(tx, ty) = SD;
                    null_index(tx, ty) = nl;
                else
                    SD = S(tx, ty);
                    nl = null_index(tx, ty);
                    p_mse = squeeze( pmap_mse(tx, ty, :) );
                    dvec(1:nl) = dvec(1:nl) * (-1.0); % polarity recovery
                end
                
                % Huber-loss fitting
                options = optimset('MaxFunEvals',5000, 'Display', 'off');
                if strcmpi( configs.type, 'gaussian')
                    pres = p_mse(2);
                elseif strcmpi( configs.type, 'fair')
                    pres = fminsearch( @(px) sum( ...
                        fairloss(dvec(:) - SAPPHIRE1param(px, tvec_sorted(:), TD, dvec(end) ),  SD .^ 2) , ...
                        'all'), p_mse(2), options);
                elseif strcmpi(configs.type, 'huber')
                    pres = fminsearch( @(px) sum( ...
                        huberloss(dvec(:) - SAPPHIRE1param(px, tvec_sorted(:), TD, dvec(end) ),  SD .^ 2) , ...
                        'all'), p_mse(2), options);
                    
                elseif strcmpi(configs.type, 'cauchy')
                    pres = fminsearch( @(px) sum( ...
                        cauchyloss(dvec(:) - SAPPHIRE1param(px, tvec_sorted(:), TD, dvec(end) ),  SD .^ 2) , ...
                        'all'), p_mse(2), options);
                    
                elseif strcmpi(configs.type, 'geman')
                    pres = fminsearch( @(px) sum( ...
                        gemanloss(dvec(:) - SAPPHIRE1param(px, tvec_sorted(:), TD, dvec(end) ),  SD .^ 2) , ...
                        'all'), p_mse(2), options);
                    
                elseif strcmpi(configs.type, 'welsch')
                    pres = fminsearch( @(px) sum( ...
                        welschloss(dvec(:) - SAPPHIRE1param(px, tvec_sorted(:), TD, dvec(end) ),  SD .^ 2) , ...
                        'all'), p_mse(2), options);
                end
                pres2 = [dvec(end), pres];
                pmap(tx, ty, :) = pres2;
                sdmap(tx, ty) = SAPPHIREComputeSD2(pres2, tvec_sorted, TD, dvec);
            end
        end
        close(f);
    else
        error("Not implemented yet!");
    end
    
end