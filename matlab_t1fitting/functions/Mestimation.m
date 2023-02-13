function [pmap_huber, SD_map_huber, pmap_cauchy, SD_map_cauchy, pmap_mse, SD_map_mse] = Mestimation(data_struct, configs)
% MESTIMATION Estimate voxel-wise T1 for T1-weighted images, supported
% sequences: MOLLI and SAPPHIRE.
%   [pmap_huber, SD_map_huber, pmap_cauchy, SD_map_cauchy, pmap_mse,
%   SD_map_mse] = Mestimation(data_struct) estimate 
    
    if nargin == 1
        configs.type = 'Huber'; 
        configs.H = -1 ; % -1 for auto
        configs.type = 'MOLLI';
    end
    
    frames = data_struct.frames;        % we assue that the frames has shape [H, W, L], where L is the number of points. For MOLLI we have L = 11
    [H, W, ~] = size(frames);
    
    % initialize results
    SD_map_huber = zeros(H, W);
    SD_map_cauchy = zeros(H, W);
    SD_map_mse = zeros(H, W);
    
    if strcmpi(configs.type , 'MOLLI' ) % MOLLI fitting
        pmap_huber = zeros(H, W, 3);
        pmap_cauchy = zeros(H, W, 3);
        pmap_mse = zeros(H, W, 3);
        
        % sort time vector
        tvec = data_struct.tvec;
        [tvec_sorted, tvec_index] = sort(tvec); 
        frames_sorted = frames(:, :, tvec_index);
        
        % pixel-wise fitting
        for tx = 1:1:H
            for ty = 1:1:W
                dvec = squeeze( frames_sorted(tx, ty, :) );
                if max(dvec) < 30 
                    % mask air voxels
                    continue;
                end
                
                % MSE fitting
                [p_mse, null_index] = polarity_recovery_fitting(tvec_sorted(:), ...
                    dvec(:), [150, 2, 1000], 2);
                pmap_mse(tx, ty, :) = p_mse;
                dvec(1:null_index) = dvec(1:null_index) * (-1.0); % polarity recovery
                pred_mse = MOLLI3param(p_mse, tvec_sorted(:));
                resid_mse = pred_mse(:) - dvec(:) ;
                SD = median( abs(resid_mse) ) * 1.48;
                SD_map_mse(tx, ty) = MOLLIComputeSD(p_mse, tvec_sorted, dvec);
                
                
                % Huber-loss fitting
                options = optimset('MaxFunEvals',5000, 'Display', 'off');
                p_huber= fminsearch( @(px) sum( ...
                    huberloss(dvec(:) - MOLLI3param(px, tvec_sorted(:) ),  SD .^ 2) , ...
                    'all'), p_mse, options);
                pmap_huber(tx, ty, :) = p_huber;
                SD_map_huber(tx, ty, :) = MOLLIComputeSD(p_huber, tvec_sorted, dvec);
                
                % Cauchy-loss fitting
                p_cauchy= fminsearch( @(px) sum( ... 
                    cauchyloss(dvec(:) - MOLLI3param(px, tvec_sorted(:)),  SD .^ 2) ,...
                    'all'), p_mse, options);
                pmap_cauchy(tx, ty, :) = p_cauchy;
                SD_map_cauchy(tx, ty, :) = MOLLIComputeSD(p_cauchy, tvec_sorted, dvec);
                
            end
        end
        
        
        
    elseif strcmpi(  configs.type , 'SAPPHIRE' )
        % TODO: implement SAPPHIRE
        % sort time vector
        
        pmap_huber = zeros(H, W, 2);
        pmap_cauchy = zeros(H, W, 2);
        pmap_mse = zeros(H, W, 2);
    
        tvec = data_struct.tvec;
        [tvec_sorted, tvec_index] = sort(tvec);
        TD = max( tvec(tvec < 1e4), [], 'all');
        frames_sorted = frames(:, :, tvec_index);
        
        % pixel-wise fitting
        for tx = 1:1:H
            for ty = 1:1:W
%                 if( mod( (tx - 1) * H + ty, 1000) == 0 )
%                     fprintf("1k points ==[%d, %d]== \n ", tx, ty);
%                 end
                dvec = squeeze( frames_sorted(tx, ty, :) );
%                 dvec(12) = 100;
                if max(dvec) < 30 
                    % mask air voxels
                    continue;
                end
%                 tvec_sorted( tvec_sorted == 10000) = 100000;
                % MSE fitting
                [p_mse, null_index] = SAPPHIRE_polarity_recovery_fitting(tvec_sorted(:), ...
                    TD, dvec(:), [100, 1000], 2);
                pmap_mse(tx, ty, :) = p_mse;
                dvec(1:null_index) = dvec(1:null_index) * (-1.0); % polarity recovery! Very important!
                pred_mse = SAPPHIRE2param(p_mse, tvec_sorted(:), TD);
                resid_mse = pred_mse(:) - dvec(:) ;
                SD = median( abs(resid_mse) ) * 1.48;
                SD_map_mse(tx, ty) = SAPPHIREComputeSD(p_mse, tvec_sorted, TD, dvec);
                
                
                % Huber-loss fitting
                options = optimset('MaxFunEvals',5000, 'Display', 'off');
                p_huber= fminsearch( @(px) sum( ...
                    huberloss(dvec(:) - SAPPHIRE2param(px, tvec_sorted(:), TD ),  SD .^ 2) , ...
                    'all'), p_mse, options);
                pmap_huber(tx, ty, :) = p_huber;
                SD_map_huber(tx, ty, :) = SAPPHIREComputeSD(p_huber, tvec_sorted, TD, dvec);
                
                % Cauchy-loss fitting
                p_cauchy= fminsearch( @(px) sum( ... 
                    cauchyloss(dvec(:) - SAPPHIRE2param(px, tvec_sorted(:), TD),  SD .^ 2) ,...
                    'all'), p_mse, options);
                pmap_cauchy(tx, ty, :) = p_cauchy;
                SD_map_cauchy(tx, ty, :) = SAPPHIREComputeSD(p_cauchy, tvec_sorted, TD, dvec);
                
            end
        end
        
    else
        error("Not implemented yet!");
    end
    
    

end