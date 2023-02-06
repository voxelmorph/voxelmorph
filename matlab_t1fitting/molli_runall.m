%% add path
addpath("functions");

%% MOLLI fitting
MOLLI_NATIVE_FILES = dir("/Users/mona/Documents/data/registration/voxelmorph/MOLLI_registered/*_MOLLI_pre_groupwise.mat");
for j = 1:length(MOLLI_NATIVE_FILES)
    x = load(strcat(MOLLI_NATIVE_FILES(j).folder, '/', MOLLI_NATIVE_FILES(j).name ));
    slices = size(x.volume, 3);
    for slice = 1:slices
        contour = x.contour2{slice};
        % estimate the center and extent of LV
        center = mean(contour.epi, 1);
        diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
        
        % build data structure
        data = struct;
        data.frames = squeeze( ...
            x.volume(:, :, slice, :));
        data.tvec = squeeze(x.tvec(slice, :));
        
        % fitting configurations
        configs = struct;
        configs.stype = 'MOLLI';

        configs.center = center; 
        configs.diameter = diameter;
        configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
        
        % Least square fitting
        configs.type = 'Gaussian';
        [pmap, sd, null_index, S] = mestimation_abs(data, configs);
        
        % The following can be ignored.
%         configs.type = 'Huber';
%         [pmap_huber, sd_huber,] = mestimation_abs(data, configs);
%        
%         configs.type = 'Fair';
%         [pmap_fair, sd_fair] = mestimation_abs(data, configs, pmap_mse, null_index, S);
%         
%         
%         configs.type = 'Cauchy';
%         [pmap_cauchy, sd_cauchy] = mestimation_abs(data, configs,...
%             pmap_mse, null_index, 1.5 * S);
%         
%         configs.type = 'Welsch';
%         [pmap_welsch, sd_welsch] = mestimation_abs(data, configs, pmap_mse, null_index, 2 * S);
        
        
        % save all data
%         fd = {data, pmap_mse, pmap_huber, pmap_fair, pmap_cauchy, pmap_welsch, ...
%         sd_mse, sd_huber, sd_fair, sd_cauchy, sd_welsch, ...
%         contour, null_index, S};
        fd = {data, pmap, sd, contour, null_index, S};
        save(sprintf("data/gMOLLI/MOLLI_%d_%d.mat", j, slice), 'fd');
        fprintf("Image %d Slice %d. \n", j, slice); 
        
    end
end

