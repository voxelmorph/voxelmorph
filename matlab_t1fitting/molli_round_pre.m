%% add path
addpath("functions");
warning('off')
pwd_path = pwd;
%% MOLLI fitting

round=1
path = "results/MOLLI_pre/Group/rank_11_11_11_11_11_11_11/nmi/smooth/image_loss_weight1/cycle_loss_weight0.01/weight0.001/bspline/cps4_svfsteps7_svfscale1/e80/test_MOLLI_pre/round";
MOLLI_REGISTER_FILES = dir(sprintf('../%s%d/moved_mat/*.mat', path, round))
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../%s%d/T1_SDerr', path, round)
% MOLLI_REGISTER_FILES = dir(sprintf('../data/MOLLI_pre_dataset/test_mat/*.mat', path));
% MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
% label = sprintf('../data/MOLLI_pre_dataset/T1_SDerr', path)
mkdir(label)

nworker = 10
myCluster = parcluster('local');
parpool(myCluster, nworker)

parfor j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractBefore(name, '_MOLLI'); 
    slice = str2num(name(end-4));
    disp(subjectid)
    register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
    x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    contour = x.contour2_pre{slice};
    % estimate the center and extent of LV
    center = mean(contour.epi, 1);
    diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
    
    % build data structure
    data = struct;
    orig_vols = squeeze(x.volume_pre(:, :, slice, :));
    regi_vols = permute(register_x.img, [2, 1, 3]);
    
    [x_1, y_1, z_1] = size(orig_vols);
    [x_2, y_2, z_2] = size(regi_vols);
    epi_BW = poly2mask(contour.epi(:,1),contour.epi(:,2),x_1, y_1);
    epi_BW = imresize(epi_BW, [x_2, y_2]);
    boundary_epi = boundarymask(epi_BW);

    endo_BW = poly2mask(contour.endo(:,1),contour.endo(:,2),x_1, y_1);    
    endo_BW = imresize(endo_BW, [x_2, y_2]);
    boundary_endo = boundarymask(endo_BW);
    boundary = boundary_endo + boundary_epi;
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(imresize(orig_vols(:,:,i)/255, [x_2, y_2]),boundary,'Transparency',0)) 
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_orig_vols.png", label, subjectid, slice));
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(regi_vols(:,:,i)/255,boundary,'Transparency',0))
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice));
    
    data.frames = regi_vols;
    data.tvec = squeeze(x.tvec_post(slice, :));
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';

    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S, areamask] = mestimation_abs(data, configs);

    fd = {data, pmap, sd, contour, null_index, S, areamask, epi_BW, endo_BW};
    parsave(sprintf("%s/MOLLI_%s_%d.mat", label, subjectid, slice), fd);
    fprintf("Subject %s Slice %d. \n", subjectid, slice); 
    close all
end

round=2
MOLLI_REGISTER_FILES = dir(sprintf('../%s%d/moved_mat/*.mat', path, round))
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../%s%d/T1_SDerr', path, round)
mkdir(label)

parfor j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractBefore(name, '_MOLLI'); 
    slice = str2num(name(end-4));
    disp(subjectid)
    register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
    x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    contour = x.contour2_pre{slice};
    % estimate the center and extent of LV
    center = mean(contour.epi, 1);
    diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
    
    % build data structure
    data = struct;
    orig_vols = squeeze(x.volume_pre(:, :, slice, :));
    regi_vols = permute(register_x.img, [2, 1, 3]);
    
    [x_1, y_1, z_1] = size(orig_vols);
    [x_2, y_2, z_2] = size(regi_vols);
    epi_BW = poly2mask(contour.epi(:,1),contour.epi(:,2),x_1, y_1);
    epi_BW = imresize(epi_BW, [x_2, y_2]);
    boundary_epi = boundarymask(epi_BW);

    endo_BW = poly2mask(contour.endo(:,1),contour.endo(:,2),x_1, y_1);    
    endo_BW = imresize(endo_BW, [x_2, y_2]);
    boundary_endo = boundarymask(endo_BW);
    boundary = boundary_endo + boundary_epi;
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(imresize(orig_vols(:,:,i)/255, [x_2, y_2]),boundary,'Transparency',0)) 
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_orig_vols.png", label, subjectid, slice));
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(regi_vols(:,:,i)/255,boundary,'Transparency',0))
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice));
    
    data.frames = regi_vols;
    data.tvec = squeeze(x.tvec_post(slice, :));
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';

    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S, areamask] = mestimation_abs(data, configs);

    fd = {data, pmap, sd, contour, null_index, S, areamask, epi_BW, endo_BW};
    parsave(sprintf("%s/MOLLI_%s_%d.mat", label, subjectid, slice), fd);
    fprintf("Subject %s Slice %d. \n", subjectid, slice); 
    close all
end

round = 3
MOLLI_REGISTER_FILES = dir(sprintf('../%s%d/moved_mat/*.mat', path, round))
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../%s%d/T1_SDerr', path, round)
mkdir(label)

parfor j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractBefore(name, '_MOLLI'); 
    slice = str2num(name(end-4));
    disp(subjectid)
    register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
    x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    contour = x.contour2_pre{slice};
    % estimate the center and extent of LV
    center = mean(contour.epi, 1);
    diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
    
    % build data structure
    data = struct;
    orig_vols = squeeze(x.volume_pre(:, :, slice, :));
    regi_vols = permute(register_x.img, [2, 1, 3]);
    
    [x_1, y_1, z_1] = size(orig_vols);
    [x_2, y_2, z_2] = size(regi_vols);
    epi_BW = poly2mask(contour.epi(:,1),contour.epi(:,2),x_1, y_1);
    epi_BW = imresize(epi_BW, [x_2, y_2]);
    boundary_epi = boundarymask(epi_BW);

    endo_BW = poly2mask(contour.endo(:,1),contour.endo(:,2),x_1, y_1);    
    endo_BW = imresize(endo_BW, [x_2, y_2]);
    boundary_endo = boundarymask(endo_BW);
    boundary = boundary_endo + boundary_epi;
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(imresize(orig_vols(:,:,i)/255, [x_2, y_2]),boundary,'Transparency',0)) 
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_orig_vols.png", label, subjectid, slice));
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(regi_vols(:,:,i)/255,boundary,'Transparency',0))
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice));
    
    data.frames = regi_vols;
    data.tvec = squeeze(x.tvec_post(slice, :));
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';

    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S, areamask] = mestimation_abs(data, configs);

    fd = {data, pmap, sd, contour, null_index, S, areamask, epi_BW, endo_BW};
    parsave(sprintf("%s/MOLLI_%s_%d.mat", label, subjectid, slice), fd);
    fprintf("Subject %s Slice %d. \n", subjectid, slice); 
    close all
end

round = 4
MOLLI_REGISTER_FILES = dir(sprintf('../%s%d/moved_mat/*.mat', path, round))
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../%s%d/T1_SDerr', path, round)
mkdir(label)

parfor j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractBefore(name, '_MOLLI'); 
    slice = str2num(name(end-4));
    disp(subjectid)
    register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
    x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    contour = x.contour2_pre{slice};
    % estimate the center and extent of LV
    center = mean(contour.epi, 1);
    diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
    
    % build data structure
    data = struct;
    orig_vols = squeeze(x.volume_pre(:, :, slice, :));
    regi_vols = permute(register_x.img, [2, 1, 3]);
    
    [x_1, y_1, z_1] = size(orig_vols);
    [x_2, y_2, z_2] = size(regi_vols);
    epi_BW = poly2mask(contour.epi(:,1),contour.epi(:,2),x_1, y_1);
    epi_BW = imresize(epi_BW, [x_2, y_2]);
    boundary_epi = boundarymask(epi_BW);

    endo_BW = poly2mask(contour.endo(:,1),contour.endo(:,2),x_1, y_1);    
    endo_BW = imresize(endo_BW, [x_2, y_2]);
    boundary_endo = boundarymask(endo_BW);
    boundary = boundary_endo + boundary_epi;
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(imresize(orig_vols(:,:,i)/255, [x_2, y_2]),boundary,'Transparency',0)) 
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_orig_vols.png", label, subjectid, slice));
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(regi_vols(:,:,i)/255,boundary,'Transparency',0))
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice));
    
    data.frames = regi_vols;
    data.tvec = squeeze(x.tvec_post(slice, :));
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';

    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S, areamask] = mestimation_abs(data, configs);

    fd = {data, pmap, sd, contour, null_index, S, areamask, epi_BW, endo_BW};
    parsave(sprintf("%s/MOLLI_%s_%d.mat", label, subjectid, slice), fd);
    fprintf("Subject %s Slice %d. \n", subjectid, slice); 
    close all
end


round = 5
MOLLI_REGISTER_FILES = dir(sprintf('../%s%d/moved_mat/*.mat', path, round))
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../%s%d/T1_SDerr', path, round)
mkdir(label)

parfor j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractBefore(name, '_MOLLI'); 
    slice = str2num(name(end-4));
    disp(subjectid)
    register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
    x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    contour = x.contour2_pre{slice};
    % estimate the center and extent of LV
    center = mean(contour.epi, 1);
    diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
    
    % build data structure
    data = struct;
    orig_vols = squeeze(x.volume_pre(:, :, slice, :));
    regi_vols = permute(register_x.img, [2, 1, 3]);
    
    [x_1, y_1, z_1] = size(orig_vols);
    [x_2, y_2, z_2] = size(regi_vols);
    epi_BW = poly2mask(contour.epi(:,1),contour.epi(:,2),x_1, y_1);
    epi_BW = imresize(epi_BW, [x_2, y_2]);
    boundary_epi = boundarymask(epi_BW);

    endo_BW = poly2mask(contour.endo(:,1),contour.endo(:,2),x_1, y_1);    
    endo_BW = imresize(endo_BW, [x_2, y_2]);
    boundary_endo = boundarymask(endo_BW);
    boundary = boundary_endo + boundary_epi;
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(imresize(orig_vols(:,:,i)/255, [x_2, y_2]),boundary,'Transparency',0)) 
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_orig_vols.png", label, subjectid, slice));
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(regi_vols(:,:,i)/255,boundary,'Transparency',0))
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice));
    
    data.frames = regi_vols;
    data.tvec = squeeze(x.tvec_post(slice, :));
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';

    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S, areamask] = mestimation_abs(data, configs);

    fd = {data, pmap, sd, contour, null_index, S, areamask, epi_BW, endo_BW};
    parsave(sprintf("%s/MOLLI_%s_%d.mat", label, subjectid, slice), fd);
    fprintf("Subject %s Slice %d. \n", subjectid, slice); 
    close all
end


round = 6
MOLLI_REGISTER_FILES = dir(sprintf('../%s%d/moved_mat/*.mat', path, round))
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../%s%d/T1_SDerr', path, round)
mkdir(label)

parfor j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractBefore(name, '_MOLLI'); 
    slice = str2num(name(end-4));
    disp(subjectid)
    register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
    x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    contour = x.contour2_pre{slice};
    % estimate the center and extent of LV
    center = mean(contour.epi, 1);
    diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
    
    % build data structure
    data = struct;
    orig_vols = squeeze(x.volume_pre(:, :, slice, :));
    regi_vols = permute(register_x.img, [2, 1, 3]);
    
    [x_1, y_1, z_1] = size(orig_vols);
    [x_2, y_2, z_2] = size(regi_vols);
    epi_BW = poly2mask(contour.epi(:,1),contour.epi(:,2),x_1, y_1);
    epi_BW = imresize(epi_BW, [x_2, y_2]);
    boundary_epi = boundarymask(epi_BW);

    endo_BW = poly2mask(contour.endo(:,1),contour.endo(:,2),x_1, y_1);    
    endo_BW = imresize(endo_BW, [x_2, y_2]);
    boundary_endo = boundarymask(endo_BW);
    boundary = boundary_endo + boundary_epi;
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(imresize(orig_vols(:,:,i)/255, [x_2, y_2]),boundary,'Transparency',0)) 
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_orig_vols.png", label, subjectid, slice));
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(regi_vols(:,:,i)/255,boundary,'Transparency',0))
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice));
    
    data.frames = regi_vols;
    data.tvec = squeeze(x.tvec_post(slice, :));
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';

    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S, areamask] = mestimation_abs(data, configs);

    fd = {data, pmap, sd, contour, null_index, S, areamask, epi_BW, endo_BW};
    parsave(sprintf("%s/MOLLI_%s_%d.mat", label, subjectid, slice), fd);
    fprintf("Subject %s Slice %d. \n", subjectid, slice); 
    close all
end

round = 7
MOLLI_REGISTER_FILES = dir(sprintf('../%s%d/moved_mat/*.mat', path, round))
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../%s%d/T1_SDerr', path, round)
mkdir(label)

parfor j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractBefore(name, '_MOLLI'); 
    slice = str2num(name(end-4));
    disp(subjectid)
    register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
    x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    contour = x.contour2_pre{slice};
    % estimate the center and extent of LV
    center = mean(contour.epi, 1);
    diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
    
    % build data structure
    data = struct;
    orig_vols = squeeze(x.volume_pre(:, :, slice, :));
    regi_vols = permute(register_x.img, [2, 1, 3]);
    
    [x_1, y_1, z_1] = size(orig_vols);
    [x_2, y_2, z_2] = size(regi_vols);
    epi_BW = poly2mask(contour.epi(:,1),contour.epi(:,2),x_1, y_1);
    epi_BW = imresize(epi_BW, [x_2, y_2]);
    boundary_epi = boundarymask(epi_BW);

    endo_BW = poly2mask(contour.endo(:,1),contour.endo(:,2),x_1, y_1);    
    endo_BW = imresize(endo_BW, [x_2, y_2]);
    boundary_endo = boundarymask(endo_BW);
    boundary = boundary_endo + boundary_epi;
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(imresize(orig_vols(:,:,i)/255, [x_2, y_2]),boundary,'Transparency',0)) 
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_orig_vols.png", label, subjectid, slice));
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        ax1 = nexttile; axis off,imshow(labeloverlay(regi_vols(:,:,i)/255,boundary,'Transparency',0))
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice));
    
    data.frames = regi_vols;
    data.tvec = squeeze(x.tvec_post(slice, :));
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';

    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S, areamask] = mestimation_abs(data, configs);

    fd = {data, pmap, sd, contour, null_index, S, areamask, epi_BW, endo_BW};
    parsave(sprintf("%s/MOLLI_%s_%d.mat", label, subjectid, slice), fd);
    fprintf("Subject %s Slice %d. \n", subjectid, slice); 
    close all
end