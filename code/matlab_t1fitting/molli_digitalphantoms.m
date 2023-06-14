% digital phantoms
%% add path
addpath("functions");
warning('off')
pwd_path = pwd;
%% MOLLI fitting
path = "data/MRXCAT_dataset/test_nii";
label = "../data/MRXCAT/test_original";
MOLLI_REGISTER_FILES = dir(sprintf('../%s/*.nii', path));
for j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractAfter(extractBefore(name, '_MOLLI'), 'phantoms_'); 
    slice = str2num(name(end-4));
%     slice = str2num(name(end-6)); lacot
    
    MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
    
    orig = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    registered_name = strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name);
    original_name = strcat("../data/MRXCAT_dataset/test_nii/", MOLLI_REGISTER_FILES(j).name);
    register_x = niftiread(registered_name);
%     register_x = permute(register_x, [2, 1, 3]);
%     register_x = permute(register_x, [1, 2, 3]);
    x = niftiread(original_name);
    
    mask = niftiread("../data/MRXCAT/myocardium_LV_112x112.nii");
    
    % estimate the center and extent of LV
    center = 50;
    diameter =  56;
    
    % build data structure
    data = struct;
    orig_vols = x;
    regi_vols = squeeze(register_x);
    [x_1, y_1, z_1] = size(orig_vols);
    
    epi_BW = (mask == 1);
    endo_BW = (mask == 2);
    boundary_epi = boundarymask(epi_BW);
    boundary_endo = boundarymask(endo_BW);
    boundary = boundary_endo + boundary_epi;
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        tmp = orig_vols(:,:,i);
        tmp = (tmp - min(tmp(:))) / (max(tmp(:)) - min(tmp(:)));
        ax1 = nexttile; axis off,imshow(labeloverlay(tmp,boundary,'Transparency',0)) 
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_orig_vols.png", label, subjectid));
    
    figure('Position', [1, 1, 1100, 100])
    t = tiledlayout(1,z_1);
    for i=1:z_1
        tmp = regi_vols(:,:,i);
        tmp = (tmp - min(tmp(:))) / (max(tmp(:)) - min(tmp(:)));
        ax1 = nexttile; axis off,imshow(labeloverlay(tmp,boundary,'Transparency',0))
    end
    t.TileSpacing = 'tight';
    t.Padding = 'tight';
    saveas(gcf,sprintf("%s/MOLLI_%s_regi_vols.png", label, subjectid));
    
    data.frames = regi_vols;
    data.tvec = squeeze(orig.tvec_pre(slice, :));
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';
    
    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S, areamask] = mestimation_abs(data, configs);
    
    contour = 0;
    fd = {data, pmap, sd, contour, null_index, S, areamask, epi_BW, endo_BW};
    parsave(sprintf("%s/MOLLI_%s_%d.mat", label, subjectid, slice), fd);
    fprintf("Subject %s Slice %d. \n", subjectid, slice); 
    close all
end