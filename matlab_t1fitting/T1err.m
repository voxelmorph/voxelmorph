%% add path
addpath("functions");

%% MOLLI fitting
MOLLI_NATIVE_FILES = dir("/Users/mona/Documents/repo/voxelmorph-test/data/LisbonD8_postconT1w_mat/*.mat");
for j = 1:length(MOLLI_NATIVE_FILES)
    x = load(strcat(MOLLI_NATIVE_FILES(j).folder, '/', MOLLI_NATIVE_FILES(j).name ));
    tvec(j,:) = x.tvec;
end
FILE = "/Users/mona/Documents/repo/voxelmorph-test/results/LisbonD8_group/test_registered.mat";
load(FILE)
n_subjects = length(MOLLI_NATIVE_FILES);

for j = 1:n_subjects
    vols = permute(img, [2, 3, 1]);
    start = (j-1)*size(tvec,2);
    vols = vols(:,:,1+start:start+size(tvec,2));
    vols = vols*600;
    [x, y, z] = size(vols);
    
    center = [x/2, y/2];
    diameter = [x/2, y/2];
    
    
    % build data structure
    data = struct;
    data.frames = squeeze(vols);
    data.tvec = tvec(j,:);
    
    % fitting configurations
    configs = struct;
    configs.stype = 'MOLLI';
    
    configs.center = center; 
    configs.diameter = diameter;
    configs.alpha = 1.3; % bounding box size = 1.3 x LV extent
    
    % Least square fitting
    configs.type = 'Gaussian';
    [pmap, sd, null_index, S] = mestimation_abs(data, configs);
    
    fd = {data, pmap, sd, null_index, S};
    S_slice(:, :, j) = S;
    sd_slice(:, :, j) = S;

    save(sprintf("data/gMOLLI/MOLLI_test_%d.mat", j), 'fd');
end


%%
mat2Nifti(S_slice, 'test.nii', [1 1 1]);
function [] = mat2Nifti(volume, savepath, voxelSize)
% save Nifti
% reference https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
temp_nii = make_nii(volume);
temp_nii.hdr.dime.pixdim(2:4) = voxelSize;

save_nii(temp_nii, savepath);
disp("Suceess to save the nifti files")
end
