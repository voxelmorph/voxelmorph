%% add path
addpath("functions");
warning('off')
pwd_path = pwd;
%% MOLLI fitting

round=1
path = "data/MOLLI_pre_dataset_v2/test_mat";
MOLLI_REGISTER_FILES = dir(sprintf('../%s/*.mat', path))
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../data/MOLLI_pre_dataset_v2/baseline_mat');
baseline = '/Users/mona/Documents/data/registration/voxelmorph/MOLLI_registered/';
mkdir(label)

% nworker = 10
% myCluster = parcluster('local');
% parpool(myCluster, nworker)

for j = 1:length(MOLLI_REGISTER_FILES)
    name = MOLLI_REGISTER_FILES(j).name;
    subjectid = extractBefore(name, '_MOLLI'); 
    slice = str2num(name(end-4));
    disp(subjectid)
%     register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
    register_x = load(sprintf('%s/%s_MOLLI_pre_groupwise.mat', baseline, subjectid));
    x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));
    
    contour = x.contour2_post{slice};
    % estimate the center and extent of LV
    center = mean(contour.epi, 1);
    diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);
    
    % build data structure
    data = struct;
    orig_vols = squeeze(x.volume_post(:, :, slice, :));
    regi_vols_0 = squeeze(register_x.volume(:,:,slice, :));

    [x_1, y_1, z_1] = size(orig_vols);
    
%     [x_2, y_2, z_2] = size(regi_vols);
    x_2 = 224;
    y_2 = 224;
    for i =1:11
        tmp = imresize(regi_vols_0(:,:,i), [x_2, y_2]);
        img(:,:,i) = tmp(112/2:224-112/2-1, 112/2:224-112/2-1);
    end
    
    save(sprintf("%s/%s_MOLLI_%d.mat", label, subjectid, slice), 'img');
    close all
end