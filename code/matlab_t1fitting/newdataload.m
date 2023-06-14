%% add path
addpath("functions");
warning('off')
pwd_path = pwd;

% 
FILE = '/Users/mona/Documents/data/registration/voxelmorph/volume.mat';
OUTPUTFOLDER = '../data/newData_orig';
mkdir(OUTPUTFOLDER)
load(FILE)

for i = 1:size(V, 1)
    data = V{i, 1};
    data = permute(data, [2, 1, 3]);
%     padding = zeros(size(data, 1), size(data, 2), 11-size(data, 3));
%     new_data = cat(3, data, padding);

    output_path = fullfile(OUTPUTFOLDER, [num2str(i), '_orig.nii.gz']);
%     mat2Nifti(new_data, output_path, [1 1 1]);
    mat2Nifti(data, output_path, [1 1 1]);
end
function [] = mat2Nifti(volume, savepath, voxelSize)
% save Nifti
% reference https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
temp_nii = make_nii(volume);
temp_nii.hdr.dime.pixdim(2:4) = voxelSize;

save_nii(temp_nii, savepath);
disp("Suceess to save the nifti files")
end