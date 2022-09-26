% save the mat file to nifti format
% Mona. Jul 19 2022
addpath(genpath('tools'))
input_folder = '/Users/mona/workSpace/data/voxelmorph/MOLLI_registered';
output_folder = '/Users/mona/workSpace/data/voxelmorph/MOLLI_registered_nii';
dirs = dir(input_folder);
for i = 4:length(dirs)
    subdirs = dir(fullfile(input_folder, dirs(i).name));
    output = fullfile(output_folder, dirs(i).name);
    mkdir(output)
    for j = 3:length(subdirs)
        file = fullfile(subdirs(j).folder, subdirs(j).name)
        load(file);
        output_path = [fullfile(output, subdirs(j).name(1:end-4)), '.nii.gz'];
        mat2Nifti(rawimages, output_path, [1 1 1]);
    end

end
function [] = mat2Nifti(volume, savepath, voxelSize)
% save Nifti
% reference https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
temp_nii = make_nii(volume);
temp_nii.hdr.dime.pixdim(2:4) = voxelSize;

save_nii(temp_nii, savepath);
disp("Suceess to save the nifti files")
end