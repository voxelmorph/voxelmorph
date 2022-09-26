% save the mat file to nifti format
% Mona. Jul 19 2022
addpath(genpath('tools'))
input_folder = '/Users/mona/workSpace/data/voxelmorph/MOLLI_original';
output_folder = '/Users/mona/workSpace/data/voxelmorph/MOLLI_original_nii';
dirs = dir(input_folder);
for i = 3:length(dirs)
% for i = 3:5
    file = fullfile(input_folder, dirs(i).name);
%     if contains(file, 'post')
        load(file)
        output = fullfile(output_folder, dirs(i).name);
        volume_re = permute(volume_post, [1, 2, 4, 3]);
        [x, y, z, s] = size(volume_re);
        for j = 1:s
            output_path = [fullfile(output_folder, dirs(i).name(1:end-4)), '_', int2str(j), '_.nii.gz'];
            mat2Nifti(squeeze(volume_re(:,:,:,j)), output_path, [1 1 1]);
        end
%     end

end
function [] = mat2Nifti(volume, savepath, voxelSize)
% save Nifti
% reference https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
temp_nii = make_nii(volume);
temp_nii.hdr.dime.pixdim(2:4) = voxelSize;

save_nii(temp_nii, savepath);
disp("Suceess to save the nifti files")
end