%% add path
addpath("functions");
warning('off')
pwd_path = pwd;
%% MOLLI fitting

round=1;
path = "results/MOLLI_pre/Group/rank_11_0_0/tc/smooth/image_loss_weight1/cycle_loss_weight0.01/weight0.001/bspline/cps4_svfsteps7_svfscale1/e240/test_MOLLI_post/round";
MOLLI_REGISTER_FILES = dir(sprintf('../%s%d/moved_mat/*.mat', path, round));
MOLLI_NATIVE_FOLDER = '../data/MOLLI_original';
label = sprintf('../%s%d/T1_SDerr', path, round)

j=10;
name = MOLLI_REGISTER_FILES(j).name;
subjectid = extractBefore(name, '_MOLLI'); 
slice = str2num(name(end-4));
disp(subjectid)
register_x = load(strcat(MOLLI_REGISTER_FILES(j).folder, '/', MOLLI_REGISTER_FILES(j).name ));
x = load(strcat(MOLLI_NATIVE_FOLDER, '/', subjectid, '_MOLLI.mat'));

contour = x.contour2_post{slice};
% estimate the center and extent of LV
center = mean(contour.epi, 1);
diameter =  max(contour.epi, [],  1) - min(contour.epi, [],  1);

% build data structure
data = struct;
orig_vols = squeeze(x.volume_post(:, :, slice, :));
regi_vols = permute(register_x.img, [2, 1, 3]);

[x_1, y_1, z_1] = size(orig_vols);
[x_2, y_2, z_2] = size(regi_vols);
epi_BW = poly2mask(contour.epi(:,1),contour.epi(:,2),x_1, y_1);
epi_BW = imresize(epi_BW, [x_2, y_2]);
boundary_epi = boundarymask(epi_BW);

endo_BW = poly2mask(contour.endo(:,1),contour.endo(:,2),x_1, y_1);    
endo_BW = imresize(endo_BW, [x_2, y_2]);
boundary_endo = boundarymask(endo_BW);
% boundary = boundary_endo + boundary_epi;
boundary = boundary_epi;
% figure('Position', [1, 1, 1100, 100])
z_1 = 4
% t1 = tiledlayout(1,z_1);
% for i=1:z_1
%     ax1 = nexttile; axis off,imshow(labeloverlay(imresize(orig_vols(:,:,i)/255, [x_2, y_2]),boundary,'Transparency',0)) 
% end
% t1.TileSpacing = 'none';
% t1.Padding = 'tight';
% 
t1 = tiledlayout(z_1, 1);
for i=1:z_1
    ax1 = nexttile; axis off,imshow(imresize(orig_vols(:,:,i)/255, [x_2, y_2])) 
end
t1.TileSpacing = 'none';
t1.Padding = 'tight';

% exportgraphics(gcf,sprintf("%s/MOLLI_%s_%d_orig_vols.png", label, subjectid, slice),'Resolution',600)
% 
% % figure('Position', [1, 1, 1100, 100])
% t2 = tiledlayout(1,z_1);
% for i=1:z_1
%     ax2 = nexttile; axis off,imshow(labeloverlay(regi_vols(:,:,i)/255,boundary,'Transparency',0))
% end
% t2.TileSpacing = 'none';
% t2.Padding = 'tight';
% exportgraphics(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice),'Resolution',600)
