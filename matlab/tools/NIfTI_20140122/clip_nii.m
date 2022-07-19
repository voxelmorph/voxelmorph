%  CLIP_NII: Clip the NIfTI volume from any of the 6 sides
%
%  Usage:	nii = clip_nii(nii, [option])
%
%  Inputs:
%
%  nii - NIfTI volume.
%
%  option - struct instructing how many voxel to be cut from which side.
%
%	option.cut_from_L = ( number of voxel )
%	option.cut_from_R = ( number of voxel )
%	option.cut_from_P = ( number of voxel )
%	option.cut_from_A = ( number of voxel )
%	option.cut_from_I = ( number of voxel )
%	option.cut_from_S = ( number of voxel )
%
%	Options description in detail:
%	==============================
%
%	cut_from_L: Number of voxels from Left side will be clipped.
%
%	cut_from_R: Number of voxels from Right side will be clipped.
%
%	cut_from_P: Number of voxels from Posterior side will be clipped.
%
%	cut_from_A: Number of voxels from Anterior side will be clipped.
%
%	cut_from_I: Number of voxels from Inferior side will be clipped.
%
%	cut_from_S: Number of voxels from Superior side will be clipped.
%
%  NIfTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function nii = clip_nii(nii, opt)

   dims = abs(nii.hdr.dime.dim(2:4));
   origin = abs(nii.hdr.hist.originator(1:3));

   if isempty(origin) | all(origin == 0)		% according to SPM
      origin = round((dims+1)/2);
   end

   cut_from_L = 0;
   cut_from_R = 0;
   cut_from_P = 0;
   cut_from_A = 0;
   cut_from_I = 0;
   cut_from_S = 0;

   if nargin > 1 & ~isempty(opt)
      if ~isstruct(opt)
         error('option argument should be a struct');
      end

      if isfield(opt,'cut_from_L')
         cut_from_L = round(opt.cut_from_L);

         if cut_from_L >= origin(1) | cut_from_L < 0
            error('cut_from_L cannot be negative or cut beyond originator');
         end
      end

      if isfield(opt,'cut_from_P')
         cut_from_P = round(opt.cut_from_P);

         if cut_from_P >= origin(2) | cut_from_P < 0
            error('cut_from_P cannot be negative or cut beyond originator');
         end
      end

      if isfield(opt,'cut_from_I')
         cut_from_I = round(opt.cut_from_I);

         if cut_from_I >= origin(3) | cut_from_I < 0
            error('cut_from_I cannot be negative or cut beyond originator');
         end
      end

      if isfield(opt,'cut_from_R')
         cut_from_R = round(opt.cut_from_R);

         if cut_from_R > dims(1)-origin(1) | cut_from_R < 0
            error('cut_from_R cannot be negative or cut beyond originator');
         end
      end

      if isfield(opt,'cut_from_A')
         cut_from_A = round(opt.cut_from_A);

         if cut_from_A > dims(2)-origin(2) | cut_from_A < 0
            error('cut_from_A cannot be negative or cut beyond originator');
         end
      end

      if isfield(opt,'cut_from_S')
         cut_from_S = round(opt.cut_from_S);

         if cut_from_S > dims(3)-origin(3) | cut_from_S < 0
            error('cut_from_S cannot be negative or cut beyond originator');
         end
      end
   end

   nii = make_nii(nii.img( (cut_from_L+1) : (dims(1)-cut_from_R), ...
			   (cut_from_P+1) : (dims(2)-cut_from_A), ...
			   (cut_from_I+1) : (dims(3)-cut_from_S), ...
			   :,:,:,:,:), nii.hdr.dime.pixdim(2:4), ...
	[origin(1)-cut_from_L origin(2)-cut_from_P origin(3)-cut_from_I], ...
	nii.hdr.dime.datatype, nii.hdr.hist.descrip);

   return;

