%  PAD_NII: Pad the NIfTI volume from any of the 6 sides
%
%  Usage:	nii = pad_nii(nii, [option])
%
%  Inputs:
%
%  nii - NIfTI volume.
%
%  option - struct instructing how many voxel to be padded from which side.
%
%	option.pad_from_L = ( number of voxel )
%	option.pad_from_R = ( number of voxel )
%	option.pad_from_P = ( number of voxel )
%	option.pad_from_A = ( number of voxel )
%	option.pad_from_I = ( number of voxel )
%	option.pad_from_S = ( number of voxel )
%	option.bg = [0]
%
%	Options description in detail:
%	==============================
%
%	pad_from_L: Number of voxels from Left side will be padded.
%
%	pad_from_R: Number of voxels from Right side will be padded.
%
%	pad_from_P: Number of voxels from Posterior side will be padded.
%
%	pad_from_A: Number of voxels from Anterior side will be padded.
%
%	pad_from_I: Number of voxels from Inferior side will be padded.
%
%	pad_from_S: Number of voxels from Superior side will be padded.
%
%	bg: Background intensity, which is 0 by default.
%
%  NIfTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jshen@research.baycrest.org)
%
function nii = pad_nii(nii, opt)

   dims = abs(nii.hdr.dime.dim(2:4));
   origin = abs(nii.hdr.hist.originator(1:3));

   if isempty(origin) | all(origin == 0)		% according to SPM
      origin = round((dims+1)/2);
   end

   pad_from_L = 0;
   pad_from_R = 0;
   pad_from_P = 0;
   pad_from_A = 0;
   pad_from_I = 0;
   pad_from_S = 0;
   bg = 0;

   if nargin > 1 & ~isempty(opt)
      if ~isstruct(opt)
         error('option argument should be a struct');
      end

      if isfield(opt,'pad_from_L')
         pad_from_L = round(opt.pad_from_L);

         if pad_from_L >= origin(1) | pad_from_L < 0
            error('pad_from_L cannot be negative');
         end
      end

      if isfield(opt,'pad_from_P')
         pad_from_P = round(opt.pad_from_P);

         if pad_from_P >= origin(2) | pad_from_P < 0
            error('pad_from_P cannot be negative');
         end
      end

      if isfield(opt,'pad_from_I')
         pad_from_I = round(opt.pad_from_I);

         if pad_from_I >= origin(3) | pad_from_I < 0
            error('pad_from_I cannot be negative');
         end
      end

      if isfield(opt,'pad_from_R')
         pad_from_R = round(opt.pad_from_R);

         if pad_from_R > dims(1)-origin(1) | pad_from_R < 0
            error('pad_from_R cannot be negative');
         end
      end

      if isfield(opt,'pad_from_A')
         pad_from_A = round(opt.pad_from_A);

         if pad_from_A > dims(2)-origin(2) | pad_from_A < 0
            error('pad_from_A cannot be negative');
         end
      end

      if isfield(opt,'pad_from_S')
         pad_from_S = round(opt.pad_from_S);

         if pad_from_S > dims(3)-origin(3) | pad_from_S < 0
            error('pad_from_S cannot be negative');
         end
      end

      if isfield(opt,'bg')
         bg = opt.bg;
      end
   end

   blk = bg * ones( pad_from_L, dims(2), dims(3) );
   nii.img = cat(1, blk, nii.img);

   blk = bg * ones( pad_from_R, dims(2), dims(3) );
   nii.img = cat(1, nii.img, blk);

   dims = size(nii.img);

   blk = bg * ones( dims(1), pad_from_P, dims(3) );
   nii.img = cat(2, blk, nii.img);

   blk = bg * ones( dims(1), pad_from_A, dims(3) );
   nii.img = cat(2, nii.img, blk);

   dims = size(nii.img);

   blk = bg * ones( dims(1), dims(2), pad_from_I );
   nii.img = cat(3, blk, nii.img);

   blk = bg * ones( dims(1), dims(2), pad_from_S );
   nii.img = cat(3, nii.img, blk);

   nii = make_nii(nii.img, nii.hdr.dime.pixdim(2:4), ...
	[origin(1)+pad_from_L origin(2)+pad_from_P origin(3)+pad_from_I], ...
	nii.hdr.dime.datatype, nii.hdr.hist.descrip);

   return;

