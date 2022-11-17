%  internal function

%  'xform_nii.m' is an internal function called by "load_nii.m", so
%  you do not need run this program by yourself. It does simplified
%  NIfTI sform/qform affine transform, and supports some of the 
%  affine transforms, including translation, reflection, and 
%  orthogonal rotation (N*90 degree).
%
%  For other affine transforms, e.g. any degree rotation, shearing
%  etc. you will have to use the included 'reslice_nii.m' program
%  to reslice the image volume. 'reslice_nii.m' is not called by
%  any other program, and you have to run 'reslice_nii.m' explicitly
%  for those NIfTI files that you want to reslice them.
%
%  Since 'xform_nii.m' does not involve any interpolation or any
%  slice change, the original image volume is supposed to be
%  untouched, although it is translated, reflected, or even 
%  orthogonally rotated, based on the affine matrix in the
%  NIfTI header.
%
%  However, the affine matrix in the header of a lot NIfTI files
%  contain slightly non-orthogonal rotation. Therefore, optional
%  input parameter 'tolerance' is used to allow some distortion
%  in the loaded image for any non-orthogonal rotation or shearing
%  of NIfTI affine matrix. If you set 'tolerance' to 0, it means
%  that you do not allow any distortion. If you set 'tolerance' to
%  1, it means that you do not care any distortion. The image will
%  fail to be loaded if it can not be tolerated. The tolerance will
%  be set to 0.1 (10%), if it is default or empty.
%
%  Because 'reslice_nii.m' has to perform 3D interpolation, it can
%  be slow depending on image size and affine matrix in the header.
%  
%  After you perform the affine transform, the 'nii' structure
%  generated from 'xform_nii.m' or new NIfTI file created from
%  'reslice_nii.m' will be in RAS orientation, i.e. X axis from
%  Left to Right, Y axis from Posterior to Anterior, and Z axis
%  from Inferior to Superior.
%
%  NOTE: This function should be called immediately after load_nii.
%  
%  Usage: [ nii ] = xform_nii(nii, [tolerance], [preferredForm])
%  
%  nii	- NIFTI structure (returned from load_nii)
%
%  tolerance (optional) - distortion allowed for non-orthogonal rotation
%	or shearing in NIfTI affine matrix. It will be set to 0.1 (10%),
%	if it is default or empty.
%
%  preferredForm (optional)  -  selects which transformation from voxels
%	to RAS coordinates; values are s,q,S,Q.  Lower case s,q indicate
%	"prefer sform or qform, but use others if preferred not present". 
%	Upper case indicate the program is forced to use the specificied
%	tranform or fail loading.  'preferredForm' will be 's', if it is
%	default or empty.	- Jeff Gunter
%  
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function nii = xform_nii(nii, tolerance, preferredForm)

   %  save a copy of the header as it was loaded.  This is the
   %  header before any sform, qform manipulation is done.
   %
   nii.original.hdr = nii.hdr;

   if ~exist('tolerance','var') | isempty(tolerance)
      tolerance = 0.1;
   elseif(tolerance<=0)
      tolerance = eps;
   end

   if ~exist('preferredForm','var') | isempty(preferredForm)
      preferredForm= 's';				% Jeff
   end

   %  if scl_slope field is nonzero, then each voxel value in the
   %  dataset should be scaled as: y = scl_slope * x + scl_inter
   %  I bring it here because hdr will be modified by change_hdr.
   %
   if nii.hdr.dime.scl_slope ~= 0 & ...
	ismember(nii.hdr.dime.datatype, [2,4,8,16,64,256,512,768]) & ...
	(nii.hdr.dime.scl_slope ~= 1 | nii.hdr.dime.scl_inter ~= 0)

      nii.img = ...
	nii.hdr.dime.scl_slope * double(nii.img) + nii.hdr.dime.scl_inter;

      if nii.hdr.dime.datatype == 64

         nii.hdr.dime.datatype = 64;
         nii.hdr.dime.bitpix = 64;
      else
         nii.img = single(nii.img);

         nii.hdr.dime.datatype = 16;
         nii.hdr.dime.bitpix = 32;
      end

      nii.hdr.dime.glmax = max(double(nii.img(:)));
      nii.hdr.dime.glmin = min(double(nii.img(:)));

      %  set scale to non-use, because it is applied in xform_nii
      %
      nii.hdr.dime.scl_slope = 0;

   end

   %  However, the scaling is to be ignored if datatype is DT_RGB24.

   %  If datatype is a complex type, then the scaling is to be applied
   %  to both the real and imaginary parts.
   %
   if nii.hdr.dime.scl_slope ~= 0 & ...
	ismember(nii.hdr.dime.datatype, [32,1792])

      nii.img = ...
	nii.hdr.dime.scl_slope * double(nii.img) + nii.hdr.dime.scl_inter;

      if nii.hdr.dime.datatype == 32
         nii.img = single(nii.img);
      end

      nii.hdr.dime.glmax = max(double(nii.img(:)));
      nii.hdr.dime.glmin = min(double(nii.img(:)));

      %  set scale to non-use, because it is applied in xform_nii
      %
      nii.hdr.dime.scl_slope = 0;

   end

   %  There is no need for this program to transform Analyze data
   %
   if nii.filetype == 0 & exist([nii.fileprefix '.mat'],'file')
      load([nii.fileprefix '.mat']);	% old SPM affine matrix
      R=M(1:3,1:3);
      T=M(1:3,4);
      T=R*ones(3,1)+T;
      M(1:3,4)=T;
      nii.hdr.hist.qform_code=0;
      nii.hdr.hist.sform_code=1;
      nii.hdr.hist.srow_x=M(1,:);
      nii.hdr.hist.srow_y=M(2,:);
      nii.hdr.hist.srow_z=M(3,:);
   elseif nii.filetype == 0
      nii.hdr.hist.rot_orient = [];
      nii.hdr.hist.flip_orient = [];
      return;				% no sform/qform for Analyze format
   end

   hdr = nii.hdr;

   [hdr,orient]=change_hdr(hdr,tolerance,preferredForm);

   %  flip and/or rotate image data
   %
   if ~isequal(orient, [1 2 3])

      old_dim = hdr.dime.dim([2:4]);

      %  More than 1 time frame
      %
      if ndims(nii.img) > 3
         pattern = 1:prod(old_dim);
      else
         pattern = [];
      end

      if ~isempty(pattern)
         pattern = reshape(pattern, old_dim);
      end

      %  calculate for rotation after flip
      %
      rot_orient = mod(orient + 2, 3) + 1;

      %  do flip:
      %
      flip_orient = orient - rot_orient;

      for i = 1:3
         if flip_orient(i)
            if ~isempty(pattern)
               pattern = flipdim(pattern, i);
            else
               nii.img = flipdim(nii.img, i);
            end
         end
      end

      %  get index of orient (rotate inversely)
      %
      [tmp rot_orient] = sort(rot_orient);

      new_dim = old_dim;
      new_dim = new_dim(rot_orient);
      hdr.dime.dim([2:4]) = new_dim;

      new_pixdim = hdr.dime.pixdim([2:4]);
      new_pixdim = new_pixdim(rot_orient);
      hdr.dime.pixdim([2:4]) = new_pixdim;

      %  re-calculate originator
      %
      tmp = hdr.hist.originator([1:3]);
      tmp = tmp(rot_orient);
      flip_orient = flip_orient(rot_orient);

      for i = 1:3
         if flip_orient(i) & ~isequal(tmp(i), 0)
            tmp(i) = new_dim(i) - tmp(i) + 1;
         end
      end

      hdr.hist.originator([1:3]) = tmp;
      hdr.hist.rot_orient = rot_orient;
      hdr.hist.flip_orient = flip_orient;

      %  do rotation:
      %
      if ~isempty(pattern)
         pattern = permute(pattern, rot_orient);
         pattern = pattern(:);

         if hdr.dime.datatype == 32 | hdr.dime.datatype == 1792 | ...
		hdr.dime.datatype == 128 | hdr.dime.datatype == 511

            tmp = reshape(nii.img(:,:,:,1), [prod(new_dim) hdr.dime.dim(5:8)]);
            tmp = tmp(pattern, :);
            nii.img(:,:,:,1) = reshape(tmp, [new_dim       hdr.dime.dim(5:8)]);

            tmp = reshape(nii.img(:,:,:,2), [prod(new_dim) hdr.dime.dim(5:8)]);
            tmp = tmp(pattern, :);
            nii.img(:,:,:,2) = reshape(tmp, [new_dim       hdr.dime.dim(5:8)]);

            if hdr.dime.datatype == 128 | hdr.dime.datatype == 511
               tmp = reshape(nii.img(:,:,:,3), [prod(new_dim) hdr.dime.dim(5:8)]);
               tmp = tmp(pattern, :);
               nii.img(:,:,:,3) = reshape(tmp, [new_dim       hdr.dime.dim(5:8)]);
            end

         else
            nii.img = reshape(nii.img, [prod(new_dim) hdr.dime.dim(5:8)]);
            nii.img = nii.img(pattern, :);
            nii.img = reshape(nii.img, [new_dim       hdr.dime.dim(5:8)]);
         end
      else
         if hdr.dime.datatype == 32 | hdr.dime.datatype == 1792 | ...
		hdr.dime.datatype == 128 | hdr.dime.datatype == 511

            nii.img(:,:,:,1) = permute(nii.img(:,:,:,1), rot_orient);
            nii.img(:,:,:,2) = permute(nii.img(:,:,:,2), rot_orient);

            if hdr.dime.datatype == 128 | hdr.dime.datatype == 511
               nii.img(:,:,:,3) = permute(nii.img(:,:,:,3), rot_orient);
            end
         else
            nii.img = permute(nii.img, rot_orient);
         end
      end
   else
      hdr.hist.rot_orient = [];
      hdr.hist.flip_orient = [];
   end

   nii.hdr = hdr;

   return;					% xform_nii


%-----------------------------------------------------------------------
function [hdr, orient] = change_hdr(hdr, tolerance, preferredForm)

   orient = [1 2 3];
   affine_transform = 1;

   %  NIFTI can have both sform and qform transform. This program
   %  will check sform_code prior to qform_code by default.
   %
   %  If user specifys "preferredForm", user can then choose the
   %  priority.					- Jeff
   %
   useForm=[];					% Jeff

   if isequal(preferredForm,'S')
       if isequal(hdr.hist.sform_code,0)
           error('User requires sform, sform not set in header');
       else
           useForm='s';
       end
   end						% Jeff

   if isequal(preferredForm,'Q')
       if isequal(hdr.hist.qform_code,0)
           error('User requires qform, qform not set in header');
       else
           useForm='q';
       end
   end						% Jeff

   if isequal(preferredForm,'s')
       if hdr.hist.sform_code > 0
           useForm='s';
       elseif hdr.hist.qform_code > 0
           useForm='q';
       end
   end						% Jeff
   
   if isequal(preferredForm,'q')
       if hdr.hist.qform_code > 0
           useForm='q';
       elseif hdr.hist.sform_code > 0
           useForm='s';
       end
   end						% Jeff

   if isequal(useForm,'s')
      R = [hdr.hist.srow_x(1:3)
           hdr.hist.srow_y(1:3)
           hdr.hist.srow_z(1:3)];

      T = [hdr.hist.srow_x(4)
           hdr.hist.srow_y(4)
           hdr.hist.srow_z(4)];

      if det(R) == 0 | ~isequal(R(find(R)), sum(R)')
         hdr.hist.old_affine = [ [R;[0 0 0]] [T;1] ];
         R_sort = sort(abs(R(:)));
         R( find( abs(R) < tolerance*min(R_sort(end-2:end)) ) ) = 0;
         hdr.hist.new_affine = [ [R;[0 0 0]] [T;1] ];

         if det(R) == 0 | ~isequal(R(find(R)), sum(R)')
            msg = [char(10) char(10) '   Non-orthogonal rotation or shearing '];
            msg = [msg 'found inside the affine matrix' char(10)];
            msg = [msg '   in this NIfTI file. You have 3 options:' char(10) char(10)];
            msg = [msg '   1. Using included ''reslice_nii.m'' program to reslice the NIfTI' char(10)];
            msg = [msg '      file. I strongly recommand this, because it will not cause' char(10)];
            msg = [msg '      negative effect, as long as you remember not to do slice' char(10)];
            msg = [msg '      time correction after using ''reslice_nii.m''.' char(10) char(10)];
            msg = [msg '   2. Using included ''load_untouch_nii.m'' program to load image' char(10)];
            msg = [msg '      without applying any affine geometric transformation or' char(10)];
            msg = [msg '      voxel intensity scaling. This is only for people who want' char(10)];
            msg = [msg '      to do some image processing regardless of image orientation' char(10)];
            msg = [msg '      and to save data back with the same NIfTI header.' char(10) char(10)];
            msg = [msg '   3. Increasing the tolerance to allow more distortion in loaded' char(10)];
            msg = [msg '      image, but I don''t suggest this.' char(10) char(10)];
            msg = [msg '   To get help, please type:' char(10) char(10) '   help reslice_nii.m' char(10)];
            msg = [msg '   help load_untouch_nii.m' char(10) '   help load_nii.m'];
            error(msg);
         end
      end

   elseif isequal(useForm,'q')
      b = hdr.hist.quatern_b;
      c = hdr.hist.quatern_c;
      d = hdr.hist.quatern_d;

      if 1.0-(b*b+c*c+d*d) < 0
         if abs(1.0-(b*b+c*c+d*d)) < 1e-5
            a = 0;
         else
            error('Incorrect quaternion values in this NIFTI data.');
         end
      else
         a = sqrt(1.0-(b*b+c*c+d*d));
      end

      qfac = hdr.dime.pixdim(1);
      if qfac==0, qfac = 1; end
      i = hdr.dime.pixdim(2);
      j = hdr.dime.pixdim(3);
      k = qfac * hdr.dime.pixdim(4);

      R = [a*a+b*b-c*c-d*d     2*b*c-2*a*d        2*b*d+2*a*c
           2*b*c+2*a*d         a*a+c*c-b*b-d*d    2*c*d-2*a*b
           2*b*d-2*a*c         2*c*d+2*a*b        a*a+d*d-c*c-b*b];

      T = [hdr.hist.qoffset_x
           hdr.hist.qoffset_y
           hdr.hist.qoffset_z];

      %  qforms are expected to generate rotation matrices R which are
      %  det(R) = 1; we'll make sure that happens.
      %  
      %  now we make the same checks as were done above for sform data
      %  BUT we do it on a transform that is in terms of voxels not mm;
      %  after we figure out the angles and squash them to closest 
      %  rectilinear direction. After that, the voxel sizes are then
      %  added.
      %
      %  This part is modified by Jeff Gunter.
      %
      if det(R) == 0 | ~isequal(R(find(R)), sum(R)')

         %  det(R) == 0 is not a common trigger for this ---
         %  R(find(R)) is a list of non-zero elements in R; if that
         %  is straight (not oblique) then it should be the same as 
         %  columnwise summation. Could just as well have checked the
         %  lengths of R(find(R)) and sum(R)' (which should be 3)
         %
         hdr.hist.old_affine = [ [R * diag([i j k]);[0 0 0]] [T;1] ];
         R_sort = sort(abs(R(:)));
         R( find( abs(R) < tolerance*min(R_sort(end-2:end)) ) ) = 0;
         R = R * diag([i j k]);
         hdr.hist.new_affine = [ [R;[0 0 0]] [T;1] ];

         if det(R) == 0 | ~isequal(R(find(R)), sum(R)')
            msg = [char(10) char(10) '   Non-orthogonal rotation or shearing '];
            msg = [msg 'found inside the affine matrix' char(10)];
            msg = [msg '   in this NIfTI file. You have 3 options:' char(10) char(10)];
            msg = [msg '   1. Using included ''reslice_nii.m'' program to reslice the NIfTI' char(10)];
            msg = [msg '      file. I strongly recommand this, because it will not cause' char(10)];
            msg = [msg '      negative effect, as long as you remember not to do slice' char(10)];
            msg = [msg '      time correction after using ''reslice_nii.m''.' char(10) char(10)];
            msg = [msg '   2. Using included ''load_untouch_nii.m'' program to load image' char(10)];
            msg = [msg '      without applying any affine geometric transformation or' char(10)];
            msg = [msg '      voxel intensity scaling. This is only for people who want' char(10)];
            msg = [msg '      to do some image processing regardless of image orientation' char(10)];
            msg = [msg '      and to save data back with the same NIfTI header.' char(10) char(10)];
            msg = [msg '   3. Increasing the tolerance to allow more distortion in loaded' char(10)];
            msg = [msg '      image, but I don''t suggest this.' char(10) char(10)];
            msg = [msg '   To get help, please type:' char(10) char(10) '   help reslice_nii.m' char(10)];
            msg = [msg '   help load_untouch_nii.m' char(10) '   help load_nii.m'];
            error(msg);
         end

      else
         R = R * diag([i j k]);
      end					% 1st det(R)

   else
      affine_transform = 0;	% no sform or qform transform
   end

   if affine_transform == 1
      voxel_size = abs(sum(R,1));
      inv_R = inv(R);
      originator = inv_R*(-T)+1;
      orient = get_orient(inv_R);

      %  modify pixdim and originator
      %
      hdr.dime.pixdim(2:4) = voxel_size;
      hdr.hist.originator(1:3) = originator;

      %  set sform or qform to non-use, because they have been
      %  applied in xform_nii
      %
      hdr.hist.qform_code = 0;
      hdr.hist.sform_code = 0;
   end

   %  apply space_unit to pixdim if not 1 (mm)
   %
   space_unit = get_units(hdr);

   if space_unit ~= 1
      hdr.dime.pixdim(2:4) = hdr.dime.pixdim(2:4) * space_unit;

      %  set space_unit of xyzt_units to millimeter, because
      %  voxel_size has been re-scaled
      %
      hdr.dime.xyzt_units = char(bitset(hdr.dime.xyzt_units,1,0));
      hdr.dime.xyzt_units = char(bitset(hdr.dime.xyzt_units,2,1));
      hdr.dime.xyzt_units = char(bitset(hdr.dime.xyzt_units,3,0));
   end

   hdr.dime.pixdim = abs(hdr.dime.pixdim);

   return;					% change_hdr


%-----------------------------------------------------------------------
function orient = get_orient(R)

   orient = [];

   for i = 1:3
      switch find(R(i,:)) * sign(sum(R(i,:)))
      case 1
         orient = [orient 1];		% Left to Right
      case 2
         orient = [orient 2];		% Posterior to Anterior
      case 3
         orient = [orient 3];		% Inferior to Superior
      case -1
         orient = [orient 4];		% Right to Left
      case -2
         orient = [orient 5];		% Anterior to Posterior
      case -3
         orient = [orient 6];		% Superior to Inferior
      end
   end

   return;					% get_orient


%-----------------------------------------------------------------------
function [space_unit, time_unit] = get_units(hdr)

   switch bitand(hdr.dime.xyzt_units, 7)	% mask with 0x07
   case 1
      space_unit = 1e+3;		% meter, m
   case 3
      space_unit = 1e-3;		% micrometer, um
   otherwise
      space_unit = 1;			% millimeter, mm
   end

   switch bitand(hdr.dime.xyzt_units, 56)	% mask with 0x38
   case 16
      time_unit = 1e-3;			% millisecond, ms
   case 24
      time_unit = 1e-6;			% microsecond, us
   otherwise
      time_unit = 1;			% second, s
   end

   return;					% get_units

