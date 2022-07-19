%  The basic application of the 'reslice_nii.m' program is to perform
%  any 3D affine transform defined by a NIfTI format image.
%
%  In addition, the 'reslice_nii.m' program can also be applied to
%  generate an isotropic image from either a NIfTI format image or
%  an ANALYZE format image.
%
%  The resliced NIfTI file will always be in RAS orientation.
%
%  This program only supports real integer or floating-point data type.
%  For other data type, the program will exit with an error message 
%  "Transform of this NIFTI data is not supported by the program".
%
%  Usage: reslice_nii(old_fn, new_fn, [voxel_size], [verbose], [bg], ...
%			[method], [img_idx], [preferredForm]);
%
%  old_fn  -	filename for original NIfTI file
%
%  new_fn  -	filename for resliced NIfTI file
%
%  voxel_size (optional)  - size of a voxel in millimeter along x y z
%		direction for resliced NIfTI file. 'voxel_size' will use
%		the minimum voxel_size in original NIfTI header,
%		if it is default or empty.
%
%  verbose (optional) - 1, 0
%			1:  show transforming progress in percentage
%			2:  progress will not be displayed
%			'verbose' is 1 if it is default or empty.
%
%  bg (optional)  -	background voxel intensity in any extra corner that
%			is caused by 3D interpolation. 0 in most cases. 'bg'
%			will be the average of two corner voxel intensities
%			in original image volume, if it is default or empty.
%
%  method (optional)  -	1, 2, or 3
%			1:  for Trilinear interpolation
%			2:  for Nearest Neighbor interpolation
%			3:  for Fischer's Bresenham interpolation
%			'method' is 1 if it is default or empty.
%
%  img_idx (optional)  -  a numerical array of image volume indices. Only
%		the specified volumes will be loaded. All available image
%		volumes will be loaded, if it is default or empty.
%
%	The number of images scans can be obtained from get_nii_frame.m,
%	or simply: hdr.dime.dim(5).
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
%  - Jimmy Shen (jshen@research.baycrest.org)
%
function reslice_nii(old_fn, new_fn, voxel_size, verbose, bg, method, img_idx, preferredForm)

   if ~exist('old_fn','var') | ~exist('new_fn','var')
      error('Usage: reslice_nii(old_fn, new_fn, [voxel_size], [verbose], [bg], [method], [img_idx])');
   end

   if ~exist('method','var') | isempty(method)
      method = 1;
   end

   if ~exist('img_idx','var') | isempty(img_idx)
      img_idx = [];
   end

   if ~exist('verbose','var') | isempty(verbose)
      verbose = 1;
   end

   if ~exist('preferredForm','var') | isempty(preferredForm)
      preferredForm= 's';				% Jeff
   end

   nii = load_nii_no_xform(old_fn, img_idx, 0, preferredForm);

   if ~ismember(nii.hdr.dime.datatype, [2,4,8,16,64,256,512,768])
      error('Transform of this NIFTI data is not supported by the program.');
   end

   if ~exist('voxel_size','var') | isempty(voxel_size)
      voxel_size = abs(min(nii.hdr.dime.pixdim(2:4)))*ones(1,3);
   elseif length(voxel_size) < 3
      voxel_size = abs(voxel_size(1))*ones(1,3);
   end

   if ~exist('bg','var') | isempty(bg)
      bg = mean([nii.img(1) nii.img(end)]);
   end

   old_M = nii.hdr.hist.old_affine;

   if nii.hdr.dime.dim(5) > 1
      for i = 1:nii.hdr.dime.dim(5)
         if verbose
            fprintf('Reslicing %d of %d volumes.\n', i, nii.hdr.dime.dim(5));
         end

         [img(:,:,:,i) M] = ...
		affine(nii.img(:,:,:,i), old_M, voxel_size, verbose, bg, method);
      end
   else
      [img M] = affine(nii.img, old_M, voxel_size, verbose, bg, method);
   end

   new_dim = size(img);
   nii.img = img;
   nii.hdr.dime.dim(2:4) = new_dim(1:3);
   nii.hdr.dime.datatype = 16;
   nii.hdr.dime.bitpix = 32;
   nii.hdr.dime.pixdim(2:4) = voxel_size(:)';
   nii.hdr.dime.glmax = max(img(:));
   nii.hdr.dime.glmin = min(img(:));
   nii.hdr.hist.qform_code = 0;
   nii.hdr.hist.sform_code = 1;
   nii.hdr.hist.srow_x = M(1,:);
   nii.hdr.hist.srow_y = M(2,:);
   nii.hdr.hist.srow_z = M(3,:);
   nii.hdr.hist.new_affine = M;

   save_nii(nii, new_fn);

   return;					% reslice_nii


%--------------------------------------------------------------------
function [nii] = load_nii_no_xform(filename, img_idx, old_RGB, preferredForm)

   if ~exist('filename','var'),
      error('Usage: [nii] = load_nii(filename, [img_idx], [old_RGB])');
   end
   
   if ~exist('img_idx','var'), img_idx = []; end
   if ~exist('old_RGB','var'), old_RGB = 0; end
   if ~exist('preferredForm','var'), preferredForm= 's'; end     % Jeff

   v = version;

   %  Check file extension. If .gz, unpack it into temp folder
   %
   if length(filename) > 2 & strcmp(filename(end-2:end), '.gz')

      if ~strcmp(filename(end-6:end), '.img.gz') & ...
	 ~strcmp(filename(end-6:end), '.hdr.gz') & ...
	 ~strcmp(filename(end-6:end), '.nii.gz')

         error('Please check filename.');
      end

      if str2num(v(1:3)) < 7.1 | ~usejava('jvm')
         error('Please use MATLAB 7.1 (with java) and above, or run gunzip outside MATLAB.');
      elseif strcmp(filename(end-6:end), '.img.gz')
         filename1 = filename;
         filename2 = filename;
         filename2(end-6:end) = '';
         filename2 = [filename2, '.hdr.gz'];

         tmpDir = tempname;
         mkdir(tmpDir);
         gzFileName = filename;

         filename1 = gunzip(filename1, tmpDir);
         filename2 = gunzip(filename2, tmpDir);
         filename = char(filename1);	% convert from cell to string
      elseif strcmp(filename(end-6:end), '.hdr.gz')
         filename1 = filename;
         filename2 = filename;
         filename2(end-6:end) = '';
         filename2 = [filename2, '.img.gz'];

         tmpDir = tempname;
         mkdir(tmpDir);
         gzFileName = filename;

         filename1 = gunzip(filename1, tmpDir);
         filename2 = gunzip(filename2, tmpDir);
         filename = char(filename1);	% convert from cell to string
      elseif strcmp(filename(end-6:end), '.nii.gz')
         tmpDir = tempname;
         mkdir(tmpDir);
         gzFileName = filename;
         filename = gunzip(filename, tmpDir);
         filename = char(filename);	% convert from cell to string
      end
   end

   %  Read the dataset header
   %
   [nii.hdr,nii.filetype,nii.fileprefix,nii.machine] = load_nii_hdr(filename);

   %  Read the header extension
   %
%   nii.ext = load_nii_ext(filename);
   
   %  Read the dataset body
   %
   [nii.img,nii.hdr] = ...
        load_nii_img(nii.hdr,nii.filetype,nii.fileprefix,nii.machine,img_idx,'','','',old_RGB);
   
   %  Perform some of sform/qform transform
   %
%   nii = xform_nii(nii, preferredForm);

   %  Clean up after gunzip
   %
   if exist('gzFileName', 'var')

      %  fix fileprefix so it doesn't point to temp location
      %
      nii.fileprefix = gzFileName(1:end-7);
      rmdir(tmpDir,'s');
   end


   hdr = nii.hdr;

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
           error('User requires sform, sform not set in header');
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

      nii.hdr.hist.old_affine = [ [R;[0 0 0]] [T;1] ];

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
      i = hdr.dime.pixdim(2);
      j = hdr.dime.pixdim(3);
      k = qfac * hdr.dime.pixdim(4);

      R = [a*a+b*b-c*c-d*d     2*b*c-2*a*d        2*b*d+2*a*c
           2*b*c+2*a*d         a*a+c*c-b*b-d*d    2*c*d-2*a*b
           2*b*d-2*a*c         2*c*d+2*a*b        a*a+d*d-c*c-b*b];

      T = [hdr.hist.qoffset_x
           hdr.hist.qoffset_y
           hdr.hist.qoffset_z];

      nii.hdr.hist.old_affine = [ [R * diag([i j k]);[0 0 0]] [T;1] ];

   elseif nii.filetype == 0 & exist([nii.fileprefix '.mat'],'file')
      load([nii.fileprefix '.mat']);	% old SPM affine matrix
      R=M(1:3,1:3);
      T=M(1:3,4);
      T=R*ones(3,1)+T;
      M(1:3,4)=T;
      nii.hdr.hist.old_affine = M;

   else
      M = diag(hdr.dime.pixdim(2:5));
      M(1:3,4) = -M(1:3,1:3)*(hdr.hist.originator(1:3)-1)';
      M(4,4) = 1;
      nii.hdr.hist.old_affine = M;
   end

   return					% load_nii_no_xform

