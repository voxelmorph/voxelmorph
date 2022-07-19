%  When you load any ANALYZE or NIfTI file with 'load_nii.m', and view
%  it with 'view_nii.m', you may find that the image is L-R flipped.
%  This is because of the confusion of radiological and neurological
%  convention in the medical image before NIfTI format is adopted. You
%  can find more details from:
%
%  http://www.rotman-baycrest.on.ca/~jimmy/UseANALYZE.htm
%
%  Sometime, people even want to convert RAS (standard orientation) back
%  to LAS orientation to satisfy the legend programs or processes. This
%  program is only written for those purpose. So PLEASE BE VERY CAUTIOUS
%  WHEN USING THIS 'FLIP_LR.M' PROGRAM.
%
%  With 'flip_lr.m', you can convert any ANALYZE or NIfTI (no matter
%  3D or 4D) file to a flipped NIfTI file. This is implemented simply
%  by flipping the affine matrix in the NIfTI header. Since the L-R
%  orientation is determined there, so the image will be flipped.
%
%  Usage: flip_lr(original_fn, flipped_fn, [old_RGB],[tolerance],[preferredForm])
%
%  original_fn  -  filename of the original ANALYZE or NIfTI (3D or 4D) file
%
%  flipped_fn  -  filename of the L-R flipped NIfTI file
%
%  old_RGB (optional)  -  a scale number to tell difference of new RGB24
%	from old RGB24. New RGB24 uses RGB triple sequentially for each
%	voxel, like [R1 G1 B1 R2 G2 B2 ...]. Analyze 6.0 from AnalyzeDirect
%	uses old RGB24, in a way like [R1 R2 ... G1 G2 ... B1 B2 ...] for
%	each slices. If the image that you view is garbled, try to set 
%	old_RGB variable to 1 and try again, because it could be in
%	old RGB24. It will be set to 0, if it is default or empty.
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
%  Example: flip_lr('avg152T1_LR_nifti.nii', 'flipped_lr.nii');
%           flip_lr('avg152T1_RL_nifti.nii', 'flipped_rl.nii');
%
%  You will find that 'avg152T1_LR_nifti.nii' and 'avg152T1_RL_nifti.nii'
%  are the same, and 'flipped_lr.nii' and 'flipped_rl.nii' are also the
%  the same, but they are L-R flipped from 'avg152T1_*'.
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function flip_lr(original_fn, flipped_fn, old_RGB, tolerance, preferredForm)

   if ~exist('original_fn','var') | ~exist('flipped_fn','var')
      error('Usage: flip_lr(original_fn, flipped_fn, [old_RGB],[tolerance])');
   end

   if ~exist('old_RGB','var') | isempty(old_RGB)
      old_RGB = 0;
   end

   if ~exist('tolerance','var') | isempty(tolerance)
      tolerance = 0.1;
   end

   if ~exist('preferredForm','var') | isempty(preferredForm)
      preferredForm= 's';				% Jeff
   end

   nii = load_nii(original_fn, [], [], [], [], old_RGB, tolerance, preferredForm);
   M = diag(nii.hdr.dime.pixdim(2:5));
   M(1:3,4) = -M(1:3,1:3)*(nii.hdr.hist.originator(1:3)-1)';
   M(1,:) = -1*M(1,:);
   nii.hdr.hist.sform_code = 1;
   nii.hdr.hist.srow_x = M(1,:);
   nii.hdr.hist.srow_y = M(2,:);
   nii.hdr.hist.srow_z = M(3,:);
   save_nii(nii, flipped_fn);

   return;					% flip_lr

