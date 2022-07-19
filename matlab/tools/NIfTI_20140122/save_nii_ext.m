%  Save NIFTI header extension.
%
%  Usage: save_nii_ext(ext, fid)
%
%  ext - struct with NIFTI header extension fields.
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function save_nii_ext(ext, fid)

   if ~exist('ext','var') | ~exist('fid','var')
      error('Usage: save_nii_ext(ext, fid)');
   end

   if ~isfield(ext,'extension') | ~isfield(ext,'section') | ~isfield(ext,'num_ext')
      error('Wrong header extension');
   end

   write_ext(ext, fid);

   return;                                      % save_nii_ext


%---------------------------------------------------------------------
function write_ext(ext, fid)

   fwrite(fid, ext.extension, 'uchar');

   for i=1:ext.num_ext
      fwrite(fid, ext.section(i).esize, 'int32');
      fwrite(fid, ext.section(i).ecode, 'int32');
      fwrite(fid, ext.section(i).edata, 'uchar');
   end

   return;                                      % write_ext

