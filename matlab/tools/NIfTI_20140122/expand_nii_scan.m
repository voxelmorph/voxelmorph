%  Expand a multiple-scan NIFTI file into multiple single-scan NIFTI files
%
%  Usage: expand_nii_scan(multi_scan_filename, [img_idx], [path_to_save])
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function expand_nii_scan(filename, img_idx, newpath)

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
      else
         gzFile = 1;
      end
   end

   if ~exist('newpath','var') | isempty(newpath), newpath = pwd; end
   if ~exist('img_idx','var') | isempty(img_idx), img_idx = 1:get_nii_frame(filename); end

   for i=img_idx
      nii_i = load_untouch_nii(filename, i);

      fn = [nii_i.fileprefix '_' sprintf('%04d',i)];
      pnfn = fullfile(newpath, fn);

      if exist('gzFile', 'var')
         pnfn = [pnfn '.nii.gz'];
      end

      save_untouch_nii(nii_i, pnfn);
   end

   return;					% expand_nii_scan

