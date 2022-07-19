%  This function is only used to save Analyze or NIfTI header that is
%  ended with .hdr and loaded by load_untouch_header_only.m. If you 
%  have NIfTI file that is ended with .nii and you want to change its
%  header only, you can use load_untouch_nii / save_untouch_nii pair.
%  
%  Usage: save_untouch_header_only(hdr, new_header_file_name)
%  
%  hdr - struct with NIfTI / Analyze header fields, which is obtained from:
%        hdr = load_untouch_header_only(original_header_file_name)
%  
%  new_header_file_name - NIfTI / Analyze header name ended with .hdr.
%        You can either copy original.img(.gz) to new.img(.gz) manually,
%        or simply input original.hdr(.gz) in save_untouch_header_only.m
%        to overwrite the original header.
%  
%  - Jimmy Shen (jshen@research.baycrest.org)
%
function save_untouch_header_only(hdr, filename)

   if ~exist('hdr','var') | isempty(hdr) | ~exist('filename','var') | isempty(filename)
      error('Usage: save_untouch_header_only(hdr, filename)');
   end

   v = version;

   %  Check file extension. If .gz, unpack it into temp folder
   %
   if length(filename) > 2 & strcmp(filename(end-2:end), '.gz')

      if ~strcmp(filename(end-6:end), '.hdr.gz')
         error('Please check filename.');
      end

      if str2num(v(1:3)) < 7.1 | ~usejava('jvm')
         error('Please use MATLAB 7.1 (with java) and above, or run gunzip outside MATLAB.');
      else
         gzFile = 1;
         filename = filename(1:end-3);
      end
   end

   [p,f] = fileparts(filename);
   fileprefix = fullfile(p, f);

   write_hdr(hdr, fileprefix);

   %  gzip output file if requested
   %
   if exist('gzFile', 'var')
      gzip([fileprefix, '.hdr']);
      delete([fileprefix, '.hdr']);
   end;

   return					% save_untouch_header_only


%-----------------------------------------------------------------------------------
function write_hdr(hdr, fileprefix)

   fid = fopen(sprintf('%s.hdr',fileprefix),'w');

   if isfield(hdr.hist,'magic')
      save_untouch_nii_hdr(hdr, fid);
   else
      save_untouch0_nii_hdr(hdr, fid);
   end

   fclose(fid);

   return					% write_hdr

