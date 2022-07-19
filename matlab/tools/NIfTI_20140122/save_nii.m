%  Save NIFTI dataset. Support both *.nii and *.hdr/*.img file extension.
%  If file extension is not provided, *.hdr/*.img will be used as default.
%  
%  Usage: save_nii(nii, filename, [old_RGB])
%  
%  nii.hdr - struct with NIFTI header fields (from load_nii.m or make_nii.m)
%
%  nii.img - 3D (or 4D) matrix of NIFTI data.
%
%  filename - NIFTI file name.
%
%  old_RGB    - an optional boolean variable to handle special RGB data 
%       sequence [R1 R2 ... G1 G2 ... B1 B2 ...] that is used only by 
%       AnalyzeDirect (Analyze Software). Since both NIfTI and Analyze
%       file format use RGB triple [R1 G1 B1 R2 G2 B2 ...] sequentially
%       for each voxel, this variable is set to FALSE by default. If you
%       would like the saved image only to be opened by AnalyzeDirect 
%       Software, set old_RGB to TRUE (or 1). It will be set to 0, if it
%       is default or empty.
%  
%  Tip: to change the data type, set nii.hdr.dime.datatype,
%	and nii.hdr.dime.bitpix to:
% 
%     0 None                     (Unknown bit per voxel) % DT_NONE, DT_UNKNOWN 
%     1 Binary                         (ubit1, bitpix=1) % DT_BINARY 
%     2 Unsigned char         (uchar or uint8, bitpix=8) % DT_UINT8, NIFTI_TYPE_UINT8 
%     4 Signed short                  (int16, bitpix=16) % DT_INT16, NIFTI_TYPE_INT16 
%     8 Signed integer                (int32, bitpix=32) % DT_INT32, NIFTI_TYPE_INT32 
%    16 Floating point    (single or float32, bitpix=32) % DT_FLOAT32, NIFTI_TYPE_FLOAT32 
%    32 Complex, 2 float32      (Use float32, bitpix=64) % DT_COMPLEX64, NIFTI_TYPE_COMPLEX64
%    64 Double precision  (double or float64, bitpix=64) % DT_FLOAT64, NIFTI_TYPE_FLOAT64 
%   128 uint RGB                  (Use uint8, bitpix=24) % DT_RGB24, NIFTI_TYPE_RGB24 
%   256 Signed char            (schar or int8, bitpix=8) % DT_INT8, NIFTI_TYPE_INT8 
%   511 Single RGB              (Use float32, bitpix=96) % DT_RGB96, NIFTI_TYPE_RGB96
%   512 Unsigned short               (uint16, bitpix=16) % DT_UNINT16, NIFTI_TYPE_UNINT16 
%   768 Unsigned integer             (uint32, bitpix=32) % DT_UNINT32, NIFTI_TYPE_UNINT32 
%  1024 Signed long long              (int64, bitpix=64) % DT_INT64, NIFTI_TYPE_INT64
%  1280 Unsigned long long           (uint64, bitpix=64) % DT_UINT64, NIFTI_TYPE_UINT64 
%  1536 Long double, float128  (Unsupported, bitpix=128) % DT_FLOAT128, NIFTI_TYPE_FLOAT128 
%  1792 Complex128, 2 float64  (Use float64, bitpix=128) % DT_COMPLEX128, NIFTI_TYPE_COMPLEX128 
%  2048 Complex256, 2 float128 (Unsupported, bitpix=256) % DT_COMPLEX128, NIFTI_TYPE_COMPLEX128 
%  
%  Part of this file is copied and modified from:
%  http://www.mathworks.com/matlabcentral/fileexchange/1878-mri-analyze-tools
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%  - "old_RGB" related codes in "save_nii.m" are added by Mike Harms (2006.06.28) 
%
function save_nii(nii, fileprefix, old_RGB)
   
   if ~exist('nii','var') | isempty(nii) | ~isfield(nii,'hdr') | ...
	~isfield(nii,'img') | ~exist('fileprefix','var') | isempty(fileprefix)

      error('Usage: save_nii(nii, filename, [old_RGB])');
   end

   if isfield(nii,'untouch') & nii.untouch == 1
      error('Usage: please use ''save_untouch_nii.m'' for the untouched structure.');
   end

   if ~exist('old_RGB','var') | isempty(old_RGB)
      old_RGB = 0;
   end

   v = version;

   %  Check file extension. If .gz, unpack it into temp folder
   %
   if length(fileprefix) > 2 & strcmp(fileprefix(end-2:end), '.gz')

      if ~strcmp(fileprefix(end-6:end), '.img.gz') & ...
	 ~strcmp(fileprefix(end-6:end), '.hdr.gz') & ...
	 ~strcmp(fileprefix(end-6:end), '.nii.gz')

         error('Please check filename.');
      end

      if str2num(v(1:3)) < 7.1 | ~usejava('jvm')
         error('Please use MATLAB 7.1 (with java) and above, or run gunzip outside MATLAB.');
      else
         gzFile = 1;
         fileprefix = fileprefix(1:end-3);
      end
   end
   
   filetype = 1;

   %  Note: fileprefix is actually the filename you want to save
   %   
   if findstr('.nii',fileprefix) & strcmp(fileprefix(end-3:end), '.nii')
      filetype = 2;
      fileprefix(end-3:end)='';
   end
   
   if findstr('.hdr',fileprefix) & strcmp(fileprefix(end-3:end), '.hdr')
      fileprefix(end-3:end)='';
   end
   
   if findstr('.img',fileprefix) & strcmp(fileprefix(end-3:end), '.img')
      fileprefix(end-3:end)='';
   end

   write_nii(nii, filetype, fileprefix, old_RGB);

   %  gzip output file if requested
   %
   if exist('gzFile', 'var')
      if filetype == 1
         gzip([fileprefix, '.img']);
         delete([fileprefix, '.img']);
         gzip([fileprefix, '.hdr']);
         delete([fileprefix, '.hdr']);
      elseif filetype == 2
         gzip([fileprefix, '.nii']);
         delete([fileprefix, '.nii']);
      end;
   end;

   if filetype == 1

      %  So earlier versions of SPM can also open it with correct originator
      %
      M=[[diag(nii.hdr.dime.pixdim(2:4)) -[nii.hdr.hist.originator(1:3).*nii.hdr.dime.pixdim(2:4)]'];[0 0 0 1]];
      save([fileprefix '.mat'], 'M');
   end
   
   return					% save_nii


%-----------------------------------------------------------------------------------
function write_nii(nii, filetype, fileprefix, old_RGB)

   hdr = nii.hdr;

   if isfield(nii,'ext') & ~isempty(nii.ext)
      ext = nii.ext;
      [ext, esize_total] = verify_nii_ext(ext);
   else
      ext = [];
   end

   switch double(hdr.dime.datatype),
   case   1,
      hdr.dime.bitpix = int16(1 ); precision = 'ubit1';
   case   2,
      hdr.dime.bitpix = int16(8 ); precision = 'uint8';
   case   4,
      hdr.dime.bitpix = int16(16); precision = 'int16';
   case   8,
      hdr.dime.bitpix = int16(32); precision = 'int32';
   case  16,
      hdr.dime.bitpix = int16(32); precision = 'float32';
   case  32,
      hdr.dime.bitpix = int16(64); precision = 'float32';
   case  64,
      hdr.dime.bitpix = int16(64); precision = 'float64';
   case 128,
      hdr.dime.bitpix = int16(24); precision = 'uint8';
   case 256 
      hdr.dime.bitpix = int16(8 ); precision = 'int8';
   case 511,
      hdr.dime.bitpix = int16(96); precision = 'float32';
   case 512 
      hdr.dime.bitpix = int16(16); precision = 'uint16';
   case 768 
      hdr.dime.bitpix = int16(32); precision = 'uint32';
   case 1024
      hdr.dime.bitpix = int16(64); precision = 'int64';
   case 1280
      hdr.dime.bitpix = int16(64); precision = 'uint64';
   case 1792,
      hdr.dime.bitpix = int16(128); precision = 'float64';
   otherwise
      error('This datatype is not supported');
   end
   
   hdr.dime.glmax = round(double(max(nii.img(:))));
   hdr.dime.glmin = round(double(min(nii.img(:))));
   
   if filetype == 2
      fid = fopen(sprintf('%s.nii',fileprefix),'w');
      
      if fid < 0,
         msg = sprintf('Cannot open file %s.nii.',fileprefix);
         error(msg);
      end
      
      hdr.dime.vox_offset = 352;

      if ~isempty(ext)
         hdr.dime.vox_offset = hdr.dime.vox_offset + esize_total;
      end

      hdr.hist.magic = 'n+1';
      save_nii_hdr(hdr, fid);

      if ~isempty(ext)
         save_nii_ext(ext, fid);
      end
   else
      fid = fopen(sprintf('%s.hdr',fileprefix),'w');
      
      if fid < 0,
         msg = sprintf('Cannot open file %s.hdr.',fileprefix);
         error(msg);
      end
      
      hdr.dime.vox_offset = 0;
      hdr.hist.magic = 'ni1';
      save_nii_hdr(hdr, fid);

      if ~isempty(ext)
         save_nii_ext(ext, fid);
      end
      
      fclose(fid);
      fid = fopen(sprintf('%s.img',fileprefix),'w');
   end

   ScanDim = double(hdr.dime.dim(5));		% t
   SliceDim = double(hdr.dime.dim(4));		% z
   RowDim   = double(hdr.dime.dim(3));		% y
   PixelDim = double(hdr.dime.dim(2));		% x
   SliceSz  = double(hdr.dime.pixdim(4));
   RowSz    = double(hdr.dime.pixdim(3));
   PixelSz  = double(hdr.dime.pixdim(2));
   
   x = 1:PixelDim;

   if filetype == 2 & isempty(ext)
      skip_bytes = double(hdr.dime.vox_offset) - 348;
   else
      skip_bytes = 0;
   end

   if double(hdr.dime.datatype) == 128

      %  RGB planes are expected to be in the 4th dimension of nii.img
      %
      if(size(nii.img,4)~=3)
         error(['The NII structure does not appear to have 3 RGB color planes in the 4th dimension']);
      end

      if old_RGB
         nii.img = permute(nii.img, [1 2 4 3 5 6 7 8]);
      else
         nii.img = permute(nii.img, [4 1 2 3 5 6 7 8]);
      end
   end

   if double(hdr.dime.datatype) == 511

      %  RGB planes are expected to be in the 4th dimension of nii.img
      %
      if(size(nii.img,4)~=3)
         error(['The NII structure does not appear to have 3 RGB color planes in the 4th dimension']);
      end

      if old_RGB
         nii.img = permute(nii.img, [1 2 4 3 5 6 7 8]);
      else
         nii.img = permute(nii.img, [4 1 2 3 5 6 7 8]);
      end
   end

   %  For complex float32 or complex float64, voxel values
   %  include [real, imag]
   %
   if hdr.dime.datatype == 32 | hdr.dime.datatype == 1792
      real_img = real(nii.img(:))';
      nii.img = imag(nii.img(:))';
      nii.img = [real_img; nii.img];
   end

   if skip_bytes
      fwrite(fid, zeros(1,skip_bytes), 'uint8');
   end

   fwrite(fid, nii.img, precision);
%   fwrite(fid, nii.img, precision, skip_bytes);        % error using skip
   fclose(fid);

   return;					% write_nii

