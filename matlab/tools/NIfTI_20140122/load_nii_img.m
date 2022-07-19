%  internal function

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)

function [img,hdr] = load_nii_img(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB)

   if ~exist('hdr','var') | ~exist('filetype','var') | ~exist('fileprefix','var') | ~exist('machine','var')
      error('Usage: [img,hdr] = load_nii_img(hdr,filetype,fileprefix,machine,[img_idx],[dim5_idx],[dim6_idx],[dim7_idx],[old_RGB]);');
   end

   if ~exist('img_idx','var') | isempty(img_idx) | hdr.dime.dim(5)<1
      img_idx = [];
   end

   if ~exist('dim5_idx','var') | isempty(dim5_idx) | hdr.dime.dim(6)<1
      dim5_idx = [];
   end

   if ~exist('dim6_idx','var') | isempty(dim6_idx) | hdr.dime.dim(7)<1
      dim6_idx = [];
   end

   if ~exist('dim7_idx','var') | isempty(dim7_idx) | hdr.dime.dim(8)<1
      dim7_idx = [];
   end

   if ~exist('old_RGB','var') | isempty(old_RGB)
      old_RGB = 0;
   end

   %  check img_idx
   %
   if ~isempty(img_idx) & ~isnumeric(img_idx)
      error('"img_idx" should be a numerical array.');
   end

   if length(unique(img_idx)) ~= length(img_idx)
      error('Duplicate image index in "img_idx"');
   end

   if ~isempty(img_idx) & (min(img_idx) < 1 | max(img_idx) > hdr.dime.dim(5))
      max_range = hdr.dime.dim(5);

      if max_range == 1
         error(['"img_idx" should be 1.']);
      else
         range = ['1 ' num2str(max_range)];
         error(['"img_idx" should be an integer within the range of [' range '].']);
      end
   end

   %  check dim5_idx
   %
   if ~isempty(dim5_idx) & ~isnumeric(dim5_idx)
      error('"dim5_idx" should be a numerical array.');
   end

   if length(unique(dim5_idx)) ~= length(dim5_idx)
      error('Duplicate index in "dim5_idx"');
   end

   if ~isempty(dim5_idx) & (min(dim5_idx) < 1 | max(dim5_idx) > hdr.dime.dim(6))
      max_range = hdr.dime.dim(6);

      if max_range == 1
         error(['"dim5_idx" should be 1.']);
      else
         range = ['1 ' num2str(max_range)];
         error(['"dim5_idx" should be an integer within the range of [' range '].']);
      end
   end

   %  check dim6_idx
   %
   if ~isempty(dim6_idx) & ~isnumeric(dim6_idx)
      error('"dim6_idx" should be a numerical array.');
   end

   if length(unique(dim6_idx)) ~= length(dim6_idx)
      error('Duplicate index in "dim6_idx"');
   end

   if ~isempty(dim6_idx) & (min(dim6_idx) < 1 | max(dim6_idx) > hdr.dime.dim(7))
      max_range = hdr.dime.dim(7);

      if max_range == 1
         error(['"dim6_idx" should be 1.']);
      else
         range = ['1 ' num2str(max_range)];
         error(['"dim6_idx" should be an integer within the range of [' range '].']);
      end
   end

   %  check dim7_idx
   %
   if ~isempty(dim7_idx) & ~isnumeric(dim7_idx)
      error('"dim7_idx" should be a numerical array.');
   end

   if length(unique(dim7_idx)) ~= length(dim7_idx)
      error('Duplicate index in "dim7_idx"');
   end

   if ~isempty(dim7_idx) & (min(dim7_idx) < 1 | max(dim7_idx) > hdr.dime.dim(8))
      max_range = hdr.dime.dim(8);

      if max_range == 1
         error(['"dim7_idx" should be 1.']);
      else
         range = ['1 ' num2str(max_range)];
         error(['"dim7_idx" should be an integer within the range of [' range '].']);
      end
   end

   [img,hdr] = read_image(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB);

   return					% load_nii_img


%---------------------------------------------------------------------
function [img,hdr] = read_image(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB)

   switch filetype
   case {0, 1}
      fn = [fileprefix '.img'];
   case 2
      fn = [fileprefix '.nii'];
   end

   fid = fopen(fn,'r',machine);

   if fid < 0,
      msg = sprintf('Cannot open file %s.',fn);
      error(msg);
   end

   %  Set bitpix according to datatype
   %
   %  /*Acceptable values for datatype are*/ 
   %
   %     0 None                     (Unknown bit per voxel) % DT_NONE, DT_UNKNOWN 
   %     1 Binary                         (ubit1, bitpix=1) % DT_BINARY 
   %     2 Unsigned char         (uchar or uint8, bitpix=8) % DT_UINT8, NIFTI_TYPE_UINT8 
   %     4 Signed short                  (int16, bitpix=16) % DT_INT16, NIFTI_TYPE_INT16 
   %     8 Signed integer                (int32, bitpix=32) % DT_INT32, NIFTI_TYPE_INT32 
   %    16 Floating point    (single or float32, bitpix=32) % DT_FLOAT32, NIFTI_TYPE_FLOAT32 
   %    32 Complex, 2 float32      (Use float32, bitpix=64) % DT_COMPLEX64, NIFTI_TYPE_COMPLEX64
   %    64 Double precision  (double or float64, bitpix=64) % DT_FLOAT64, NIFTI_TYPE_FLOAT64 
   %   128 uint8 RGB                 (Use uint8, bitpix=24) % DT_RGB24, NIFTI_TYPE_RGB24 
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
   switch hdr.dime.datatype
   case   1,
      hdr.dime.bitpix = 1;  precision = 'ubit1';
   case   2,
      hdr.dime.bitpix = 8;  precision = 'uint8';
   case   4,
      hdr.dime.bitpix = 16; precision = 'int16';
   case   8,
      hdr.dime.bitpix = 32; precision = 'int32';
   case  16,
      hdr.dime.bitpix = 32; precision = 'float32';
   case  32,
      hdr.dime.bitpix = 64; precision = 'float32';
   case  64,
      hdr.dime.bitpix = 64; precision = 'float64';
   case 128,
      hdr.dime.bitpix = 24; precision = 'uint8';
   case 256 
      hdr.dime.bitpix = 8;  precision = 'int8';
   case 511 
      hdr.dime.bitpix = 96; precision = 'float32';
   case 512 
      hdr.dime.bitpix = 16; precision = 'uint16';
   case 768 
      hdr.dime.bitpix = 32; precision = 'uint32';
   case 1024
      hdr.dime.bitpix = 64; precision = 'int64';
   case 1280
      hdr.dime.bitpix = 64; precision = 'uint64';
   case 1792,
      hdr.dime.bitpix = 128; precision = 'float64';
   otherwise
      error('This datatype is not supported'); 
   end

   hdr.dime.dim(find(hdr.dime.dim < 1)) = 1;

   %  move pointer to the start of image block
   %
   switch filetype
   case {0, 1}
      fseek(fid, 0, 'bof');
   case 2
      fseek(fid, hdr.dime.vox_offset, 'bof');
   end

   %  Load whole image block for old Analyze format or binary image;
   %  otherwise, load images that are specified in img_idx, dim5_idx,
   %  dim6_idx, and dim7_idx
   %
   %  For binary image, we have to read all because pos can not be
   %  seeked in bit and can not be calculated the way below.
   %
   if hdr.dime.datatype == 1 | isequal(hdr.dime.dim(5:8),ones(1,4)) | ...
	(isempty(img_idx) & isempty(dim5_idx) & isempty(dim6_idx) & isempty(dim7_idx))

      %  For each frame, precision of value will be read 
      %  in img_siz times, where img_siz is only the 
      %  dimension size of an image, not the byte storage
      %  size of an image.
      %
      img_siz = prod(hdr.dime.dim(2:8));

      %  For complex float32 or complex float64, voxel values
      %  include [real, imag]
      %
      if hdr.dime.datatype == 32 | hdr.dime.datatype == 1792
         img_siz = img_siz * 2;
      end
	 
      %MPH: For RGB24, voxel values include 3 separate color planes
      %
      if hdr.dime.datatype == 128 | hdr.dime.datatype == 511
	 img_siz = img_siz * 3;
      end

      img = fread(fid, img_siz, sprintf('*%s',precision));

      d1 = hdr.dime.dim(2);
      d2 = hdr.dime.dim(3);
      d3 = hdr.dime.dim(4);
      d4 = hdr.dime.dim(5);
      d5 = hdr.dime.dim(6);
      d6 = hdr.dime.dim(7);
      d7 = hdr.dime.dim(8);

      if isempty(img_idx)
         img_idx = 1:d4;
      end

      if isempty(dim5_idx)
         dim5_idx = 1:d5;
      end

      if isempty(dim6_idx)
         dim6_idx = 1:d6;
      end

      if isempty(dim7_idx)
         dim7_idx = 1:d7;
      end
   else

      d1 = hdr.dime.dim(2);
      d2 = hdr.dime.dim(3);
      d3 = hdr.dime.dim(4);
      d4 = hdr.dime.dim(5);
      d5 = hdr.dime.dim(6);
      d6 = hdr.dime.dim(7);
      d7 = hdr.dime.dim(8);

      if isempty(img_idx)
         img_idx = 1:d4;
      end

      if isempty(dim5_idx)
         dim5_idx = 1:d5;
      end

      if isempty(dim6_idx)
         dim6_idx = 1:d6;
      end

      if isempty(dim7_idx)
         dim7_idx = 1:d7;
      end

      %  compute size of one image
      %
      img_siz = prod(hdr.dime.dim(2:4));

      %  For complex float32 or complex float64, voxel values
      %  include [real, imag]
      %
      if hdr.dime.datatype == 32 | hdr.dime.datatype == 1792
         img_siz = img_siz * 2;
      end

      %MPH: For RGB24, voxel values include 3 separate color planes
      %
      if hdr.dime.datatype == 128 | hdr.dime.datatype == 511
         img_siz = img_siz * 3;
      end

      % preallocate img
      img = zeros(img_siz, length(img_idx)*length(dim5_idx)*length(dim6_idx)*length(dim7_idx) );
      currentIndex = 1;

      for i7=1:length(dim7_idx)
         for i6=1:length(dim6_idx)
            for i5=1:length(dim5_idx)
               for t=1:length(img_idx)

                  %  Position is seeked in bytes. To convert dimension size
                  %  to byte storage size, hdr.dime.bitpix/8 will be
                  %  applied.
                  %
                  pos = sub2ind([d1 d2 d3 d4 d5 d6 d7], 1, 1, 1, ...
			img_idx(t), dim5_idx(i5),dim6_idx(i6),dim7_idx(i7)) -1;
                  pos = pos * hdr.dime.bitpix/8;

                  if filetype == 2
                     fseek(fid, pos + hdr.dime.vox_offset, 'bof');
                  else
                     fseek(fid, pos, 'bof');
                  end

                  %  For each frame, fread will read precision of value
                  %  in img_siz times
                  %
                  img(:,currentIndex) = fread(fid, img_siz, sprintf('*%s',precision));
                  currentIndex = currentIndex +1;

               end
            end
         end
      end
   end

   %  For complex float32 or complex float64, voxel values
   %  include [real, imag]
   %
   if hdr.dime.datatype == 32 | hdr.dime.datatype == 1792
      img = reshape(img, [2, length(img)/2]);
      img = complex(img(1,:)', img(2,:)');
   end

   fclose(fid);

   %  Update the global min and max values 
   %
   hdr.dime.glmax = double(max(img(:)));
   hdr.dime.glmin = double(min(img(:)));

   %  old_RGB treat RGB slice by slice, now it is treated voxel by voxel
   %
   if old_RGB & hdr.dime.datatype == 128 & hdr.dime.bitpix == 24
      % remove squeeze
      img = (reshape(img, [hdr.dime.dim(2:3) 3 hdr.dime.dim(4) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
      img = permute(img, [1 2 4 3 5 6 7 8]);
   elseif hdr.dime.datatype == 128 & hdr.dime.bitpix == 24
      % remove squeeze
      img = (reshape(img, [3 hdr.dime.dim(2:4) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
      img = permute(img, [2 3 4 1 5 6 7 8]);
   elseif hdr.dime.datatype == 511 & hdr.dime.bitpix == 96
      img = double(img(:));
      img = single((img - min(img))/(max(img) - min(img)));
      % remove squeeze
      img = (reshape(img, [3 hdr.dime.dim(2:4) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
      img = permute(img, [2 3 4 1 5 6 7 8]);
   else
      % remove squeeze
      img = (reshape(img, [hdr.dime.dim(2:4) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
   end

   if ~isempty(img_idx)
      hdr.dime.dim(5) = length(img_idx);
   end

   if ~isempty(dim5_idx)
      hdr.dime.dim(6) = length(dim5_idx);
   end

   if ~isempty(dim6_idx)
      hdr.dime.dim(7) = length(dim6_idx);
   end

   if ~isempty(dim7_idx)
      hdr.dime.dim(8) = length(dim7_idx);
   end

   return						% read_image

