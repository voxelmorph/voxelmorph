%  Make ANALYZE 7.5 data structure specified by a 3D or 4D matrix.
%  Optional parameters can also be included, such as: voxel_size, 
%  origin, datatype, and description. 
%  
%  Once the ANALYZE structure is made, it can be saved into ANALYZE 7.5 
%  format data file using "save_untouch_nii" command (for more detail, 
%  type: help save_untouch_nii). 
%  
%  Usage: ana = make_ana(img, [voxel_size], [origin], [datatype], [description])
%
%  Where:
%
%	img:		a 3D matrix [x y z], or a 4D matrix with time
%			series [x y z t]. When image is in RGB format,
%			make sure that the size of 4th dimension is 
%			always 3 (i.e. [R G B]). In that case, make 
%			sure that you must specify RGB datatype to 128.
%
%	voxel_size (optional):	Voxel size in millimeter for each
%				dimension. Default is [1 1 1].
%
%	origin (optional):	The AC origin. Default is [0 0 0].
%
%	datatype (optional):	Storage data type:
%		2 - uint8,  4 - int16,  8 - int32,  16 - float32,
%		64 - float64,  128 - RGB24
%			Default will use the data type of 'img' matrix
%			For RGB image, you must specify it to 128.
%
%	description (optional):	Description of data. Default is ''.
%
%  e.g.:
%     origin = [33 44 13]; datatype = 64;
%     ana = make_ana(img, [], origin, datatype);    % default voxel_size
%
%  ANALYZE 7.5 format: http://www.rotman-baycrest.on.ca/~jimmy/ANALYZE75.pdf
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function ana = make_ana(varargin)

   ana.img = varargin{1};
   dims = size(ana.img);
   dims = [4 dims ones(1,8)];
   dims = dims(1:8);

   voxel_size = [0 ones(1,3) zeros(1,4)];
   origin = zeros(1,5);
   descrip = '';

   switch class(ana.img)
      case 'uint8'
         datatype = 2;
      case 'int16'
         datatype = 4;
      case 'int32'
         datatype = 8;
      case 'single'
         datatype = 16;
      case 'double'
         datatype = 64;
      otherwise
         error('Datatype is not supported by make_ana.');
   end

   if nargin > 1 & ~isempty(varargin{2})
      voxel_size(2:4) = double(varargin{2});
   end

   if nargin > 2 & ~isempty(varargin{3})
      origin(1:3) = double(varargin{3});
   end

   if nargin > 3 & ~isempty(varargin{4})
      datatype = double(varargin{4});

      if datatype == 128 | datatype == 511
         dims(5) = [];
         dims = [dims 1];
      end
   end

   if nargin > 4 & ~isempty(varargin{5})
      descrip = varargin{5};
   end

   if ndims(ana.img) > 4
      error('NIfTI only allows a maximum of 4 Dimension matrix.');
   end

   maxval = round(double(max(ana.img(:))));
   minval = round(double(min(ana.img(:))));

   ana.hdr = make_header(dims, voxel_size, origin, datatype, ...
	descrip, maxval, minval);
   ana.filetype = 0;
   ana.ext = [];
   ana.untouch = 1;

   switch ana.hdr.dime.datatype
   case 2
      ana.img = uint8(ana.img);
   case 4
      ana.img = int16(ana.img);
   case 8
      ana.img = int32(ana.img);
   case 16
      ana.img = single(ana.img);
   case 64
      ana.img = double(ana.img);
   case 128
      ana.img = uint8(ana.img);
   otherwise
      error('Datatype is not supported by make_ana.');
   end

   return;					% make_ana


%---------------------------------------------------------------------
function hdr = make_header(dims, voxel_size, origin, datatype, ...
	descrip, maxval, minval)

   hdr.hk   = header_key;
   hdr.dime = image_dimension(dims, voxel_size, datatype, maxval, minval);
   hdr.hist = data_history(origin, descrip);
    
   return;					% make_header


%---------------------------------------------------------------------
function hk = header_key

    hk.sizeof_hdr       = 348;			% must be 348!
    hk.data_type        = '';
    hk.db_name          = '';
    hk.extents          = 0;
    hk.session_error    = 0;
    hk.regular          = 'r';
    hk.hkey_un0         = '0';
    
    return;					% header_key


%---------------------------------------------------------------------
function dime = image_dimension(dims, voxel_size, datatype, maxval, minval)
   
   dime.dim = dims;
   dime.vox_units = 'mm';
   dime.cal_units = '';
   dime.unused1 = 0;
   dime.datatype = datatype;
   
   switch dime.datatype
   case   2,
      dime.bitpix = 8;  precision = 'uint8';
   case   4,
      dime.bitpix = 16; precision = 'int16';
   case   8,
      dime.bitpix = 32; precision = 'int32';
   case  16,
      dime.bitpix = 32; precision = 'float32';
   case  64,
      dime.bitpix = 64; precision = 'float64';
   case 128
      dime.bitpix = 24;  precision = 'uint8';
   otherwise
      error('Datatype is not supported by make_ana.');
   end
   
   dime.dim_un0 = 0;
   dime.pixdim = voxel_size;
   dime.vox_offset = 0;
   dime.roi_scale = 1;
   dime.funused1 = 0;
   dime.funused2 = 0;
   dime.cal_max = 0;
   dime.cal_min = 0;
   dime.compressed = 0;
   dime.verified = 0;
   dime.glmax = maxval;
   dime.glmin = minval;
   
   return;					% image_dimension


%---------------------------------------------------------------------
function hist = data_history(origin, descrip)
   
   hist.descrip = descrip;
   hist.aux_file = 'none';
   hist.orient = 0;
   hist.originator = origin;
   hist.generated = '';
   hist.scannum = '';
   hist.patient_id = '';
   hist.exp_date = '';
   hist.exp_time = '';
   hist.hist_un0 = '';
   hist.views = 0;
   hist.vols_added = 0;
   hist.start_field = 0;
   hist.field_skip = 0;
   hist.omax = 0;
   hist.omin = 0;
   hist.smax = 0;
   hist.smin = 0;
   
   return;					% data_history

