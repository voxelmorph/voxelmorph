%  internal function
  
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)

function save_nii_hdr(hdr, fid)
   
   if ~exist('hdr','var') | ~exist('fid','var')
      error('Usage: save_nii_hdr(hdr, fid)');
   end
   
   if ~isequal(hdr.hk.sizeof_hdr,348),
      error('hdr.hk.sizeof_hdr must be 348.');
   end
   
   if hdr.hist.qform_code == 0 & hdr.hist.sform_code == 0
      hdr.hist.sform_code = 1;
      hdr.hist.srow_x(1) = hdr.dime.pixdim(2);
      hdr.hist.srow_x(2) = 0;
      hdr.hist.srow_x(3) = 0;
      hdr.hist.srow_y(1) = 0;
      hdr.hist.srow_y(2) = hdr.dime.pixdim(3);
      hdr.hist.srow_y(3) = 0;
      hdr.hist.srow_z(1) = 0;
      hdr.hist.srow_z(2) = 0;
      hdr.hist.srow_z(3) = hdr.dime.pixdim(4);
      hdr.hist.srow_x(4) = (1-hdr.hist.originator(1))*hdr.dime.pixdim(2);
      hdr.hist.srow_y(4) = (1-hdr.hist.originator(2))*hdr.dime.pixdim(3);
      hdr.hist.srow_z(4) = (1-hdr.hist.originator(3))*hdr.dime.pixdim(4);
   end
   
   write_header(hdr, fid);

   return;					% save_nii_hdr


%---------------------------------------------------------------------
function write_header(hdr, fid)

        %  Original header structures
	%  struct dsr				/* dsr = hdr */
	%       { 
	%       struct header_key hk;            /*   0 +  40       */
	%       struct image_dimension dime;     /*  40 + 108       */
	%       struct data_history hist;        /* 148 + 200       */
	%       };                               /* total= 348 bytes*/
   
   header_key(fid, hdr.hk);
   image_dimension(fid, hdr.dime);
   data_history(fid, hdr.hist);
   
   %  check the file size is 348 bytes
   %
   fbytes = ftell(fid);
   
   if ~isequal(fbytes,348),
      msg = sprintf('Header size is not 348 bytes.');
      warning(msg);
   end
    
   return;					% write_header


%---------------------------------------------------------------------
function header_key(fid, hk)
   
   fseek(fid,0,'bof');

	%  Original header structures    
	%  struct header_key                      /* header key      */ 
	%       {                                /* off + size      */
	%       int sizeof_hdr                   /*  0 +  4         */
	%       char data_type[10];              /*  4 + 10         */
	%       char db_name[18];                /* 14 + 18         */
	%       int extents;                     /* 32 +  4         */
	%       short int session_error;         /* 36 +  2         */
	%       char regular;                    /* 38 +  1         */
	%       char dim_info;   % char hkey_un0;        /* 39 +  1 */
	%       };                               /* total=40 bytes  */
        
   fwrite(fid, hk.sizeof_hdr(1),    'int32');	% must be 348.
    
   % data_type = sprintf('%-10s',hk.data_type);	% ensure it is 10 chars from left
   % fwrite(fid, data_type(1:10), 'uchar');
   pad = zeros(1, 10-length(hk.data_type));
   hk.data_type = [hk.data_type  char(pad)];
   fwrite(fid, hk.data_type(1:10), 'uchar');
    
   % db_name   = sprintf('%-18s', hk.db_name);	% ensure it is 18 chars from left
   % fwrite(fid, db_name(1:18), 'uchar');
   pad = zeros(1, 18-length(hk.db_name));
   hk.db_name = [hk.db_name  char(pad)];
   fwrite(fid, hk.db_name(1:18), 'uchar');
    
   fwrite(fid, hk.extents(1),       'int32');
   fwrite(fid, hk.session_error(1), 'int16');
   fwrite(fid, hk.regular(1),       'uchar');	% might be uint8
    
   % fwrite(fid, hk.hkey_un0(1),    'uchar');
   % fwrite(fid, hk.hkey_un0(1),    'uint8');
   fwrite(fid, hk.dim_info(1),      'uchar');
    
   return;					% header_key


%---------------------------------------------------------------------
function image_dimension(fid, dime)

	%  Original header structures        
	%  struct image_dimension
	%       {                                /* off + size      */
	%       short int dim[8];                /* 0 + 16          */
	%       float intent_p1;   % char vox_units[4];   /* 16 + 4       */
	%       float intent_p2;   % char cal_units[8];   /* 20 + 4       */
	%       float intent_p3;   % char cal_units[8];   /* 24 + 4       */
	%       short int intent_code;   % short int unused1;   /* 28 + 2 */
	%       short int datatype;              /* 30 + 2          */
	%       short int bitpix;                /* 32 + 2          */
	%       short int slice_start;   % short int dim_un0;   /* 34 + 2 */
	%       float pixdim[8];                 /* 36 + 32         */
	%			/*
	%				pixdim[] specifies the voxel dimensions:
	%				pixdim[1] - voxel width
	%				pixdim[2] - voxel height
	%				pixdim[3] - interslice distance
	%				pixdim[4] - volume timing, in msec
	%					..etc
	%			*/
	%       float vox_offset;                /* 68 + 4          */
	%       float scl_slope;   % float roi_scale;     /* 72 + 4 */
	%       float scl_inter;   % float funused1;      /* 76 + 4 */
	%       short slice_end;   % float funused2;      /* 80 + 2 */
	%       char slice_code;   % float funused2;      /* 82 + 1 */
	%       char xyzt_units;   % float funused2;      /* 83 + 1 */
	%       float cal_max;                   /* 84 + 4          */
	%       float cal_min;                   /* 88 + 4          */
	%       float slice_duration;   % int compressed; /* 92 + 4 */
	%       float toffset;   % int verified;          /* 96 + 4 */
	%       int glmax;                       /* 100 + 4         */
	%       int glmin;                       /* 104 + 4         */
	%       };                               /* total=108 bytes */
	
   fwrite(fid, dime.dim(1:8),        'int16');
   fwrite(fid, dime.intent_p1(1),  'float32');
   fwrite(fid, dime.intent_p2(1),  'float32');
   fwrite(fid, dime.intent_p3(1),  'float32');
   fwrite(fid, dime.intent_code(1),  'int16');
   fwrite(fid, dime.datatype(1),     'int16');
   fwrite(fid, dime.bitpix(1),       'int16');
   fwrite(fid, dime.slice_start(1),  'int16');
   fwrite(fid, dime.pixdim(1:8),   'float32');
   fwrite(fid, dime.vox_offset(1), 'float32');
   fwrite(fid, dime.scl_slope(1),  'float32');
   fwrite(fid, dime.scl_inter(1),  'float32');
   fwrite(fid, dime.slice_end(1),    'int16');
   fwrite(fid, dime.slice_code(1),   'uchar');
   fwrite(fid, dime.xyzt_units(1),   'uchar');
   fwrite(fid, dime.cal_max(1),    'float32');
   fwrite(fid, dime.cal_min(1),    'float32');
   fwrite(fid, dime.slice_duration(1), 'float32');
   fwrite(fid, dime.toffset(1),    'float32');
   fwrite(fid, dime.glmax(1),        'int32');
   fwrite(fid, dime.glmin(1),        'int32');
   
   return;					% image_dimension


%---------------------------------------------------------------------
function data_history(fid, hist)
    
	% Original header structures
	%struct data_history       
	%       {                                /* off + size      */
	%       char descrip[80];                /* 0 + 80          */
	%       char aux_file[24];               /* 80 + 24         */
	%       short int qform_code;            /* 104 + 2         */
	%       short int sform_code;            /* 106 + 2         */
	%       float quatern_b;                 /* 108 + 4         */
	%       float quatern_c;                 /* 112 + 4         */
	%       float quatern_d;                 /* 116 + 4         */
	%       float qoffset_x;                 /* 120 + 4         */
	%       float qoffset_y;                 /* 124 + 4         */
	%       float qoffset_z;                 /* 128 + 4         */
	%       float srow_x[4];                 /* 132 + 16        */
	%       float srow_y[4];                 /* 148 + 16        */
	%       float srow_z[4];                 /* 164 + 16        */
	%       char intent_name[16];            /* 180 + 16        */
	%       char magic[4];   % int smin;     /* 196 + 4         */
	%       };                               /* total=200 bytes */
	
   % descrip     = sprintf('%-80s', hist.descrip);     % 80 chars from left
   % fwrite(fid, descrip(1:80),    'uchar');
   pad = zeros(1, 80-length(hist.descrip));
   hist.descrip = [hist.descrip  char(pad)];
   fwrite(fid, hist.descrip(1:80), 'uchar');
    
   % aux_file    = sprintf('%-24s', hist.aux_file);    % 24 chars from left
   % fwrite(fid, aux_file(1:24),   'uchar');
   pad = zeros(1, 24-length(hist.aux_file));
   hist.aux_file = [hist.aux_file  char(pad)];
   fwrite(fid, hist.aux_file(1:24), 'uchar');
    
   fwrite(fid, hist.qform_code,    'int16');
   fwrite(fid, hist.sform_code,    'int16');
   fwrite(fid, hist.quatern_b,   'float32');
   fwrite(fid, hist.quatern_c,   'float32');
   fwrite(fid, hist.quatern_d,   'float32');
   fwrite(fid, hist.qoffset_x,   'float32');
   fwrite(fid, hist.qoffset_y,   'float32');
   fwrite(fid, hist.qoffset_z,   'float32');
   fwrite(fid, hist.srow_x(1:4), 'float32');
   fwrite(fid, hist.srow_y(1:4), 'float32');
   fwrite(fid, hist.srow_z(1:4), 'float32');

   % intent_name = sprintf('%-16s', hist.intent_name);	% 16 chars from left
   % fwrite(fid, intent_name(1:16),    'uchar');
   pad = zeros(1, 16-length(hist.intent_name));
   hist.intent_name = [hist.intent_name  char(pad)];
   fwrite(fid, hist.intent_name(1:16), 'uchar');
    
   % magic	= sprintf('%-4s', hist.magic);		% 4 chars from left
   % fwrite(fid, magic(1:4),           'uchar');
   pad = zeros(1, 4-length(hist.magic));
   hist.magic = [hist.magic  char(pad)];
   fwrite(fid, hist.magic(1:4),        'uchar');
    
   return;					% data_history

