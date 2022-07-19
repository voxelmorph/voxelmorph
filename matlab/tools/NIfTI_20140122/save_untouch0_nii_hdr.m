%  internal function

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)

function save_nii_hdr(hdr, fid)

   if ~isequal(hdr.hk.sizeof_hdr,348),
      error('hdr.hk.sizeof_hdr must be 348.');
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
	%       char hkey_un0;                   /* 39 +  1 */
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
   fwrite(fid, hk.regular(1),       'uchar');

   fwrite(fid, hk.hkey_un0(1),    'uchar');
    
   return;					% header_key


%---------------------------------------------------------------------
function image_dimension(fid, dime)

	%struct image_dimension
	%       {                                /* off + size      */
	%       short int dim[8];                /* 0 + 16          */
	%       char vox_units[4];               /* 16 + 4          */
	%       char cal_units[8];               /* 20 + 8          */
	%       short int unused1;               /* 28 + 2          */
	%       short int datatype;              /* 30 + 2          */
	%       short int bitpix;                /* 32 + 2          */
	%       short int dim_un0;               /* 34 + 2          */
	%       float pixdim[8];                 /* 36 + 32         */
	%			/*
	%				pixdim[] specifies the voxel dimensions:
	%				pixdim[1] - voxel width
	%				pixdim[2] - voxel height
	%				pixdim[3] - interslice distance
	%					..etc
	%			*/
	%       float vox_offset;                /* 68 + 4          */
	%       float roi_scale;                 /* 72 + 4          */
	%       float funused1;                  /* 76 + 4          */
	%       float funused2;                  /* 80 + 4          */
	%       float cal_max;                   /* 84 + 4          */
	%       float cal_min;                   /* 88 + 4          */
	%       int compressed;                  /* 92 + 4          */
	%       int verified;                    /* 96 + 4          */
	%       int glmax;                       /* 100 + 4         */
	%       int glmin;                       /* 104 + 4         */
	%       };                               /* total=108 bytes */
	
   fwrite(fid, dime.dim(1:8),      'int16');

   pad = zeros(1, 4-length(dime.vox_units));
   dime.vox_units = [dime.vox_units  char(pad)];
   fwrite(fid, dime.vox_units(1:4),  'uchar');

   pad = zeros(1, 8-length(dime.cal_units));
   dime.cal_units = [dime.cal_units  char(pad)];
   fwrite(fid, dime.cal_units(1:8),  'uchar');

   fwrite(fid, dime.unused1(1),    'int16');
   fwrite(fid, dime.datatype(1),   'int16');
   fwrite(fid, dime.bitpix(1),     'int16');
   fwrite(fid, dime.dim_un0(1),    'int16');
   fwrite(fid, dime.pixdim(1:8),   'float32');
   fwrite(fid, dime.vox_offset(1), 'float32');
   fwrite(fid, dime.roi_scale(1),  'float32');
   fwrite(fid, dime.funused1(1),   'float32');
   fwrite(fid, dime.funused2(1),   'float32');
   fwrite(fid, dime.cal_max(1),    'float32');
   fwrite(fid, dime.cal_min(1),    'float32');
   fwrite(fid, dime.compressed(1), 'int32');
   fwrite(fid, dime.verified(1),   'int32');
   fwrite(fid, dime.glmax(1),      'int32');
   fwrite(fid, dime.glmin(1),      'int32');
   
   return;					% image_dimension


%---------------------------------------------------------------------
function data_history(fid, hist)
    
	% Original header structures - ANALYZE 7.5
	%struct data_history       
	%       {                                /* off + size      */
	%       char descrip[80];                /* 0 + 80          */
	%       char aux_file[24];               /* 80 + 24         */
	%       char orient;                     /* 104 + 1         */
	%       char originator[10];             /* 105 + 10        */
	%       char generated[10];              /* 115 + 10        */
	%       char scannum[10];                /* 125 + 10        */
	%       char patient_id[10];             /* 135 + 10        */
	%       char exp_date[10];               /* 145 + 10        */
	%       char exp_time[10];               /* 155 + 10        */
	%       char hist_un0[3];                /* 165 + 3         */
	%       int views                        /* 168 + 4         */
	%       int vols_added;                  /* 172 + 4         */
	%       int start_field;                 /* 176 + 4         */
	%       int field_skip;                  /* 180 + 4         */
	%       int omax;                        /* 184 + 4         */
	%       int omin;                        /* 188 + 4         */
	%       int smax;                        /* 192 + 4         */
	%       int smin;                        /* 196 + 4         */
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

   fwrite(fid, hist.orient(1),      'uchar');
   fwrite(fid, hist.originator(1:5), 'int16');

   pad = zeros(1, 10-length(hist.generated));
   hist.generated = [hist.generated  char(pad)];
   fwrite(fid, hist.generated(1:10),  'uchar');

   pad = zeros(1, 10-length(hist.scannum));
   hist.scannum = [hist.scannum  char(pad)];
   fwrite(fid, hist.scannum(1:10),  'uchar');

   pad = zeros(1, 10-length(hist.patient_id));
   hist.patient_id = [hist.patient_id  char(pad)];
   fwrite(fid, hist.patient_id(1:10),  'uchar');

   pad = zeros(1, 10-length(hist.exp_date));
   hist.exp_date = [hist.exp_date  char(pad)];
   fwrite(fid, hist.exp_date(1:10),  'uchar');

   pad = zeros(1, 10-length(hist.exp_time));
   hist.exp_time = [hist.exp_time  char(pad)];
   fwrite(fid, hist.exp_time(1:10),  'uchar');

   pad = zeros(1, 3-length(hist.hist_un0));
   hist.hist_un0 = [hist.hist_un0  char(pad)];
   fwrite(fid, hist.hist_un0(1:3),  'uchar');

   fwrite(fid, hist.views(1),      'int32');
   fwrite(fid, hist.vols_added(1), 'int32');
   fwrite(fid, hist.start_field(1),'int32');
   fwrite(fid, hist.field_skip(1), 'int32');
   fwrite(fid, hist.omax(1),       'int32');
   fwrite(fid, hist.omin(1),       'int32');
   fwrite(fid, hist.smax(1),       'int32');
   fwrite(fid, hist.smin(1),       'int32');
    
   return;					% data_history

