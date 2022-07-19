%  internal function

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)

function hdr = load_nii_hdr(fileprefix, machine)

   fn = sprintf('%s.hdr',fileprefix);
   fid = fopen(fn,'r',machine);
    
   if fid < 0,
      msg = sprintf('Cannot open file %s.',fn);
      error(msg);
   else
      fseek(fid,0,'bof');
      hdr = read_header(fid);
      fclose(fid);
   end

   return					% load_nii_hdr


%---------------------------------------------------------------------
function [ dsr ] = read_header(fid)

        %  Original header structures
	%  struct dsr
	%       { 
	%       struct header_key hk;            /*   0 +  40       */
	%       struct image_dimension dime;     /*  40 + 108       */
	%       struct data_history hist;        /* 148 + 200       */
	%       };                               /* total= 348 bytes*/

    dsr.hk   = header_key(fid);
    dsr.dime = image_dimension(fid);
    dsr.hist = data_history(fid);

    return					% read_header


%---------------------------------------------------------------------
function [ hk ] = header_key(fid)

    fseek(fid,0,'bof');
    
	%  Original header structures	
	%  struct header_key                     /* header key      */ 
	%       {                                /* off + size      */
	%       int sizeof_hdr                   /*  0 +  4         */
	%       char data_type[10];              /*  4 + 10         */
	%       char db_name[18];                /* 14 + 18         */
	%       int extents;                     /* 32 +  4         */
	%       short int session_error;         /* 36 +  2         */
	%       char regular;                    /* 38 +  1         */
	%       char hkey_un0;                   /* 39 +  1 */
	%       };                               /* total=40 bytes  */
	%
	% int sizeof_header   Should be 348.
	% char regular        Must be 'r' to indicate that all images and 
	%                     volumes are the same size. 

    v6 = version;
    if str2num(v6(1))<6
       directchar = '*char';
    else
       directchar = 'uchar=>char';
    end
	
    hk.sizeof_hdr    = fread(fid, 1,'int32')';	% should be 348!
    hk.data_type     = deblank(fread(fid,10,directchar)');
    hk.db_name       = deblank(fread(fid,18,directchar)');
    hk.extents       = fread(fid, 1,'int32')';
    hk.session_error = fread(fid, 1,'int16')';
    hk.regular       = fread(fid, 1,directchar)';
    hk.hkey_un0      = fread(fid, 1,directchar)';
    
    return					% header_key


%---------------------------------------------------------------------
function [ dime ] = image_dimension(fid)

	%struct image_dimension
	%       {                                /* off + size      */
	%       short int dim[8];                /* 0 + 16          */
    %           /*
    %           dim[0]      Number of dimensions in database; usually 4. 
    %           dim[1]      Image X dimension;  number of *pixels* in an image row. 
    %           dim[2]      Image Y dimension;  number of *pixel rows* in slice. 
    %           dim[3]      Volume Z dimension; number of *slices* in a volume. 
    %           dim[4]      Time points; number of volumes in database
    %           */
	%       char vox_units[4];               /* 16 + 4          */
	%       char cal_units[8];               /* 20 + 8          */
	%       short int unused1;               /* 28 + 2          */
	%       short int datatype;              /* 30 + 2          */
	%       short int bitpix;                /* 32 + 2          */
	%       short int dim_un0;               /* 34 + 2          */
	%       float pixdim[8];                 /* 36 + 32         */
	%			/*
	%				pixdim[] specifies the voxel dimensions:
	%				pixdim[1] - voxel width, mm
	%				pixdim[2] - voxel height, mm
	%				pixdim[3] - slice thickness, mm
    %               pixdim[4] - volume timing, in msec
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

    v6 = version;
    if str2num(v6(1))<6
       directchar = '*char';
    else
       directchar = 'uchar=>char';
    end
	
    dime.dim        = fread(fid,8,'int16')';
    dime.vox_units  = deblank(fread(fid,4,directchar)');
    dime.cal_units  = deblank(fread(fid,8,directchar)');
    dime.unused1    = fread(fid,1,'int16')';
    dime.datatype   = fread(fid,1,'int16')';
    dime.bitpix     = fread(fid,1,'int16')';
    dime.dim_un0    = fread(fid,1,'int16')';
    dime.pixdim     = fread(fid,8,'float32')';
    dime.vox_offset = fread(fid,1,'float32')';
    dime.roi_scale  = fread(fid,1,'float32')';
    dime.funused1   = fread(fid,1,'float32')';
    dime.funused2   = fread(fid,1,'float32')';
    dime.cal_max    = fread(fid,1,'float32')';
    dime.cal_min    = fread(fid,1,'float32')';
    dime.compressed = fread(fid,1,'int32')';
    dime.verified   = fread(fid,1,'int32')';
    dime.glmax      = fread(fid,1,'int32')';
    dime.glmin      = fread(fid,1,'int32')';
        
    return					% image_dimension


%---------------------------------------------------------------------
function [ hist ] = data_history(fid)
        
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

    v6 = version;
    if str2num(v6(1))<6
       directchar = '*char';
    else
       directchar = 'uchar=>char';
    end

    hist.descrip     = deblank(fread(fid,80,directchar)');
    hist.aux_file    = deblank(fread(fid,24,directchar)');
    hist.orient      = fread(fid, 1,'char')';
    hist.originator  = fread(fid, 5,'int16')';
    hist.generated   = deblank(fread(fid,10,directchar)');
    hist.scannum     = deblank(fread(fid,10,directchar)');
    hist.patient_id  = deblank(fread(fid,10,directchar)');
    hist.exp_date    = deblank(fread(fid,10,directchar)');
    hist.exp_time    = deblank(fread(fid,10,directchar)');
    hist.hist_un0    = deblank(fread(fid, 3,directchar)');
    hist.views       = fread(fid, 1,'int32')';
    hist.vols_added  = fread(fid, 1,'int32')';
    hist.start_field = fread(fid, 1,'int32')';
    hist.field_skip  = fread(fid, 1,'int32')';
    hist.omax        = fread(fid, 1,'int32')';
    hist.omin        = fread(fid, 1,'int32')';
    hist.smax        = fread(fid, 1,'int32')';
    hist.smin        = fread(fid, 1,'int32')';
    
    return					% data_history

