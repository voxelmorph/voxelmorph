%  Return time frame of a NIFTI dataset. Support both *.nii and 
%  *.hdr/*.img file extension. If file extension is not provided,
%  *.hdr/*.img will be used as default. 
%
%  It is a lightweighted "load_nii_hdr", and is equivalent to
%  hdr.dime.dim(5)
%  
%  Usage: [ total_scan ] = get_nii_frame(filename)
%
%  filename - NIFTI file name.
%
%  Returned values:
%
%  total_scan - total number of image scans for the time frame
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function [ total_scan ] = get_nii_frame(filename)

   if ~exist('filename','var'),
      error('Usage: [ total_scan ] = get_nii_frame(filename)');
   end

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
      elseif strcmp(filename(end-6:end), '.img.gz')
         filename1 = filename;
         filename2 = filename;
         filename2(end-6:end) = '';
         filename2 = [filename2, '.hdr.gz'];

         tmpDir = tempname;
         mkdir(tmpDir);
         gzFileName = filename;

         filename1 = gunzip(filename1, tmpDir);
         filename2 = gunzip(filename2, tmpDir);
         filename = char(filename1);	% convert from cell to string
      elseif strcmp(filename(end-6:end), '.hdr.gz')
         filename1 = filename;
         filename2 = filename;
         filename2(end-6:end) = '';
         filename2 = [filename2, '.img.gz'];

         tmpDir = tempname;
         mkdir(tmpDir);
         gzFileName = filename;

         filename1 = gunzip(filename1, tmpDir);
         filename2 = gunzip(filename2, tmpDir);
         filename = char(filename1);	% convert from cell to string
      elseif strcmp(filename(end-6:end), '.nii.gz')
         tmpDir = tempname;
         mkdir(tmpDir);
         gzFileName = filename;
         filename = gunzip(filename, tmpDir);
         filename = char(filename);	% convert from cell to string
      end
   end

   fileprefix = filename;
   machine = 'ieee-le';
   new_ext = 0;

   if findstr('.nii',fileprefix) & strcmp(fileprefix(end-3:end), '.nii')
      new_ext = 1;
      fileprefix(end-3:end)='';
   end

   if findstr('.hdr',fileprefix) & strcmp(fileprefix(end-3:end), '.hdr')
      fileprefix(end-3:end)='';
   end

   if findstr('.img',fileprefix) & strcmp(fileprefix(end-3:end), '.img')
      fileprefix(end-3:end)='';
   end

   if new_ext
      fn = sprintf('%s.nii',fileprefix);

      if ~exist(fn)
         msg = sprintf('Cannot find file "%s.nii".', fileprefix);
         error(msg);
      end
   else
      fn = sprintf('%s.hdr',fileprefix);

      if ~exist(fn)
         msg = sprintf('Cannot find file "%s.hdr".', fileprefix);
         error(msg);
      end
   end

   fid = fopen(fn,'r',machine);
    
   if fid < 0,
      msg = sprintf('Cannot open file %s.',fn);
      error(msg);
   else
      hdr = read_header(fid);
      fclose(fid);
   end
   
   if hdr.sizeof_hdr ~= 348
      % first try reading the opposite endian to 'machine'
      switch machine,
      case 'ieee-le', machine = 'ieee-be';
      case 'ieee-be', machine = 'ieee-le';
      end
        
      fid = fopen(fn,'r',machine);
        
      if fid < 0,
         msg = sprintf('Cannot open file %s.',fn);
         error(msg);
      else
         hdr = read_header(fid);
         fclose(fid);
      end
   end

   if hdr.sizeof_hdr ~= 348
      % Now throw an error
      msg = sprintf('File "%s" is corrupted.',fn);
      error(msg);
   end

   total_scan = hdr.dim(5);

   %  Clean up after gunzip
   %
   if exist('gzFileName', 'var')
      rmdir(tmpDir,'s');
   end

   return;					% get_nii_frame


%---------------------------------------------------------------------
function [ dsr ] = read_header(fid)

    fseek(fid,0,'bof');
    dsr.sizeof_hdr    = fread(fid,1,'int32')';  % should be 348!

    fseek(fid,40,'bof');
    dsr.dim           = fread(fid,8,'int16')';

    return;					% read_header

