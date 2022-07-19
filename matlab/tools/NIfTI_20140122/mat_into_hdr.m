%MAT_INTO_HDR  The old versions of SPM (any version before SPM5) store
%	an affine matrix of the SPM Reoriented image into a matlab file 
%	(.mat extension). The file name of this SPM matlab file is the
%	same as the SPM Reoriented image file (.img/.hdr extension).
%
%	This program will convert the ANALYZE 7.5 SPM Reoriented image
%	file into NIfTI format, and integrate the affine matrix in the
%	SPM matlab file into its header file (.hdr extension).
%
%	WARNING: Before you run this program, please save the header
%	file (.hdr extension) into another file name or into another
%	folder location, because all header files (.hdr extension)
%	will be overwritten after they are converted into NIfTI
%	format.
%
%  Usage: mat_into_hdr(filename);
%
%  filename:	file name(s) with .hdr or .mat file extension, like:
%		'*.hdr', or '*.mat', or a single .hdr or .mat file.
%	e.g.	mat_into_hdr('T1.hdr')
%		mat_into_hdr('*.mat')
%

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
%-------------------------------------------------------------------------
function mat_into_hdr(files)

   pn = fileparts(files);
   file_lst = dir(files);
   file_lst = {file_lst.name};
   file1 = file_lst{1};
   [p n e]= fileparts(file1);

   for i=1:length(file_lst)
      [p n e]= fileparts(file_lst{i});
      disp(['working on file ', num2str(i) ,' of ', num2str(length(file_lst)), ': ', n,e]);
      process=1;

      if isequal(e,'.hdr')
         mat=fullfile(pn, [n,'.mat']);
         hdr=fullfile(pn, file_lst{i});

         if ~exist(mat,'file')
            warning(['Cannot find file "',mat  , '". File "', n, e, '" will not be processed.']);
            process=0;
         end
      elseif isequal(e,'.mat')
         hdr=fullfile(pn, [n,'.hdr']);
         mat=fullfile(pn, file_lst{i});

         if ~exist(hdr,'file')
            warning(['Can not find file "',hdr  , '". File "', n, e, '" will not be processed.']);
            process=0;
         end
      else
         warning(['Input file must have .mat or .hdr extension. File "', n, e, '" will not be processed.']);
         process=0;
      end

      if process
         load(mat);
         R=M(1:3,1:3);
         T=M(1:3,4);
         T=R*ones(3,1)+T;
         M(1:3,4)=T;

         [h filetype fileprefix machine]=load_nii_hdr(hdr);
         h.hist.qform_code=0;
         h.hist.sform_code=1;
         h.hist.srow_x=M(1,:);
         h.hist.srow_y=M(2,:);
         h.hist.srow_z=M(3,:);
         h.hist.magic='ni1';

         fid = fopen(hdr,'w',machine);
         save_nii_hdr(h,fid);
         fclose(fid);
      end
   end

   return;				% mat_into_hdr

