%  Decode extra NIFTI header information into hdr.extra
%
%  Usage: hdr = extra_nii_hdr(hdr)
%
%  hdr can be obtained from load_nii_hdr
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function hdr = extra_nii_hdr(hdr)

   switch hdr.dime.datatype
   case 1
      extra.NIFTI_DATATYPES = 'DT_BINARY';
   case 2
      extra.NIFTI_DATATYPES = 'DT_UINT8';
   case 4
      extra.NIFTI_DATATYPES = 'DT_INT16';
   case 8
      extra.NIFTI_DATATYPES = 'DT_INT32';
   case 16
      extra.NIFTI_DATATYPES = 'DT_FLOAT32';
   case 32
      extra.NIFTI_DATATYPES = 'DT_COMPLEX64';
   case 64
      extra.NIFTI_DATATYPES = 'DT_FLOAT64';
   case 128
      extra.NIFTI_DATATYPES = 'DT_RGB24';
   case 256
      extra.NIFTI_DATATYPES = 'DT_INT8';
   case 512
      extra.NIFTI_DATATYPES = 'DT_UINT16';
   case 768
      extra.NIFTI_DATATYPES = 'DT_UINT32';
   case 1024
      extra.NIFTI_DATATYPES = 'DT_INT64';
   case 1280
      extra.NIFTI_DATATYPES = 'DT_UINT64';
   case 1536
      extra.NIFTI_DATATYPES = 'DT_FLOAT128';
   case 1792
      extra.NIFTI_DATATYPES = 'DT_COMPLEX128';
   case 2048
      extra.NIFTI_DATATYPES = 'DT_COMPLEX256';
   otherwise
      extra.NIFTI_DATATYPES = 'DT_UNKNOWN';
   end

   switch hdr.dime.intent_code
   case 2
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_CORREL';
   case 3
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_TTEST';
   case 4
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_FTEST';
   case 5
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_ZSCORE';
   case 6
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_CHISQ';
   case 7
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_BETA';
   case 8
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_BINOM';
   case 9
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_GAMMA';
   case 10
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_POISSON';
   case 11
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_NORMAL';
   case 12
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_FTEST_NONC';
   case 13
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_CHISQ_NONC';
   case 14
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_LOGISTIC';
   case 15
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_LAPLACE';
   case 16
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_UNIFORM';
   case 17
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_TTEST_NONC';
   case 18
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_WEIBULL';
   case 19
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_CHI';
   case 20
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_INVGAUSS';
   case 21
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_EXTVAL';
   case 22
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_PVAL';
   case 23
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_LOGPVAL';
   case 24
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_LOG10PVAL';
   case 1001
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_ESTIMATE';
   case 1002
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_LABEL';
   case 1003
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_NEURONAME';
   case 1004
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_GENMATRIX';
   case 1005
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_SYMMATRIX';
   case 1006
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_DISPVECT';
   case 1007
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_VECTOR';
   case 1008
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_POINTSET';
   case 1009
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_TRIANGLE';
   case 1010
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_QUATERNION';
   case 1011
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_DIMLESS';
   otherwise
      extra.NIFTI_INTENT_CODES = 'NIFTI_INTENT_NONE';
   end

   extra.NIFTI_INTENT_NAMES = hdr.hist.intent_name;

   if hdr.hist.sform_code > 0
      switch hdr.hist.sform_code
      case 1
         extra.NIFTI_SFORM_CODES = 'NIFTI_XFORM_SCANNER_ANAT';
      case 2
         extra.NIFTI_SFORM_CODES = 'NIFTI_XFORM_ALIGNED_ANAT';
      case 3
         extra.NIFTI_SFORM_CODES = 'NIFTI_XFORM_TALAIRACH';
      case 4
         extra.NIFTI_SFORM_CODES = 'NIFTI_XFORM_MNI_152';
      otherwise
         extra.NIFTI_SFORM_CODES = 'NIFTI_XFORM_UNKNOWN';
      end

      extra.NIFTI_QFORM_CODES = 'NIFTI_XFORM_UNKNOWN';
   elseif hdr.hist.qform_code > 0
      extra.NIFTI_SFORM_CODES = 'NIFTI_XFORM_UNKNOWN';

      switch hdr.hist.qform_code
      case 1
         extra.NIFTI_QFORM_CODES = 'NIFTI_XFORM_SCANNER_ANAT';
      case 2
         extra.NIFTI_QFORM_CODES = 'NIFTI_XFORM_ALIGNED_ANAT';
      case 3
         extra.NIFTI_QFORM_CODES = 'NIFTI_XFORM_TALAIRACH';
      case 4
         extra.NIFTI_QFORM_CODES = 'NIFTI_XFORM_MNI_152';
      otherwise
         extra.NIFTI_QFORM_CODES = 'NIFTI_XFORM_UNKNOWN';
      end
   else
      extra.NIFTI_SFORM_CODES = 'NIFTI_XFORM_UNKNOWN';
      extra.NIFTI_QFORM_CODES = 'NIFTI_XFORM_UNKNOWN';
   end

   switch bitand(hdr.dime.xyzt_units, 7)	% mask with 0x07
   case 1
      extra.NIFTI_SPACE_UNIT = 'NIFTI_UNITS_METER';
   case 2
      extra.NIFTI_SPACE_UNIT = 'NIFTI_UNITS_MM';	% millimeter
   case 3
      extra.NIFTI_SPACE_UNIT = 'NIFTI_UNITS_MICRO';
   otherwise
      extra.NIFTI_SPACE_UNIT = 'NIFTI_UNITS_UNKNOWN';
   end

   switch bitand(hdr.dime.xyzt_units, 56)	% mask with 0x38
   case 8
      extra.NIFTI_TIME_UNIT = 'NIFTI_UNITS_SEC';
   case 16
      extra.NIFTI_TIME_UNIT = 'NIFTI_UNITS_MSEC';
   case 24
      extra.NIFTI_TIME_UNIT = 'NIFTI_UNITS_USEC';	% microsecond
   otherwise
      extra.NIFTI_TIME_UNIT = 'NIFTI_UNITS_UNKNOWN';
   end

   switch hdr.dime.xyzt_units
   case 32
      extra.NIFTI_SPECTRAL_UNIT = 'NIFTI_UNITS_HZ';
   case 40
      extra.NIFTI_SPECTRAL_UNIT = 'NIFTI_UNITS_PPM';	% part per million
   case 48
      extra.NIFTI_SPECTRAL_UNIT = 'NIFTI_UNITS_RADS';	% radians per second
   otherwise
      extra.NIFTI_SPECTRAL_UNIT = 'NIFTI_UNITS_UNKNOWN';
   end

   %  MRI-specific spatial and temporal information
   %
   dim_info = hdr.hk.dim_info;
   extra.NIFTI_FREQ_DIM = bitand(dim_info, 3);
   extra.NIFTI_PHASE_DIM = bitand(bitshift(dim_info, -2), 3);
   extra.NIFTI_SLICE_DIM = bitand(bitshift(dim_info, -4), 3);

   %  Check slice code
   %
   switch hdr.dime.slice_code
   case 1
      extra.NIFTI_SLICE_ORDER = 'NIFTI_SLICE_SEQ_INC';	% sequential increasing
   case 2
      extra.NIFTI_SLICE_ORDER = 'NIFTI_SLICE_SEQ_DEC';	% sequential decreasing
   case 3
      extra.NIFTI_SLICE_ORDER = 'NIFTI_SLICE_ALT_INC';	% alternating increasing
   case 4
      extra.NIFTI_SLICE_ORDER = 'NIFTI_SLICE_ALT_DEC';	% alternating decreasing
   case 5
      extra.NIFTI_SLICE_ORDER = 'NIFTI_SLICE_ALT_INC2';	% ALT_INC # 2
   case 6
      extra.NIFTI_SLICE_ORDER = 'NIFTI_SLICE_ALT_DEC2';	% ALT_DEC # 2
   otherwise
      extra.NIFTI_SLICE_ORDER = 'NIFTI_SLICE_UNKNOWN';
   end

   %  Check NIFTI version
   %
   if     ~isempty(hdr.hist.magic) & strcmp(hdr.hist.magic(1),'n') & ...
	( strcmp(hdr.hist.magic(2),'i') | strcmp(hdr.hist.magic(2),'+') ) & ...
	  str2num(hdr.hist.magic(3)) >= 1 & str2num(hdr.hist.magic(3)) <= 9

      extra.NIFTI_VERSION = str2num(hdr.hist.magic(3));
   else
      extra.NIFTI_VERSION = 0;
   end

   %  Check if data stored in the same file (*.nii) or separate
   %  files (*.hdr/*.img)
   %
   if isempty(hdr.hist.magic)
      extra.NIFTI_ONEFILE = 0;
   else
      extra.NIFTI_ONEFILE = strcmp(hdr.hist.magic(2), '+');
   end

   %  Swap has been taken care of by checking whether sizeof_hdr is
   %  348 (machine is 'ieee-le' or 'ieee-be' etc)
   %
   % extra.NIFTI_NEEDS_SWAP = (hdr.dime.dim(1) < 0 | hdr.dime.dim(1) > 7);

   %  Check NIFTI header struct contains a 5th (vector) dimension
   %
   if hdr.dime.dim(1) > 4 & hdr.dime.dim(6) > 1
      extra.NIFTI_5TH_DIM = hdr.dime.dim(6);
   else
      extra.NIFTI_5TH_DIM = 0;
   end

   hdr.extra = extra;

   return;					% extra_nii_hdr

