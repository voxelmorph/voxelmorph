%  Convert image of different orientations to standard Analyze orientation
%
%  Usage: nii = rri_orient(nii);

%  Jimmy Shen (jimmy@rotman-baycrest.on.ca), 26-APR-04
%___________________________________________________________________

function [nii, orient, pattern] = rri_orient(nii, varargin)

   if nargin > 1
      pattern = varargin{1};
   else
      pattern = [];
   end

   if(nargin > 2)
       orient = varargin{2};
       if(length(find(orient>6)) || length(find(orient<1))) %value checking
           orient=[1 2 3]; %set  to default if bogus values set
       end
   else
       orient = [1 2 3];
   end
   
   
   dim = double(nii.hdr.dime.dim([2:4]));

   if ~isempty(pattern) & ~isequal(length(pattern), prod(dim))
      return;
   end

   %  get orient of the current image
   %
   if isequal(orient, [1 2 3])
    orient = rri_orient_ui;
    pause(.1);
   end

   %  no need for conversion
   %
   if isequal(orient, [1 2 3])
      return;
   end

   if isempty(pattern)
      pattern = 1:prod(dim);
   end

   pattern = reshape(pattern, dim);
   img = nii.img;

   %  calculate after flip orient
   %
   rot_orient = mod(orient + 2, 3) + 1;

   %  do flip:
   %
   flip_orient = orient - rot_orient;

   for i = 1:3
      if flip_orient(i)
         pattern = flipdim(pattern, i);
         img = flipdim(img, i);
      end
   end

   %  get index of orient (do inverse)
   %
   [tmp rot_orient] = sort(rot_orient);

   %  do rotation:
   %
   pattern = permute(pattern, rot_orient);
   img = permute(img, [rot_orient 4 5 6]);

   %  rotate resolution, or 'dim'
   %
   new_dim = nii.hdr.dime.dim([2:4]);
   new_dim = new_dim(rot_orient);
   nii.hdr.dime.dim([2:4]) = new_dim;

   %  rotate voxel_size, or 'pixdim'
   %
   tmp = nii.hdr.dime.pixdim([2:4]);
   tmp = tmp(rot_orient);
   nii.hdr.dime.pixdim([2:4]) = tmp;

   %  re-calculate originator
   %
   tmp = nii.hdr.hist.originator([1:3]);
   tmp = tmp(rot_orient);
   flip_orient = flip_orient(rot_orient);

   for i = 1:3
      if flip_orient(i) & ~isequal(double(tmp(i)), 0)
         tmp(i) = int16(double(new_dim(i)) - double(tmp(i)) + 1);
      end
   end

   nii.hdr.hist.originator([1:3]) = tmp;

   nii.img = img;
   pattern = pattern(:);

   return;						% rri_orient

