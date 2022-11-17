%  Undo the flipping and rotations performed by xform_nii; spit back only
%  the raw img data block. Initial cut will only deal with 3D volumes
%  strongly assume we have called xform_nii to write down the steps used 
%  in xform_nii.
%
%  Usage:  a = load_nii('original_name');
%          manipulate a.img to make array b;
%
%          if you use unxform_nii to un-tranform the image (img) data
%          block, then nii.original.hdr is the corresponding header.
%
%          nii.original.img = unxform_nii(a, b);
%          save_nii(nii.original,'newname');
%
%  Where, 'newname' is created with data in the same space as the
%         original_name data    
%
%  - Jeff Gunter, 26-JUN-06
%
function outblock = unxform_nii(nii, inblock)
  
   if isempty(nii.hdr.hist.rot_orient)     
      outblock=inblock;
   else
      [dummy unrotate_orient] = sort(nii.hdr.hist.rot_orient);
      outblock = permute(inblock, unrotate_orient);
   end

   if ~isempty(nii.hdr.hist.flip_orient)
      flip_orient = nii.hdr.hist.flip_orient(unrotate_orient);

      for i = 1:3
         if flip_orient(i)
            outblock = flipdim(outblock, i);
         end
      end
   end;

   return;

