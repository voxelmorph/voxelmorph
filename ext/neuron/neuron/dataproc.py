''' data processing for neuron project '''

# built-in
import sys
import os
import shutil
import six

# third party
import nibabel as nib
import numpy as np
import scipy.ndimage.interpolation
from tqdm import tqdm_notebook as tqdm # for verbosity for forloops
from PIL import Image
import matplotlib.pyplot as plt



# note sure if tqdm_notebook reverts back to 
try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except:
    from tqdm import tqdm as tqdm

from subprocess import call


# import local ndutils
import pynd.ndutils as nd
import re

from imp import reload
reload(nd)

# from imp import reload # for re-loading modules, since some of the modules are still in development
# reload(nd)


def proc_mgh_vols(inpath,
                  outpath,
                  ext='.mgz',
                  label_idx=None,
                  **kwargs): 
    ''' process mgh data from mgz format and save to numpy format

    1. load file
    2. normalize intensity
    3. resize
    4. save as python block

    TODO: check header info and such.?
    '''

    # get files in input directory
    files = [f for f in os.listdir(inpath) if f.endswith(ext)]

    # go through each file
    list_skipped_files = ()
    for fileidx in tqdm(range(len(files)), ncols=80):

        # load nifti volume
        volnii = nib.load(os.path.join(inpath, files[fileidx]))

        # get the data out
        vol_data = volnii.get_data().astype(float)

        if ('dim' in volnii.header) and volnii.header['dim'][4] > 1:
            vol_data = vol_data[:, :, :, -1]

        # process volume
        try:
            vol_data = vol_proc(vol_data, **kwargs)
        except Exception as e:
            list_skipped_files += (files[fileidx], )
            print("Skipping %s\nError: %s" % (files[fileidx], str(e)), file=sys.stderr)
            continue

        if label_idx is not None:
            vol_data = (vol_data == label_idx).astype(int)

        # save numpy file
        outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[0] + '.npz'
        np.savez_compressed(outname, vol_data=vol_data)

    for file in list_skipped_files:
        print("Skipped: %s" % file, file=sys.stderr)


def scans_to_slices(inpath, outpath, slice_nrs,
                    ext='.mgz',
                    label_idx=None,
                    dim_idx=2,
                    out_ext='.png',
                    slice_pad=0,
                    vol_inner_pad_for_slice_nrs=0,
                    **kwargs):  # vol_proc args

    # get files in input directory
    files = [f for f in os.listdir(inpath) if f.endswith(ext)]

    # go through each file
    list_skipped_files = ()
    for fileidx in tqdm(range(len(files)), ncols=80):

        # load nifti volume
        volnii = nib.load(os.path.join(inpath, files[fileidx]))

        # get the data out
        vol_data = volnii.get_data().astype(float)

        if ('dim' in volnii.header) and volnii.header['dim'][4] > 1:
            vol_data = vol_data[:, :, :, -1]

        if slice_pad > 0:
            assert (out_ext != '.png'), "slice pad can only be used with volumes"

        # process volume
        try:
            vol_data = vol_proc(vol_data, **kwargs)
        except Exception as e:
            list_skipped_files += (files[fileidx], )
            print("Skipping %s\nError: %s" % (files[fileidx], str(e)), file=sys.stderr)
            continue
            
        mult_fact = 255
        if label_idx is not None:
            vol_data = (vol_data == label_idx).astype(int)
            mult_fact = 1

        # extract slice
        if slice_nrs is None:
            slice_nrs_sel = range(vol_inner_pad_for_slice_nrs+slice_pad, vol_data.shape[dim_idx]-slice_pad-vol_inner_pad_for_slice_nrs)
        else:
            slice_nrs_sel = slice_nrs

        for slice_nr in slice_nrs_sel:
            slice_nr_out = range(slice_nr - slice_pad, slice_nr + slice_pad + 1)
            if dim_idx == 2:  # TODO: fix in one line
                vol_img = np.squeeze(vol_data[:, :, slice_nr_out])
            elif dim_idx == 1:
                vol_img = np.squeeze(vol_data[:, slice_nr_out, :])
            else:
                vol_img = np.squeeze(vol_data[slice_nr_out, :, :])
           
            # save file
            if out_ext == '.png':
                # save png file
                img = (vol_img*mult_fact).astype('uint8')
                outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[0] + '_slice%d.png' % slice_nr
                Image.fromarray(img).convert('RGB').save(outname)
            else:
                if slice_pad == 0:  # dimenion has collapsed
                    assert vol_img.ndim == 2
                    vol_img = np.expand_dims(vol_img, dim_idx)
                # assuming nibabel saving image
                nii = nib.Nifti1Image(vol_img, np.diag([1,1,1,1]))
                outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[0] + '_slice%d.nii.gz' % slice_nr
                nib.save(nii, outname)


def vol_proc(vol_data,
             crop=None,
             resize_shape=None, # None (to not resize), or vector. If vector, third entry can be None
             interp_order=None,
             rescale=None,
             rescale_prctle=None,
             resize_slices=None,
             resize_slices_dim=None,
             offset=None,
             clip=None,
             extract_nd=None,  # extracts a particular section
             force_binary=None,  # forces anything > 0 to be 1
             permute=None):
    ''' process a volume with a series of intensity rescale, resize and crop rescale'''

    if offset is not None:
        vol_data = vol_data + offset

    # intensity normalize data .* rescale
    if rescale is not None:
        vol_data = np.multiply(vol_data, rescale)

    if rescale_prctle is not None:
        # print("max:", np.max(vol_data.flat))
        # print("test")
        rescale = np.percentile(vol_data.flat, rescale_prctle)
        # print("rescaling by 1/%f" % (rescale))
        vol_data = np.multiply(vol_data.astype(float), 1/rescale)

    if resize_slices is not None:
        resize_slices = [*resize_slices]
        assert resize_shape is None, "if resize_slices is given, resize_shape has to be None"
        resize_shape = resize_slices
        if resize_slices_dim is None:
            resize_slices_dim = np.where([f is None for f in resize_slices])[0]
            assert len(resize_slices_dim) == 1, "Could not find dimension or slice resize"
            resize_slices_dim = resize_slices_dim[0]
        resize_shape[resize_slices_dim] = vol_data.shape[resize_slices_dim]

    # resize (downsample) matrices
    if resize_shape is not None and resize_shape != vol_data.shape:
        resize_shape = [*resize_shape]
        # allow for the last entry to be None
        if resize_shape[-1] is None:
            resize_ratio = np.divide(resize_shape[0], vol_data.shape[0])
            resize_shape[-1] = np.round(resize_ratio * vol_data.shape[-1]).astype('int')
        resize_ratio = np.divide(resize_shape, vol_data.shape)
        vol_data = scipy.ndimage.interpolation.zoom(vol_data, resize_ratio, order=interp_order)

    # crop data if necessary
    if crop is not None:
        vol_data = nd.volcrop(vol_data, crop=crop)

    # needs to be last to guarantee clip limits.
    # For e.g., resize might screw this up due to bicubic interpolation if it was done after.
    if clip is not None:
        vol_data = np.clip(vol_data, clip[0], clip[1])

    if extract_nd is not None:
        vol_data = vol_data[np.ix_(*extract_nd)]

    if force_binary:
        vol_data = (vol_data > 0).astype(float)

    # return with checks. this check should be right at the end before rturn
    if clip is not None:
        assert np.max(vol_data) <= clip[1], "clip failed"
        assert np.min(vol_data) >= clip[0], "clip failed"
    return vol_data


def prior_to_weights(prior_filename, nargout=1, min_freq=0, force_binary=False, verbose=False):
    
    ''' transform a 4D prior (3D + nb_labels) into a class weight vector '''

    # load prior
    if isinstance(prior_filename, six.string_types):
        prior = np.load(prior_filename)['prior']
    else:
        prior = prior_filename

    # assumes prior is 4D.
    assert np.ndim(prior) == 4 or np.ndim(prior) == 3, "prior is the wrong number of dimensions"
    prior_flat = np.reshape(prior, (np.prod(prior.shape[0:(np.ndim(prior)-1)]), prior.shape[-1]))

    if force_binary:
        nb_labels = prior_flat.shape[-1]
        prior_flat[:, 1] = np.sum(prior_flat[:, 1:nb_labels], 1)
        prior_flat = np.delete(prior_flat, range(2, nb_labels), 1)

    # sum total class votes
    class_count = np.sum(prior_flat, 0)
    class_prior = class_count / np.sum(class_count)

    # adding minimum frequency
    class_prior[class_prior < min_freq] = min_freq
    class_prior = class_prior / np.sum(class_prior)

    if np.any(class_prior == 0):
        print("Warning, found a label with 0 support. Setting its weight to 0!", file=sys.stderr)
        class_prior[class_prior == 0] = np.inf

    # compute weights from class frequencies
    weights = 1/class_prior
    weights = weights / np.sum(weights)
    # weights[0] = 0 # explicitly don't care about bg

    # a bit of verbosity
    if verbose:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.bar(range(prior.size), np.log(prior))
        ax1.set_title('log class freq')
        ax2.bar(range(weights.size), weights)
        ax2.set_title('weights')
        ax3.bar(range(weights.size), np.log((weights))-np.min(np.log((weights))))
        ax3.set_title('log(weights)-minlog')
        f.set_size_inches(12, 3)
        plt.show()
        np.set_printoptions(precision=3)

    # return
    if nargout == 1:
        return weights
    else:
        return (weights, prior)




def filestruct_change(in_path, out_path, re_map,
                      mode='subj_to_type',
                      use_symlinks=False, name=""):
    """
    change from independent subjects in a folder to breakdown structure 

    example: filestruct_change('/../in_path', '/../out_path', {'asegs.nii.gz':'asegs', 'norm.nii.gz':'vols'})


    input structure: 
        /.../in_path/subj_1 --> with files that match regular repressions defined in re_map.keys()
        /.../in_path/subj_2 --> with files that match regular repressions defined in re_map.keys()
        ...
    output structure:
        /.../out_path/asegs/subj_1.nii.gz, subj_2.nii.gz
        /.../out_path/vols/subj_1.nii.gz, subj_2.nii.gz

    Parameters:
        in_path (string): input path
        out_path (string): output path
        re_map (dictionary): keys are reg-exs that match files in the input folders. 
            values are the folders to put those files in the new structure. 
            values can also be tuples, in which case values[0] is the dst folder, 
            and values[1] is the extension of the output file
        mode (optional)
        use_symlinks (bool): whether to just use symlinks rather than copy files
            default:True
    """

    
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # go through folders
    for subj in tqdm(os.listdir(in_path), desc=name):

        # go through files in a folder
        files = os.listdir(os.path.join(in_path, subj))
        for file in files:

            # see which key matches. Make sure only one does.
            matches = [re.match(k, file) for k in re_map.keys()]
            nb_matches = sum([f is not None for f in matches])
            assert nb_matches == 1, "Found %d matches for file %s/%s" %(nb_matches, file, subj)

            # get the matches key
            match_idx = [i for i,f in enumerate(matches) if f is not None][0]
            matched_dst = re_map[list(re_map.keys())[match_idx]]
            _, ext = os.path.splitext(file)
            if isinstance(matched_dst, tuple):
                ext = matched_dst[1]
                matched_dst = matched_dst[0]

            # prepare source and destination file
            src_file = os.path.join(in_path, subj, file)
            dst_path = os.path.join(out_path, matched_dst)
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            dst_file = os.path.join(dst_path, subj + ext)

            if use_symlinks:
                # on windows there are permission problems. 
                # Can try : call(['mklink', 'LINK', 'TARGET'], shell=True)
                # or note https://stackoverflow.com/questions/6260149/os-symlink-support-in-windows
                os.symlink(src_file, dst_file)

            else:
                shutil.copyfile(src_file, dst_file)


def ml_split(in_path, out_path,
             cat_titles=['train', 'validate', 'test'],
             cat_prop=[0.5, 0.3, 0.2],
             use_symlinks=False,
             seed=None,
             tqdm=tqdm):
    """
    split dataset 
    """

    if seed is not None:
        np.random.seed(seed)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # get subjects and randomize their order
    subjs = sorted(os.listdir(in_path))
    nb_subj = len(subjs)
    subj_order = np.random.permutation(nb_subj)

    # prepare split
    cat_tot = np.cumsum(cat_prop)
    if not cat_tot[-1] == 1:
        print("split_prop sums to %f, re-normalizing" % cat_tot)
        cat_tot = np.array(cat_tot) / cat_tot[-1]
    nb_cat_subj = np.round(cat_tot * nb_subj).astype(int)
    cat_subj_start = [0, *nb_cat_subj[:-1]]

    # go through each category
    for cat_idx, cat in enumerate(cat_titles):
        if not os.path.isdir(os.path.join(out_path, cat)): 
            os.mkdir(os.path.join(out_path, cat))

        cat_subj_idx = subj_order[cat_subj_start[cat_idx]:nb_cat_subj[cat_idx]]
        for subj_idx in tqdm(cat_subj_idx, desc=cat):
            src_folder = os.path.join(in_path, subjs[subj_idx])
            dst_folder = os.path.join(out_path, cat, subjs[subj_idx])

            if use_symlinks:
                # on windows there are permission problems. 
                # Can try : call(['mklink', 'LINK', 'TARGET'], shell=True)
                # or note https://stackoverflow.com/questions/6260149/os-symlink-support-in-windows
                os.symlink(src_folder, dst_folder)

            else:
                if os.path.isdir(src_folder):
                    shutil.copytree(src_folder, dst_folder)
                else:
                    shutil.copyfile(src_folder, dst_folder)
        









