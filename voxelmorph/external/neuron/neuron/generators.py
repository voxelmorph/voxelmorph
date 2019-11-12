""" generators for the neuron project """

# general imports
import sys
import os

# third party imports
import numpy as np
import nibabel as nib
import scipy
from keras.utils import np_utils 
import keras
import keras.preprocessing
import keras.preprocessing.image
from keras.models import Model

# local packages
import pynd.ndutils as nd
import pytools.patchlib as pl
import pytools.timer as timer

# reload patchlib (it's often updated right now...)
from imp import reload
reload(pl)

# other neuron (this project) packages
from . import dataproc as nrn_proc
from . import models as nrn_models


class Vol(object):
    
    def __init__(self, 
                 volpath,
                 ext='.npz',
                 nb_restart_cycle=None,     # number of files to restart after
                 name='single_vol',         # name
                 fixed_vol_size=True,       # assumes each volume is fixed size
                 ):

        # get filenames at given paths
        volfiles = _get_file_list(volpath, ext, vol_rand_seed)
        nb_files = len(volfiles)
        assert nb_files > 0, "Could not find any files at %s with extension %s" % (volpath, ext)

        # set up restart cycle for volume files --
        # i.e. after how many volumes do we restart
        if nb_restart_cycle is None:
            nb_restart_cycle = nb_files

        # compute subvolume split
        vol_data = _load_medical_volume(os.path.join(volpath, volfiles[0]), ext)
        # process volume
        if data_proc_fn is not None:
            vol_data = data_proc_fn(vol_data)
            [f for f in _npz_headers(npz, namelist=['vol_data.npy'])][0][1]

        nb_patches_per_vol = 1
        if fixed_vol_size and (patch_size is not None) and all(f is not None for f in patch_size):
            nb_patches_per_vol = np.prod(pl.gridsize(vol_data.shape, patch_size, patch_stride))

        assert nb_restart_cycle <= (nb_files * nb_patches_per_vol), \
            '%s restart cycle (%s) too big (%s) in %s' % \
            (name, nb_restart_cycle, nb_files * nb_patches_per_vol, volpath)


def vol(volpath,
        ext='.npz',
        batch_size=1,
        expected_nb_files=-1,
        expected_files=None,
        data_proc_fn=None,  # processing function that takes in one arg (the volume)
        relabel=None,       # relabeling array
        nb_labels_reshape=0,  # reshape to categorial format for keras, need # labels
        keep_vol_size=False,  # whether to keep the volume size on categorical resizing
        name='single_vol',  # name, optional
        nb_restart_cycle=None,  # number of files to restart after
        patch_size=None,     # split the volume in patches? if so, get patch_size
        patch_stride=1,  # split the volume in patches? if so, get patch_stride
        collapse_2d=None,
        extract_slice=None,
        force_binary=False,
        nb_feats=1,
        patch_rand=False,
        patch_rand_seed=None,
        vol_rand_seed=None,
        binary=False,
        yield_incomplete_final_batch=True,
        verbose=False):
    """
    generator for single volume (or volume patches) from a list of files

    simple volume generator that loads a volume (via npy/mgz/nii/niigz), processes it,
    and prepares it for keras model formats

    if a patch size is passed, breaks the volume into patches and generates those
    """

    # get filenames at given paths
    volfiles = _get_file_list(volpath, ext, vol_rand_seed)
    nb_files = len(volfiles)
    assert nb_files > 0, "Could not find any files at %s with extension %s" % (volpath, ext)

    # compute subvolume split
    vol_data = _load_medical_volume(os.path.join(volpath, volfiles[0]), ext)

    # process volume
    if data_proc_fn is not None:
        vol_data = data_proc_fn(vol_data)

    nb_patches_per_vol = 1
    if patch_size is not None and all(f is not None for f in patch_size):
        if relabel is None and len(patch_size) == (len(vol_data.shape) - 1):
            tmp_patch_size = [f for f in patch_size]
            patch_size = [*patch_size, vol_data.shape[-1]]
            patch_stride = [f for f in patch_stride]
            patch_stride = [*patch_stride, vol_data.shape[-1]]
        assert len(vol_data.shape) == len(patch_size), "Vol dims %d are  not equal to patch dims %d" % (len(vol_data.shape), len(patch_size))
        nb_patches_per_vol = np.prod(pl.gridsize(vol_data.shape, patch_size, patch_stride))
    if nb_restart_cycle is None:
        print("setting restart cycle to", nb_files)
        nb_restart_cycle = nb_files
    
    assert nb_restart_cycle <= (nb_files * nb_patches_per_vol), \
        '%s restart cycle (%s) too big (%s) in %s' % \
        (name, nb_restart_cycle, nb_files * nb_patches_per_vol, volpath)

    # check the number of files matches expected (if passed)
    if expected_nb_files >= 0:
        assert nb_files == expected_nb_files, \
            "number of files do not match: %d, %d" % (nb_files, expected_nb_files)
    if expected_files is not None:
        if not (volfiles == expected_files):
            print('file lists did not match. You should probably stop execution.', file=sys.stderr)
            print(len(volfiles), len(expected_files))

    if verbose:
        print('nb_restart_cycle:', nb_restart_cycle)

    # iterate through files
    fileidx = -1
    batch_idx = -1
    feat_idx = 0
    batch_shape = None
    while 1:
        fileidx = np.mod(fileidx + 1, nb_restart_cycle)
        if verbose and fileidx == 0:
            print('starting %s cycle' % name)

        # read next file (circular)
      
        try:
            if verbose:
                print('opening %s' % os.path.join(volpath, volfiles[fileidx]))
            file_name = os.path.join(volpath, volfiles[fileidx])
            vol_data = _load_medical_volume(file_name, ext, verbose)
            # print(file_name, " was loaded", vol_data.shape)
        except:
            debug_error_msg = "#files: %d, fileidx: %d, nb_restart_cycle: %d. error: %s"
            print(debug_error_msg % (len(volfiles), fileidx, nb_restart_cycle, sys.exc_info()[0]))
            raise

        # process volume
        if data_proc_fn is not None:
            vol_data = data_proc_fn(vol_data)

        # the original segmentation files have non-sequential relabel (i.e. some relabel are
        # missing to avoid exploding our model, we only care about the relabel that exist.
        if relabel is not None:
            vol_data = _relabel(vol_data, relabel)

        # split volume into patches if necessary and yield
        if patch_size is None:
            this_patch_size = vol_data.shape
            patch_stride = [1 for f in this_patch_size]
        
        else:
            this_patch_size = [f for f in patch_size]
            for pi, p in enumerate(this_patch_size):
                if p is None:
                    this_patch_size[pi] = vol_data.shape[pi]
                    patch_stride[pi] = 1

        assert ~np.any(np.isnan(vol_data)), "Found a nan for %s" % volfiles[fileidx]
        assert np.all(np.isfinite(vol_data)), "Found a inf for %s" % volfiles[fileidx]

        patch_gen = patch(vol_data, this_patch_size,
                          patch_stride=patch_stride,
                          nb_labels_reshape=nb_labels_reshape,
                          batch_size=1,
                          infinite=False,
                          collapse_2d=collapse_2d,
                          patch_rand=patch_rand,
                          patch_rand_seed=patch_rand_seed,
                          keep_vol_size=keep_vol_size)

        empty_gen = True
        patch_idx = -1
        for lpatch in patch_gen:
            empty_gen = False
            patch_idx += 1

            # add to feature
            if np.mod(feat_idx, nb_feats) == 0:
                vol_data_feats = lpatch
                
            else:
                vol_data_feats = np.concatenate([vol_data_feats, lpatch], np.ndim(lpatch)-1)
            feat_idx += 1

            if binary:
                vol_data_feats = vol_data_feats.astype(bool)

            if np.mod(feat_idx, nb_feats) == 0:
                feats_shape = vol_data_feats[1:]

                # yield previous batch if the new volume has different patch sizes
                if batch_shape is not None and (feats_shape != batch_shape):
                    batch_idx = -1
                    batch_shape = None
                    print('switching patch sizes')
                    yield np.vstack(vol_data_batch)

                # add to batch of volume data, unless the batch is currently empty
                if batch_idx == -1:
                    vol_data_batch = [vol_data_feats]
                    batch_shape = vol_data_feats[1:]
                else:
                    vol_data_batch = [*vol_data_batch, vol_data_feats]

                # yield patch
                batch_idx += 1
                batch_done = batch_idx == batch_size - 1
                files_done = np.mod(fileidx + 1, nb_restart_cycle) == 0
                final_batch = yield_incomplete_final_batch and files_done and patch_idx == (nb_patches_per_vol-1)
                if final_batch: # verbose and 
                    print('last batch in %s cycle %d. nb_batch:%d' % (name, fileidx, len(vol_data_batch)))

                if batch_done or final_batch:
                    batch_idx = -1
                    q = np.vstack(vol_data_batch)
                    yield q

        if empty_gen:
            raise ValueError('Patch generator was empty for file %s', volfiles[fileidx])


def patch(vol_data,             # the volume
          patch_size,           # patch size
          patch_stride=1,       # patch stride (spacing)
          nb_labels_reshape=1,  # number of labels for categorical resizing. 0 if no resizing
          keep_vol_size=False,  # whether to keep the volume size on categorical resizing
          batch_size=1,         # batch size
          collapse_2d=None,
          patch_rand=False,
          patch_rand_seed=None,
          variable_batch_size=False,
          infinite=False):      # whether the generator should continue (re)-generating patches
    """
    generate patches from volume for keras package

    Yields:
        patch: nd array of shape [batch_size, *patch_size], unless resized via nb_labels_reshape
    """

    # some parameter setup
    assert batch_size >= 1, "batch_size should be at least 1"
    if patch_size is None:
        patch_size = vol_data.shape
    for pi,p in enumerate(patch_size):
        if p is None:
            patch_size[pi] = vol_data.shape[pi]
    batch_idx = -1
    if variable_batch_size:
        batch_size = yield


    # do while. if not infinite, will break at the end
    while True:
        # create patch generator
        gen = pl.patch_gen(vol_data, patch_size,
                           stride=patch_stride,
                           rand=patch_rand,
                           rand_seed=patch_rand_seed)

        # go through the patch generator
        empty_gen = True
        for lpatch in gen:

            empty_gen = False
            # reshape output layer as categorical and prep proper size
            # print(lpatch.shape, nb_labels_reshape, keep_vol_size, patch_size)
            lpatch = _categorical_prep(lpatch, nb_labels_reshape, keep_vol_size, patch_size)

            if collapse_2d is not None:
                lpatch = np.squeeze(lpatch, collapse_2d + 1)  # +1 due to batch in first dim

            # add this patch to the stack
            if batch_idx == -1:
                if batch_size == 1:
                    patch_data_batch = lpatch
                else:
                    patch_data_batch = np.zeros([batch_size, *lpatch.shape[1:]])
                    patch_data_batch[0, :] = lpatch

            else:
                patch_data_batch[batch_idx+1, :] = lpatch

            # yield patch
            batch_idx += 1
            if batch_idx == batch_size - 1:
                batch_idx = -1
                batch_size_y = yield patch_data_batch
                if variable_batch_size:
                    batch_size = batch_size_y

        assert not empty_gen, 'generator was empty. vol size was %s' % ''.join(['%d '%d for d in vol_data.shape])

        # if not infinite generation, yield the last batch and break the while
        if not infinite:
            if batch_idx >= 0:
                patch_data_batch = patch_data_batch[:(batch_idx+1), :]
                yield patch_data_batch
            break


def vol_seg(volpath,
            segpath,
            proc_vol_fn=None,
            proc_seg_fn=None,
            verbose=False,
            name='vol_seg', # name, optional
            ext='.npz',
            nb_restart_cycle=None,  # number of files to restart after
            nb_labels_reshape=-1,
            collapse_2d=None,
            force_binary=False,
            nb_input_feats=1,
            relabel=None,
            vol_rand_seed=None,
            seg_binary=False,
            vol_subname='norm',  # subname of volume
            seg_subname='aseg',  # subname of segmentation
            **kwargs):
    """
    generator with (volume, segmentation)

    verbose is passed down to the base generators.py primitive generator (e.g. vol, here)

    ** kwargs are any named arguments for vol(...),
        except verbose, data_proc_fn, ext, nb_labels_reshape and name
            (which this function will control when calling vol())
    """

    # get vol generator
    vol_gen = vol(volpath, **kwargs, ext=ext,
                  nb_restart_cycle=nb_restart_cycle, collapse_2d=collapse_2d, force_binary=False,
                  relabel=None, data_proc_fn=proc_vol_fn, nb_labels_reshape=1, name=name+' vol',
                  verbose=verbose, nb_feats=nb_input_feats, vol_rand_seed=vol_rand_seed)

    # get seg generator, matching nb_files
    # vol_files = [f.replace('norm', 'aseg') for f in _get_file_list(volpath, ext)]
    # vol_files = [f.replace('orig', 'aseg') for f in vol_files]
    vol_files = [f.replace(vol_subname, seg_subname) for f in _get_file_list(volpath, ext, vol_rand_seed)]
    seg_gen = vol(segpath, **kwargs, ext=ext, nb_restart_cycle=nb_restart_cycle, collapse_2d=collapse_2d,
                  force_binary=force_binary, relabel=relabel, vol_rand_seed=vol_rand_seed,
                  data_proc_fn=proc_seg_fn, nb_labels_reshape=nb_labels_reshape, keep_vol_size=True,
                  expected_files=vol_files, name=name+' seg', binary=seg_binary, verbose=False)

    # on next (while):
    while 1:
        # get input and output (seg) vols
        input_vol = next(vol_gen).astype('float16')
        output_vol = next(seg_gen).astype('float16')  # was int8. Why? need float possibility...

        # output input and output
        yield (input_vol, output_vol)


# def seg_seg(volpath,
#             segpath,
#             crop=None, resize_shape=None, rescale=None, # processing parameters
#             verbose=False,
#             name='seg_seg', # name, optional
#             ext='.npz',
#             nb_restart_cycle=None,  # number of files to restart after
#             nb_labels_reshape=-1,
#             collapse_2d=None,
#             force_binary=False,
#             nb_input_feats=1,
#             **kwargs):
#     """
#     generator with (volume, segmentation)

#     verbose is passed down to the base generators.py primitive generator (e.g. vol, here)

#     ** kwargs are any named arguments for vol(...), 
#         except verbose, data_proc_fn, ext, nb_labels_reshape and name 
#             (which this function will control when calling vol())
#     """

#     # compute processing function
#     proc_seg_fn = lambda x: nrn_proc.vol_proc(x, crop=crop, resize_shape=resize_shape,
#                                               interp_order=0, rescale=rescale)

#     # get seg generator, matching nb_files
#     seg_gen = vol(segpath, **kwargs, ext=ext, nb_restart_cycle=nb_restart_cycle,
#                   force_binary=force_binary, collapse_2d = collapse_2d,
#                   data_proc_fn=proc_seg_fn, nb_labels_reshape=nb_labels_reshape, keep_vol_size=True,
#                   name=name+' seg', verbose=False)

#     # on next (while):
#     while 1:
#         # get input and output (seg) vols
#         input_vol = next(seg_gen).astype('float16')
#         vol_size = np.prod(input_vol.shape[1:-1])  # get the volume size
#         output_vol = np.reshape(input_vol, (input_vol.shape[0], vol_size, input_vol.shape[-1]))

#         # output input and output
#         yield (input_vol, output_vol)



def vol_cat(volpaths, # expect two folders in here
            crop=None, resize_shape=None, rescale=None, # processing parameters
            verbose=False,
            name='vol_cat', # name, optional
            ext='.npz',
            nb_labels_reshape=-1,
            vol_rand_seed=None,
            **kwargs): # named arguments for vol(...), except verbose, data_proc_fn, ext, nb_labels_reshape and name (which this function will control when calling vol()) 
    """
    generator with (volume, binary_bit) (random order)
    ONLY works with abtch size of 1 for now

    verbose is passed down to the base generators.py primitive generator (e.g. vol, here)
    """

    folders = [f for f in sorted(os.listdir(volpaths))]

    # compute processing function
    proc_vol_fn = lambda x: nrn_proc.vol_proc(x, crop=crop, resize_shape=resize_shape,
                                              interp_order=2, rescale=rescale)

    # get vol generators
    generators = ()
    generators_len = ()
    for folder in folders:
        vol_gen = vol(os.path.join(volpaths, folder), **kwargs, ext=ext, vol_rand_seed=vol_rand_seed,
                      data_proc_fn=proc_vol_fn, nb_labels_reshape=1, name=folder, verbose=False)
        generators_len += (len(_get_file_list(os.path.join(volpaths, folder), '.npz')), )
        generators += (vol_gen, )

    bake_data_test = False
    if bake_data_test:
        print('fake_data_test', file=sys.stderr)

    # on next (while):
    while 1:
        # build the random order stack
        order = np.hstack((np.zeros(generators_len[0]), np.ones(generators_len[1]))).astype('int')
        np.random.shuffle(order) # shuffle
        for idx in order:
            gen = generators[idx]

        # for idx, gen in enumerate(generators):
            z = np.zeros([1, 2]) #1,1,2 for categorical binary style
            z[0,idx] = 1 #
            # z[0,0,0] = idx

            data = next(gen).astype('float32')
            if bake_data_test and idx == 0:
                # data = data*idx
                data = -data

            yield (data, z)



def add_prior(gen,
              proc_vol_fn=None,
              proc_seg_fn=None,
              prior_type='location',  # file-static, file-gen, location
              prior_file=None,  # prior filename
              prior_feed='input',  # input or output
              patch_stride=1,
              patch_size=None,
              batch_size=1,
              collapse_2d=None,
              extract_slice=None,
              force_binary=False,
              verbose=False,
              patch_rand=False,
              patch_rand_seed=None):
    """
    #
    # add a prior generator to a given generator
    # with the number of patches in batch matching output of gen
    """

    # get prior
    if prior_type == 'location':
        prior_vol = nd.volsize2ndgrid(vol_size)
        prior_vol = np.transpose(prior_vol, [1, 2, 3, 0])
        prior_vol = np.expand_dims(prior_vol, axis=0) # reshape for model

    elif prior_type == 'file': # assumes a npz filename passed in prior_file
        with timer.Timer('loading prior', True):
            data = np.load(prior_file)
            prior_vol = data['prior'].astype('float16')

    else: # assumes a volume
        with timer.Timer('loading prior', True):
            prior_vol = prior_file.astype('float16')

    if force_binary:
        nb_labels = prior_vol.shape[-1]
        prior_vol[:, :, :, 1] = np.sum(prior_vol[:, :, :, 1:nb_labels], 3)
        prior_vol = np.delete(prior_vol, range(2, nb_labels), 3)

    nb_channels = prior_vol.shape[-1]

    if extract_slice is not None:
        if isinstance(extract_slice, int):
            prior_vol = prior_vol[:, :, extract_slice, np.newaxis, :]
        else:  # assume slices
            prior_vol = prior_vol[:, :, extract_slice, :]

    # get the prior to have the right volume [x, y, z, nb_channels]
    assert np.ndim(prior_vol) == 4 or np.ndim(prior_vol) == 3, "prior is the wrong size"

    # prior generator
    if patch_size is None:
        patch_size = prior_vol.shape[0:3]
    assert len(patch_size) == len(patch_stride)
    prior_gen = patch(prior_vol, [*patch_size, nb_channels],
                      patch_stride=[*patch_stride, nb_channels],
                      batch_size=batch_size,
                      collapse_2d=collapse_2d,
                      keep_vol_size=True,
                      infinite=True,
                      patch_rand=patch_rand,
                      patch_rand_seed=patch_rand_seed,
                      variable_batch_size=True,
                      nb_labels_reshape=0)
    assert next(prior_gen) is None, "bad prior gen setup"

    # generator loop
    while 1:

        # generate input and output volumes
        gen_sample = next(gen)

        # generate prior batch
        gs_sample = _get_shape(gen_sample)
        prior_batch = prior_gen.send(gs_sample)

        yield (gen_sample, prior_batch)


def vol_prior(*args,
              proc_vol_fn=None,
              proc_seg_fn=None,
              prior_type='location',  # file-static, file-gen, location
              prior_file=None,  # prior filename
              prior_feed='input',  # input or output
              patch_stride=1,
              patch_size=None,
              batch_size=1,
              collapse_2d=None,
              extract_slice=None,
              force_binary=False,
              nb_input_feats=1,
              verbose=False,
              vol_rand_seed=None,
              patch_rand=False,
              **kwargs):  # anything else you'd like to pass to vol()
    """
    generator that appends prior to (volume, segmentation) depending on input
    e.g. could be ((volume, prior), segmentation)
    """

    patch_rand_seed = None
    if patch_rand:
        patch_rand_seed = np.random.random()


    # prepare the vol_seg
    vol_gen = vol(*args,
                  **kwargs,
                  collapse_2d=collapse_2d,
                  force_binary=False,
                  verbose=verbose,
                  vol_rand_seed=vol_rand_seed)
    gen = vol(*args, **kwargs,
              proc_vol_fn=None,
              proc_seg_fn=None,
              collapse_2d=collapse_2d,
              extract_slice=extract_slice,
              force_binary=force_binary,
              verbose=verbose,
              patch_size=patch_size,
              patch_stride=patch_stride,
              batch_size=batch_size,
              vol_rand_seed=vol_rand_seed,
              patch_rand=patch_rand,
              patch_rand_seed=patch_rand_seed,
              nb_input_feats=nb_input_feats)

    # add prior to output
    pgen = add_prior(gen,
                     proc_vol_fn=proc_vol_fn,
                     proc_seg_fn=proc_seg_fn,
                     prior_type=prior_type,
                     prior_file=prior_file,
                     prior_feed=prior_feed,
                     patch_stride=patch_stride,
                     patch_size=patch_size,
                     batch_size=batch_size,
                     collapse_2d=collapse_2d,
                     extract_slice=extract_slice,
                     force_binary=force_binary,
                     verbose=verbose,
                     patch_rand=patch_rand,
                     patch_rand_seed=patch_rand_seed,
                     vol_rand_seed=vol_rand_seed)

    # generator loop
    while 1:

        gen_sample, prior_batch = next(pgen)
        input_vol, output_vol = gen_sample

        if prior_feed == 'input':
            yield ([input_vol, prior_batch], output_vol)
        else:
            assert prior_feed == 'output'
            yield (input_vol, [output_vol, prior_batch])


def vol_seg_prior(*args,
                  proc_vol_fn=None,
                  proc_seg_fn=None,
                  prior_type='location',  # file-static, file-gen, location
                  prior_file=None,  # prior filename
                  prior_feed='input',  # input or output
                  patch_stride=1,
                  patch_size=None,
                  batch_size=1,
                  collapse_2d=None,
                  extract_slice=None,
                  force_binary=False,
                  nb_input_feats=1,
                  verbose=False,
                  vol_rand_seed=None,
                  patch_rand=None,
                  **kwargs):
    """
    generator that appends prior to (volume, segmentation) depending on input
    e.g. could be ((volume, prior), segmentation)
    """


    patch_rand_seed = None
    if patch_rand:
        patch_rand_seed = np.random.random()

    # prepare the vol_seg
    gen = vol_seg(*args, **kwargs,
                  proc_vol_fn=None,
                  proc_seg_fn=None,
                  collapse_2d=collapse_2d,
                  extract_slice=extract_slice,
                  force_binary=force_binary,
                  verbose=verbose,
                  patch_size=patch_size,
                  patch_stride=patch_stride,
                  batch_size=batch_size,
                  vol_rand_seed=vol_rand_seed,
                  patch_rand=patch_rand,
                  patch_rand_seed=patch_rand_seed,
                  nb_input_feats=nb_input_feats)

    # add prior to output
    pgen = add_prior(gen,
                     proc_vol_fn=proc_vol_fn,
                     proc_seg_fn=proc_seg_fn,
                     prior_type=prior_type,
                     prior_file=prior_file,
                     prior_feed=prior_feed,
                     patch_stride=patch_stride,
                     patch_size=patch_size,
                     batch_size=batch_size,
                     collapse_2d=collapse_2d,
                     extract_slice=extract_slice,
                     force_binary=force_binary,
                     verbose=verbose,
                     patch_rand=patch_rand,
                     patch_rand_seed=patch_rand_seed)

    # generator loop
    while 1:

        gen_sample, prior_batch = next(pgen)
        input_vol, output_vol = gen_sample

        if prior_feed == 'input':
            yield ([input_vol, prior_batch], output_vol)
        else:
            assert prior_feed == 'output'
            yield (input_vol, [output_vol, prior_batch])



def vol_prior_hack(*args,
                   proc_vol_fn=None,
                   proc_seg_fn=None,
                   prior_type='location',  # file-static, file-gen, location
                   prior_file=None,  # prior filename
                   prior_feed='input',  # input or output
                   patch_stride=1,
                   patch_size=None,
                   batch_size=1,
                   collapse_2d=None,
                   extract_slice=None,
                   force_binary=False,
                   nb_input_feats=1,
                   verbose=False,
                   vol_rand_seed=None,
                   **kwargs):
    """
    
    """
    # prepare the vol_seg
    gen = vol_seg_hack(*args, **kwargs,
                        proc_vol_fn=None,
                        proc_seg_fn=None,
                        collapse_2d=collapse_2d,
                        extract_slice=extract_slice,
                        force_binary=force_binary,
                        verbose=verbose,
                        patch_size=patch_size,
                        patch_stride=patch_stride,
                        batch_size=batch_size,
                        vol_rand_seed=vol_rand_seed,
                        nb_input_feats=nb_input_feats)

    # get prior
    if prior_type == 'location':
        prior_vol = nd.volsize2ndgrid(vol_size)
        prior_vol = np.transpose(prior_vol, [1, 2, 3, 0])
        prior_vol = np.expand_dims(prior_vol, axis=0) # reshape for model

    elif prior_type == 'file': # assumes a npz filename passed in prior_file
        with timer.Timer('loading prior', True):
            data = np.load(prior_file)
            prior_vol = data['prior'].astype('float16')
    else : # assumes a volume
        with timer.Timer('astyping prior', verbose):
            prior_vol = prior_file
            if not (prior_vol.dtype == 'float16'):
                prior_vol = prior_vol.astype('float16')

    if force_binary:
        nb_labels = prior_vol.shape[-1]
        prior_vol[:, :, :, 1] = np.sum(prior_vol[:, :, :, 1:nb_labels], 3)
        prior_vol = np.delete(prior_vol, range(2, nb_labels), 3)

    nb_channels = prior_vol.shape[-1]

    if extract_slice is not None:
        if isinstance(extract_slice, int):
            prior_vol = prior_vol[:, :, extract_slice, np.newaxis, :]
        else:  # assume slices
            prior_vol = prior_vol[:, :, extract_slice, :]

    # get the prior to have the right volume [x, y, z, nb_channels]
    assert np.ndim(prior_vol) == 4 or np.ndim(prior_vol) == 3, "prior is the wrong size"

    # prior generator
    if patch_size is None:
        patch_size = prior_vol.shape[0:3]
    assert len(patch_size) == len(patch_stride)
    prior_gen = patch(prior_vol, [*patch_size, nb_channels],
                      patch_stride=[*patch_stride, nb_channels],
                      batch_size=batch_size,
                      collapse_2d=collapse_2d,
                      keep_vol_size=True,
                      infinite=True,
                      #variable_batch_size=True,  # this
                      nb_labels_reshape=0)
    # assert next(prior_gen) is None, "bad prior gen setup"

    # generator loop
    while 1:

        # generate input and output volumes
        input_vol = next(gen)

        if verbose and np.all(input_vol.flat == 0):
            print("all entries are 0")

        # generate prior batch
        # with timer.Timer("with send?"):
            # prior_batch = prior_gen.send(input_vol.shape[0])
        prior_batch = next(prior_gen)

        if prior_feed == 'input':
            yield ([input_vol, prior_batch], input_vol)
        else:
            assert prior_feed == 'output'
            yield (input_vol, [input_vol, prior_batch])


def vol_seg_hack(volpath,
            segpath,
            proc_vol_fn=None,
            proc_seg_fn=None,
            verbose=False,
            name='vol_seg', # name, optional
            ext='.npz',
            nb_restart_cycle=None,  # number of files to restart after
            nb_labels_reshape=-1,
            collapse_2d=None,
            force_binary=False,
            nb_input_feats=1,
            relabel=None,
            vol_rand_seed=None,
            seg_binary=False,
            vol_subname='norm',  # subname of volume
            seg_subname='aseg',  # subname of segmentation
            **kwargs):
    """
    generator with (volume, segmentation)

    verbose is passed down to the base generators.py primitive generator (e.g. vol, here)

    ** kwargs are any named arguments for vol(...),
        except verbose, data_proc_fn, ext, nb_labels_reshape and name
            (which this function will control when calling vol())
    """

    # get vol generator
    vol_gen = vol(volpath, **kwargs, ext=ext,
                  nb_restart_cycle=nb_restart_cycle, collapse_2d=collapse_2d, force_binary=False,
                  relabel=None, data_proc_fn=proc_vol_fn, nb_labels_reshape=1, name=name+' vol',
                  verbose=verbose, nb_feats=nb_input_feats, vol_rand_seed=vol_rand_seed)
  

    # on next (while):
    while 1:
        # get input and output (seg) vols
        input_vol = next(vol_gen).astype('float16')

        # output input and output
        yield input_vol

def vol_sr_slices(volpath,
                  nb_input_slices,
                  nb_slice_spacing,
                  batch_size=1,
                  ext='.npz',
                  vol_rand_seed=None,
                  nb_restart_cycle=None,
                  name='vol_sr_slices',
                  rand_slices=True,  # randomize init slice order (i.e. across entries per batch) given a volume
                  simulate_whole_sparse_vol=False,
                  verbose=False
                  ):
    """
    default generator for slice-wise super resolution
    """

    def indices_to_batch(vol_data, start_indices, nb_slices_in_subvol, nb_slice_spacing):
        idx = start_indices[0]
        output_batch = np.expand_dims(vol_data[:,:,idx:idx+nb_slices_in_subvol], 0)
        input_batch = np.expand_dims(vol_data[:,:,idx:(idx+nb_slices_in_subvol):(nb_slice_spacing+1)], 0)
        
        for idx in start_indices[1:]:
            out_sel = np.expand_dims(vol_data[:,:,idx:idx+nb_slices_in_subvol], 0)
            output_batch = np.vstack([output_batch, out_sel])
            input_batch = np.vstack([input_batch, np.expand_dims(vol_data[:,:,idx:(idx+nb_slices_in_subvol):(nb_slice_spacing+1)], 0)])
        output_batch = np.reshape(output_batch, [batch_size, -1, output_batch.shape[-1]])
        
        return (input_batch, output_batch)


    print('vol_sr_slices: SHOULD PROPERLY RANDOMIZE accross different subjects', file=sys.stderr)
    
    volfiles = _get_file_list(volpath, ext, vol_rand_seed)
    nb_files = len(volfiles)

    if nb_restart_cycle is None:
        nb_restart_cycle = nb_files

    # compute the number of slices we'll need in a subvolume
    nb_slices_in_subvol = (nb_input_slices - 1) * (nb_slice_spacing + 1) + 1

    # iterate through files
    fileidx = -1
    while 1:
        fileidx = np.mod(fileidx + 1, nb_restart_cycle)
        if verbose and fileidx == 0:
            print('starting %s cycle' % name)


        try:
            vol_data = _load_medical_volume(os.path.join(volpath, volfiles[fileidx]), ext, verbose)
        except:
            debug_error_msg = "#files: %d, fileidx: %d, nb_restart_cycle: %d. error: %s"
            print(debug_error_msg % (len(volfiles), fileidx, nb_restart_cycle, sys.exc_info()[0]))
            raise

        # compute some random slice
        nb_slices = vol_data.shape[2]
        nb_start_slices = nb_slices - nb_slices_in_subvol + 1

        # prepare batches
        if simulate_whole_sparse_vol:  # if essentially simulate a whole sparse volume for consistent inputs, and yield slices like that:
            init_slice = 0
            if rand_slices:
                init_slice = np.random.randint(0, high=nb_start_slices-1)

            all_start_indices = list(range(init_slice, nb_start_slices, nb_slice_spacing+1))

            for batch_start in range(0, len(all_start_indices), batch_size*(nb_input_slices-1)):
                start_indices = [all_start_indices[s] for s in range(batch_start, batch_start + batch_size)]
                input_batch, output_batch = indices_to_batch(vol_data, start_indices, nb_slices_in_subvol, nb_slice_spacing)
                yield (input_batch, output_batch)
        
        # if just random slices, get a batch of random starts from this volume and that's it.
        elif rand_slices:
            assert not simulate_whole_sparse_vol
            start_indices = np.random.choice(range(nb_start_slices), size=batch_size, replace=False)
            input_batch, output_batch = indices_to_batch(vol_data, start_indices, nb_slices_in_subvol, nb_slice_spacing)
            yield (input_batch, output_batch)

        # go slice by slice (overlapping regions)
        else:
            for batch_start in range(0, nb_start_slices, batch_size):
                start_indices = list(range(batch_start, batch_start + batch_size))
                input_batch, output_batch = indices_to_batch(vol_data, start_indices, nb_slices_in_subvol, nb_slice_spacing)
                yield (input_batch, output_batch)



        

    







def ext_data(segpath,
             volpath,
             batch_size,
             expected_files=None,
             ext='.npy',
             nb_restart_cycle=None,
             data_proc_fn=None,  # processing function that takes in one arg (the volume)
             vol_rand_seed=None,
             name='ext_data',
             verbose=False,
             yield_incomplete_final_batch=True,
             patch_stride=1,
             extract_slice=None,
             patch_size=None,
             ext_data_fields=None,
             expected_nb_files = -1):
    assert ext_data_fields is not None, "Need some external data fields"

   # get filenames at given paths
    volfiles = _get_file_list(segpath, ext, vol_rand_seed)
    nb_files = len(volfiles)
    assert nb_files > 0, "Could not find any files at %s with extension %s" % (segpath, ext)

    # compute subvolume split
    vol_data = _load_medical_volume(os.path.join(volpath, expected_files[0]), '.npz')
    # process volume
    if data_proc_fn is not None:
        vol_data = data_proc_fn(vol_data)

    # nb_patches_per_vol = 1
    # if patch_size is not None and all(f is not None for f in patch_size):
        # nb_patches_per_vol = np.prod(pl.gridsize(vol_data.shape, patch_size, patch_stride))
    if nb_restart_cycle is None:
        nb_restart_cycle = nb_files

    # assert nb_restart_cycle <= (nb_files * nb_patches_per_vol), \
        # '%s restart cycle (%s) too big (%s) in %s' % \
        # (name, nb_restart_cycle, nb_files * nb_patches_per_vol, volpath)

    # check the number of files matches expected (if passed)
    if expected_nb_files >= 0:
        assert nb_files == expected_nb_files, \
            "number of files do not match: %d, %d" % (nb_files, expected_nb_files)
    if expected_files is not None:
        if not (volfiles == expected_files):
            print('file lists did not match !!!', file=sys.stderr)

    # iterate through files
    fileidx = -1
    batch_idx = -1
    feat_idx = 0
    while 1:
        fileidx = np.mod(fileidx + 1, nb_restart_cycle)
        if verbose and fileidx == 0:
            print('starting %s cycle' % name)

        this_ext_data = np.load(os.path.join(segpath, volfiles[fileidx]))
        # print(os.path.join(segpath, volfiles[fileidx]), " was loaded")

        for _ in range(nb_patches_per_vol):
            if batch_idx == -1:
                ext_data_batch = [this_ext_data[f] for f in ext_data_fields]
            else:
                tmp_data = [this_ext_data[f] for f in ext_data_fields]
                ext_data_batch = [[*ext_data_batch[f], this_ext_data[f]] for f in range(len(tmp_data))]
            
            # yield patch
            batch_idx += 1
            batch_done = batch_idx == batch_size - 1
            files_done = np.mod(fileidx + 1, nb_restart_cycle) == 0
            final_batch = (yield_incomplete_final_batch and files_done)
            if verbose and final_batch:
                print('last batch in %s cycle %d' % (name, fileidx))

        if batch_done or final_batch:
            for fi,f in enumerate(ext_data_batch):
                ext_data_batch[fi] = np.array(f)

            batch_idx = -1
            yield ext_data_batch



def vol_ext_data(volpath,
                 segpath,
                 proc_vol_fn=None,
                 proc_seg_fn=None,
                 verbose=False,
                 name='vol_seg', # name, optional
                 ext='.npz',
                 nb_restart_cycle=None,  # number of files to restart after
                 nb_labels_reshape=-1,
                 collapse_2d=None,
                 force_binary=False,
                 nb_input_feats=1,
                 relabel=None,
                 vol_rand_seed=None,
                 vol_subname='norm',  # subname of volume
                 seg_subname='norm',  # subname of segmentation
                 **kwargs):
    """
    generator with (volume, segmentation)

    verbose is passed down to the base generators.py primitive generator (e.g. vol, here)

    ** kwargs are any named arguments for vol(...),
        except verbose, data_proc_fn, ext, nb_labels_reshape and name
            (which this function will control when calling vol())
    """

    # get vol generator
    vol_gen = vol(volpath, **kwargs, ext=ext,
                  nb_restart_cycle=nb_restart_cycle, collapse_2d=collapse_2d, force_binary=False,
                  relabel=None, data_proc_fn=proc_vol_fn, nb_labels_reshape=1, name=name+' vol',
                  verbose=verbose, nb_feats=nb_input_feats, vol_rand_seed=vol_rand_seed)

    # get seg generator, matching nb_files
    # vol_files = [f.replace('norm', 'aseg') for f in _get_file_list(volpath, ext)]
    # vol_files = [f.replace('orig', 'aseg') for f in vol_files]
    vol_files = [f.replace(vol_subname, seg_subname) for f in _get_file_list(volpath, ext, vol_rand_seed)]
    seg_gen = ext_data(segpath, 
                       volpath,
                       **kwargs,
                       data_proc_fn=proc_seg_fn,
                       ext='.npy',
                       nb_restart_cycle=nb_restart_cycle,
                       vol_rand_seed=vol_rand_seed,
                       expected_files=vol_files,
                       name=name+' ext_data',
                       verbose=False)

    # on next (while):
    while 1:
        # get input and output (seg) vols
        input_vol = next(vol_gen).astype('float16')
        output_vol = next(seg_gen) #.astype('float16')

        # output input and output
        yield (input_vol, output_vol)




def vol_ext_data_prior(*args,
                  proc_vol_fn=None,
                  proc_seg_fn=None,
                  prior_type='location',  # file-static, file-gen, location
                  prior_file=None,  # prior filename
                  prior_feed='input',  # input or output
                  patch_stride=1,
                  patch_size=None,
                  batch_size=1,
                  collapse_2d=None,
                  extract_slice=None,
                  force_binary=False,
                  nb_input_feats=1,
                  verbose=False,
                  vol_rand_seed=None,
                  **kwargs):
    """
    generator that appends prior to (volume, segmentation) depending on input
    e.g. could be ((volume, prior), segmentation)
    """

    if verbose:
        print('starting vol_seg_prior')

    # prepare the vol_seg
    gen = vol_ext_data(*args, **kwargs,
                  proc_vol_fn=None,
                  proc_seg_fn=None,
                  collapse_2d=collapse_2d,
                  extract_slice=extract_slice,
                  force_binary=force_binary,
                  verbose=verbose,
                  patch_size=patch_size,
                  patch_stride=patch_stride,
                  batch_size=batch_size,
                  vol_rand_seed=vol_rand_seed,
                  nb_input_feats=nb_input_feats)

    # get prior
    if prior_type == 'location':
        prior_vol = nd.volsize2ndgrid(vol_size)
        prior_vol = np.transpose(prior_vol, [1, 2, 3, 0])
        prior_vol = np.expand_dims(prior_vol, axis=0) # reshape for model

    else: # assumes a npz filename passed in prior_file
        with timer.Timer('loading prior', verbose):
            data = np.load(prior_file)
            prior_vol = data['prior'].astype('float16')

    if force_binary:
        nb_labels = prior_vol.shape[-1]
        prior_vol[:, :, :, 1] = np.sum(prior_vol[:, :, :, 1:nb_labels], 3)
        prior_vol = np.delete(prior_vol, range(2, nb_labels), 3)

    nb_channels = prior_vol.shape[-1]

    if extract_slice is not None:
        if isinstance(extract_slice, int):
            prior_vol = prior_vol[:, :, extract_slice, np.newaxis, :]
        else:  # assume slices
            prior_vol = prior_vol[:, :, extract_slice, :]

    # get the prior to have the right volume [x, y, z, nb_channels]
    assert np.ndim(prior_vol) == 4, "prior is the wrong size"

    # prior generator
    if patch_size is None:
        patch_size = prior_vol.shape[0:3]
    assert len(patch_size) == len(patch_stride)
    prior_gen = patch(prior_vol, [*patch_size, nb_channels],
                      patch_stride=[*patch_stride, nb_channels],
                      batch_size=batch_size,
                      collapse_2d=collapse_2d,
                      infinite=True,
                      variable_batch_size=True,
                      nb_labels_reshape=0)
    assert next(prior_gen) is None, "bad prior gen setup"

    # generator loop
    while 1:

        # generate input and output volumes
        input_vol, output_vol = next(gen)
        if verbose and np.all(input_vol.flat == 0):
            print("all entries are 0")

        # generate prior batch
        prior_batch = prior_gen.send(input_vol.shape[0])

        if prior_feed == 'input':
            yield ([input_vol, prior_batch], output_vol)
        else:
            assert prior_feed == 'output'
            yield (input_vol, [output_vol, prior_batch])


def max_patch_in_vol_cat(volpaths,
                         patch_size,
                         patch_stride,
                         model,
                         tst_model,
                         out_layer,
                         name='max_patch_in_vol_cat', # name, optional
                         **kwargs): # named arguments for vol_cat(...)
    """
    given a model by reference
    goes to the next volume and, given a patch_size, finds
    the highest prediction of out_layer_name, and yields that patch index to the model

    TODO: this doesn't work if called while training, something about running through the graph
    perhaps need to do a different tf session?
    """

    # strategy: call vol_cat on full volume
    vol_cat_gen = vol_cat(volpaths, **kwargs)

    # need to make a deep copy:
    #model_tmp = Model(inputs=model.inputs, outputs=model.get_layer(out_layer).output)
    # nrn_models.copy_weights(model_tmp, tst_model)
    #nrn_models.copy_weights(tst_model, tst_model)
    #asdasd
    tst_model = model

    while 1:
        # get next volume
        try:
            sample = next(vol_cat_gen)
        except:
            print("Failed loading file. Skipping", file=sys.stderr)
            continue
        sample_vol = np.squeeze(sample[0])
        patch_gen = patch(sample_vol, patch_size, patch_stride=patch_stride)

        # go through its patches
        max_resp = -np.infty
        max_idx = np.nan
        max_out = None
        for idx, ptc in enumerate(patch_gen):
            # predict
            res = np.squeeze(tst_model.predict(ptc))[1]
            if res > max_resp:
                max_ptc = ptc
                max_out = sample[1]
                max_idx = idx

        # yield the right patch
        yield (max_ptc, max_out)


def img_seg(volpath,
            segpath,
            batch_size=1,
            verbose=False,
            nb_restart_cycle=None,
            name='img_seg', # name, optional
            ext='.png',
            vol_rand_seed=None,
            **kwargs):
    """
    generator for (image, segmentation)
    """

    def imggen(path, ext, nb_restart_cycle=None):
        """
        TODO: should really use the volume generators for this
        """
        files = _get_file_list(path, ext, vol_rand_seed)
        if nb_restart_cycle is None:
            nb_restart_cycle = len(files)

        idx = -1
        while 1:
            idx = np.mod(idx+1, nb_restart_cycle)
            im = scipy.misc.imread(os.path.join(path, files[idx]))[:, :, 0]
            yield im.reshape((1,) + im.shape)

    img_gen = imggen(volpath, ext, nb_restart_cycle)
    seg_gen = imggen(segpath, ext)

    # on next (while):
    while 1:
        input_vol = np.vstack([next(img_gen).astype('float16')/255 for i in range(batch_size)])
        input_vol = np.expand_dims(input_vol, axis=-1)

        output_vols = [np_utils.to_categorical(next(seg_gen).astype('int8'), num_classes=2) for i in range(batch_size)]
        output_vol = np.vstack([np.expand_dims(f, axis=0) for f in output_vols])

        # output input and output
        yield (input_vol, output_vol)


# Some internal use functions

def _get_file_list(volpath, ext=None, vol_rand_seed=None):
    """
    get a list of files at the given path with the given extension
    """
    files = [f for f in sorted(os.listdir(volpath)) if ext is None or f.endswith(ext)]
    if vol_rand_seed is not None:
        np.random.seed(vol_rand_seed)
        files = np.random.permutation(files).tolist()
    return files


def _load_medical_volume(filename, ext, verbose=False):
    """
    load a medical volume from one of a number of file types
    """
    with timer.Timer('load_vol', verbose >= 2):
        if ext == '.npz':
            vol_file = np.load(filename)
            vol_data = vol_file['vol_data']
        elif ext == 'npy':
            vol_data = np.load(filename)
        elif ext == '.mgz' or ext == '.nii' or ext == '.nii.gz':
            vol_med = nib.load(filename)
            vol_data = vol_med.get_data()
        else:
            raise ValueError("Unexpected extension %s" % ext)

    return vol_data


def _categorical_prep(vol_data, nb_labels_reshape, keep_vol_size, patch_size):

    if nb_labels_reshape > 1:
        
        lpatch = _to_categorical(vol_data, nb_labels_reshape, keep_vol_size)
        # if keep_vol_size:
            # lpatch = np.reshape(lpatch, [*patch_size, nb_labels_reshape])
    elif nb_labels_reshape == 1:
        lpatch = np.expand_dims(vol_data, axis=-1)
    else:
        assert nb_labels_reshape == 0
        lpatch = vol_data
    lpatch = np.expand_dims(lpatch, axis=0)

    return lpatch



def _to_categorical(y, num_classes=None, reshape=True):
    """
    # copy of keras.utils.np_utils.to_categorical, but with a boolean matrix instead of float

    Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    oshape = y.shape
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), bool)
    categorical[np.arange(n), y] = 1
    
    if reshape:
        categorical = np.reshape(categorical, [*oshape, num_classes])
    
    return categorical

def _relabel(vol_data, labels, forcecheck=False):
    
    if forcecheck:
        vd = np.unique(vol_data.flat)
        assert len(vd) == len(labels), "number of given labels does not match number of actual labels"
    
    # by doing zeros, any label not in labels gets left to 0
    new_vol_data = np.zeros(vol_data.shape, vol_data.dtype)
    for idx, val in np.ndenumerate(labels):
        new_vol_data[vol_data == val] = idx
    
    return new_vol_data




import zipfile

def _npz_headers(npz, namelist=None):
    """
    taken from https://stackoverflow.com/a/43223420

    Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).

    namelist is a list with variable names, ending in '.npy'. 
    e.g. if variable 'var' is in the file, namelist could be ['var.npy']
    """
    with zipfile.ZipFile(npz) as archive:
        if namelist is None:
            namelist = archive.namelist()

        for name in namelist:
            if not name.endswith('.npy'):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            yield name[:-4], shape, dtype

def _get_shape(x):
    if isinstance(x, (list, tuple)):
        return _get_shape(x[0])
    else:
        return x.shape[0]
