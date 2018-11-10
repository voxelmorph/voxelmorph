# voxelmorph
Unsupervised Learning with CNNs for Image Registration  
This repository incorporates several variants, first presented at CVPR2018 (initial unsupervised learning) and then MICCAI2018  (probabilistic & diffeomorphic formulation)

# Instructions

## Setup
It might be useful to have each folder inside the `ext` folder on your python path. 
assuming voxelmorph is setup at `/path/to/voxelmorph/`:

```
export PYTHONPATH=$PYTHONPATH:/path/to/voxelmorph/ext/neuron/:/path/to/voxelmorph/ext/pynd-lib/:/path/to/voxelmorph/ext/pytools-lib/
```

If you would like to train/test your own model, you will likely need to write some of the data loading code in 'datagenerator.py' for your own datasets and data formats. There are several hard-coded elements related to data preprocessing and format. 


## Training
These instructions are for the MICCAI2018 variant using `train_miccai2018.py`.  
If you'd like to run the CVPR version (no diffeomorphism or uncertainty measures, and using CC/MSE as a loss function) use `train.py`

1. Change the top parameters in `train_miccai2018.py` to the location of your image files.
2. Run `train_miccai2018.py` with options described in the main function at the bottom of the file. Example:  
```
train_miccai2018.py /my/path/to/data --gpu 0 --model_dir /my/path/to/save/models 
```

In our experiments, `/my/path/to/data` contains one `npz` file for each subject saved in the variable `vol_data`.

We provide a T1 brain atlas used in our papers at `data/atlas_norm.npz`.

## Testing (measuring Dice scores)
1. Put test filenames in data/test_examples.txt, and anatomical labels in data/test_labels.mat.
2. Run `python test_miccai2018.py [gpu-id] [model_dir] [iter-num]`


## Parameter choices

### CVPR version
For the CC loss function, we found a reg parameter of 1 to work best. For the MSE loss function, we found 0.01 to work best.

### MICCAI version

For our data, we found `image_sigma=0.01` and `prior_lambda=25` to work best.

In the original MICCAI code, the parameters were applied after the scaling of the velocity field. With the newest code, this has been "fixed", with different default parameters reflecting the change. We recommend running the updated code. However, if you'd like to run the very original MICCAI2018 mode, please use `xy` indexing and `use_miccai_int` network option, with MICCAI2018 parameters.


## Spatial Transforms and Integration

- The spatial transform code, found at [`neuron.layers.SpatialTransform`](https://github.com/adalca/neuron/blob/master/neuron/layers.py), accepts N-dimensional affine and dense transforms, including linear and nearest neighbor interpolation options. Note that original development of VoxelMorph used `xy` indexing, whereas we are now emphasizing `ij` indexing.

- For the MICCAI2018 version, we integrate the velocity field using [`neuron.layers.VecInt`]((https://github.com/adalca/neuron/blob/master/neuron/layers.py)). By default we integrate using scaling and squaring, which we found efficient.


# Papers

If you use voxelmorph or some part of the code, please cite:

**Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)


**An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)

# Significant Updates
2018-11-10: Added support for multi-gpu training  
2018-10-12: Significant overhaul of code, especially training scripts and new model files.  
2018-09-15: Added MICCAI2018 support and py3 transition  
2018-05-14: Initial Repository for CVPR version, py2.7


# Contact:
For and problems or questions please open an issue in github or email us at voxelmorph@mit.edu
