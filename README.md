# voxelmorph
Unsupervised Learning with CNNs for Image Registration  
This repository incorporates several variants, first presented at CVPR2018 (initial unsupervised learning) and then MICCAI2018  (probabilistic & diffeomorphic formulation)

keywords: machine learning, convolutional neural networks, alignment, mapping, registration

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


## Registration
If you simply want to register two images:
1. Choose the appropriate model, or train your own.
2. Use `register.py`. For example, Let's say we have a model trained to register subject (moving) to atlas (fixed). One could run:
```
python register.py --gpu 0 /path/to/test_vol.nii.gz /path/to/atlas_norm.nii.gz --out_img /path/to/out.nii.gz --model_file ../models/cvpr2018_vm2_cc.h5 
```
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

If you use voxelmorph or some part of the code, please cite (see [bibtex](citations.bib)):

  * For the diffeomorphic or probabilistic model:

    **Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
Under Review. ArXiv 2019. [eprint arXiv:1903.03545](https://arxiv.org/abs/1903.03545) 

    **Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)



* For the original CNN model, MSE, CC, or segmentation-based losses:

    **VoxelMorph: A Learning Framework for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
IEEE TMI: Transactions on Medical Imaging. 2019. 
[eprint arXiv:1809.05231](https://arxiv.org/abs/1809.05231)

    **An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)

# Significant Updates
2019-01-07: Added example register.py file
2018-11-10: Added support for multi-gpu training  
2018-10-12: Significant overhaul of code, especially training scripts and new model files.  
2018-09-15: Added MICCAI2018 support and py3 transition  
2018-05-14: Initial Repository for CVPR version, py2.7


# Contact:
For and problems or questions please open an issue in github or email us at voxelmorph@mit.edu
