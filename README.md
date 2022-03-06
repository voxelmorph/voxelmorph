# voxelmorph: Learning-Based Image Registration  

**voxelmorph** is a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations.

# Tutorial

Visit the [VoxelMorph tutorial](http://tutorial.voxelmorph.net/) to learn about VoxelMorph and Learning-based Registration. Here's an [additional small tutorial](https://colab.research.google.com/drive/1V0CutSIfmtgDJg1XIkEnGteJuw0u7qT-#scrollTo=h1KXYz-Nauwn) on warping annotations together with images, and another on [template (atlas) construction](https://colab.research.google.com/drive/1SkQbrWTQHpQFrG4J2WoBgGZC9yAzUas2?usp=sharing) with VoxelMorph.


# Instructions

To use the VoxelMorph library, either clone this repository and install the requirements listed in `setup.py` or install directly with pip.

```
pip install voxelmorph
```

## Pre-trained models

See list of pre-trained models available [here](data/readme.md#models).

## Training

If you would like to train your own model, you will likely need to customize some of the data-loading code in `voxelmorph/generators.py` for your own datasets and data formats. However, it is possible to run many of the example scripts out-of-the-box, assuming that you provide a list of filenames in the training dataset. Training data can be in the NIfTI, MGZ, or npz (numpy) format, and it's assumed that each npz file in your data list has a `vol` parameter, which points to the image data to be registered, and an optional `seg` variable, which points to a corresponding discrete segmentation (for semi-supervised learning). It's also assumed that the shape of all training image data is consistent, but this, of course, can be handled in a customized generator if desired.

For a given image list file `/images/list.txt` and output directory `/models/output`, the following script will train an image-to-image registration network (described in MICCAI 2018 by default) with an unsupervised loss. Model weights will be saved to a path specified by the `--model-dir` flag.

```
./scripts/tf/train.py --img-list /images/list.txt --model-dir /models/output --gpu 0
```

The `--img-prefix` and `--img-suffix` flags can be used to provide a consistent prefix or suffix to each path specified in the image list. Image-to-atlas registration can be enabled by providing an atlas file, e.g. `--atlas atlas.npz`. If you'd like to train using the original dense CVPR network (no diffeomorphism), use the `--int-steps 0` flag to specify no flow integration steps. Use the `--help` flag to inspect all of the command line options that can be used to fine-tune network architecture and training.


## Registration

If you simply want to register two images, you can use the `register.py` script with the desired model file. For example, if we have a model `model.h5` trained to register a subject (moving) to an atlas (fixed), we could run:

```
./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0
```

This will save the moved image to `warped.nii.gz`. To also save the predicted deformation field, use the `--save-warp` flag. Both npz or nifty files can be used as input/output in this script.


## Testing (measuring Dice scores)

To test the quality of a model by computing dice overlap between an atlas segmentation and warped test scan segmentations, run:

```
./scripts/tf/test.py --model model.h5 --atlas atlas.npz --scans scan01.npz scan02.npz scan03.npz --labels labels.npz
```

Just like for the training data, the atlas and test npz files include `vol` and `seg` parameters and the `labels.npz` file contains a list of corresponding anatomical labels to include in the computed dice score.


## Parameter choices


### CVPR version

For the CC loss function, we found a reg parameter of 1 to work best. For the MSE loss function, we found 0.01 to work best.


### MICCAI version

For our data, we found `image_sigma=0.01` and `prior_lambda=25` to work best.

In the original MICCAI code, the parameters were applied after the scaling of the velocity field. With the newest code, this has been "fixed", with different default parameters reflecting the change. We recommend running the updated code. However, if you'd like to run the very original MICCAI2018 mode, please use `xy` indexing and `use_miccai_int` network option, with MICCAI2018 parameters.


## Spatial Transforms and Integration

- The spatial transform code, found at `voxelmorph.layers.SpatialTransformer`, accepts N-dimensional affine and dense transforms, including linear and nearest neighbor interpolation options. Note that original development of VoxelMorph used `xy` indexing, whereas we are now emphasizing `ij` indexing.

- For the MICCAI2018 version, we integrate the velocity field using `voxelmorph.layers.VecInt`. By default we integrate using scaling and squaring, which we found efficient.


# VoxelMorph Papers

If you use voxelmorph or some part of the code, please cite (see [bibtex](citations.bib)):

  * HyperMorph, avoiding the need to tune registration hyperparameters:   

    **HyperMorph: Amortized Hyperparameter Learning for Image Registration.**  
    Andrew Hoopes, [Malte Hoffmann](https://nmr.mgh.harvard.edu/malte), Bruce Fischl, [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
    IPMI: Information Processing in Medical Imaging. 2021. [eprint arxiv:2101.01035](https://arxiv.org/abs/2101.01035)


  * [SynthMorph](https://synthmorph.voxelmorph.net), avoiding the need to have data at training (!):  

    **SynthMorph: learning contrast-invariant registration without acquired images.**  
    [Malte Hoffmann](https://nmr.mgh.harvard.edu/malte), Benjamin Billot, [Juan Eugenio Iglesias](https://scholar.harvard.edu/iglesias), Bruce Fischl, [Adrian V. Dalca](http://adalca.mit.edu)  
    IEEE TMI: Transactions on Medical Imaging. 2022. [eprint arXiv:2004.10282](https://arxiv.org/abs/2004.10282)

  * For the atlas formation model:  
  
    **Learning Conditional Deformable Templates with Convolutional Networks**  
  [Adrian V. Dalca](http://adalca.mit.edu), [Marianne Rakic](https://mariannerakic.github.io/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
  NeurIPS 2019. [eprint arXiv:1908.02738](https://arxiv.org/abs/1908.02738)

  * For the diffeomorphic or probabilistic model:

    **Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MedIA: Medial Image Analysis. 2019. [eprint arXiv:1903.03545](https://arxiv.org/abs/1903.03545) 

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


# Notes:
- **keywords**: machine learning, convolutional neural networks, alignment, mapping, registration  
- The `master` branch is still in testing as we roll out a major refactoring of the library.     
- If you'd like to run code from VoxelMorph publications, please use the `legacy` branch.  
- **data in papers**: 
In our initial papers, we used publicly available data, but unfortunately we cannot redistribute it (due to the constraints of those datasets). We do a certain amount of pre-processing for the brain images we work with, to eliminate sources of variation and be able to compare algorithms on a level playing field. In particular, we perform FreeSurfer `recon-all` steps up to skull stripping and affine normalization to Talairach space, and crop the images via `((48, 48), (31, 33), (3, 29))`. 

We encourage users to download and process their own data. See [a list of medical imaging datasets here](https://github.com/adalca/medical-datasets). Note that you likely do not need to perform all of the preprocessing steps, and indeed VoxelMorph has been used in other work with other data.


# Creation of Deformable Templates

To experiment with this method, please use `train_template.py` for unconditional templates and `train_cond_template.py` for conditional templates, which use the same conventions as voxelmorph (please note that these files are less polished than the rest of the voxelmorph library).

We've also provided an unconditional atlas in `data/generated_uncond_atlas.npz.npy`. 

Models in h5 format weights are provided for [unconditional atlas here](http://people.csail.mit.edu/adalca/voxelmorph/atlas_creation_uncond_NCC_1500.h5), and [conditional atlas here](http://people.csail.mit.edu/adalca/voxelmorph/atlas_creation_cond_NCC_1022.h5).

**Explore the atlases [interactively here](http://voxelmorph.mit.edu/atlas_creation/)** with tipiX!


# SynthMorph

SynthMorph is a strategy for learning registration without acquired imaging data, producing powerful networks agnostic to contrast induced by MRI ([eprint arXiv:2004.10282](https://arxiv.org/abs/2004.10282)). For a video and a demo showcasing the steps of generating random label maps from noise distributions and using these to train a network, visit [synthmorph.voxelmorph.net](https://synthmorph.voxelmorph.net).

We provide model files for a ["shapes" variant](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5) of SynthMorph, that we train using images synthesized from random shapes only, and a ["brains" variant](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/brains-dice-vel-0.5-res-16-256f.h5), that we train using images synthesized from brain label maps. We train the brains variant by optimizing a loss term that measures volume overlap of a [selection of brain labels](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/fs-labels.npy). For registration with either model, please use the `register.py` script with the respective model weights.

Accurate registration requires the input images to be min-max normalized, such that voxel intensities range from 0 to 1, and to be resampled in the affine space of a [reference image](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/ref.nii.gz). The affine registration can be performed with a variety of packages, and we choose FreeSurfer. First, we skull-strip the images with [SAMSEG](https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg), keeping brain labels only. Second, we run [mri_robust_register](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_register):

```
mri_robust_register --mov in.nii.gz --dst out.nii.gz --lta transform.lta --satit --iscale
mri_robust_register --mov in.nii.gz --dst out.nii.gz --lta transform.lta --satit --iscale --ixform transform.lta --affine
```

where we replace `--satit --iscale` with `--cost NMI` for registration across MRI contrasts.


# Contact:
For any problems or questions please [open an issue](https://github.com/voxelmorph/voxelmorph/issues/new?labels=voxelmorph) for code problems/questions or [start a discussion](https://github.com/voxelmorph/voxelmorph/discussions) for general registration/voxelmorph question/discussion.  
