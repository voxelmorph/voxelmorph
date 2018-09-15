# voxelmorph
Unsupervised Learning for Image Registration  
We incorporate several variants, presented at CVPR2018 and MICCAI2018  

## Notes

- We provide A T1 atlas used in our papers at data/atlas_norm.npz.

- The spatial transform code, found at `neuron.layers.SpatialTransform`, accepts N-dimensional affine and dense transforms, including linear and nearest neighbor interpolation options. Note that original development of VoxelMorph used `xy` indexing, whereas we are not emphasizing `ij` indexing.

- For the MICCAI2018 version, we integrate the velocity field using `neuron.layers.VecInt`. By default we integrate using scaling and squaring, which we found efficient.

- You will likely need to rewrite some of the data loading code in 
'datagenerator.py' for your own datasets and data formats. There are several hard-coded elements related to data preprocessing and format. 


## Instructions

### Training:
These instructions are for the MICCAI2018 paper. If you'd like the CVPR version (no diffeomorphism or uncertainty measures and using CC instead of L2) use train.py

1. Change the top parameters in train_miccai2018.py to the location of your image files.
2. Run train_miccai2018.py with options described in the main function. Example:  
```
train_miccai2018.py --gpu 0 --model_dir /my/path/to/models 
```

### Testing (measuring Dice scores):
1. Put test filenames in data/test_examples.txt, and anatomical labels in data/test_labels.mat.
2. Run test_miccai2018.py [gpu-id] [model_dir] [iter-num]

## Papers
**Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)


**An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)



## Contact:
For and problems or questions please open an issue in github or email us at voxelmorph@mit.edu
