# voxelmorph


## Notes
- Code was initially written in python 2.7, and now transfered to 3.5. 

- We are currently cleaning up our code for general use. There are several hard-coded elements related
to data preprocessing and format. You will likely need to rewrite some of the data loading code in 
'datagenerator.py' for your own datasets.

- We provide the atlas used in our papers at data/atlas_norm.npz.

## Papers
**Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)


**An Unsupervised Learning Model for Deformable Medical Image Registration**  
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca   
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)

## Instructions

### Training:

1. Change base_data_dir in train.py to the location of your image files.
2. Run train.py [model_name] [gpu-id] 

### Testing (Dice scores):
Put test filenames in data/test_examples.txt, and anatomical labels in data/test_labels.mat.
1. Run test.py [model_name] [gpu-id] [iter-num]

### Contact:
voxelmorph@mit.edu
