# voxelmorph

Unsupervised Learning for Image Registration

## Notes
- Code was initially written in python 2.7, and now transfered to 3.5. 

- We are currently cleaning up our code for general use. There are several hard-coded elements related
to data preprocessing and format. You will likely need to rewrite some of the data loading code in 
'datagenerator.py' for your own datasets.

- We provide the atlas used in our papers at data/atlas_norm.npz.

## Instructions

### Training:

1. Change base_data_dir in train.py to the location of your image files.
2. Run train.py [model_name] [gpu-id] 

### Testing (Dice scores):
Put test filenames in data/test_examples.txt, and anatomical labels in data/test_labels.mat.
1. Run test.py [model_name] [gpu-id] [iter-num]

## Papers
**Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)


**An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)



## Contact:
voxelmorph@mit.edu
