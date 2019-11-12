# neuron
A Neural networks toolbox with a focus on medical image analysis in tensorflow/keras for now.  
Note: this is **under ongoing development**

### Main tools
- [layers](neuron/layers.py): various network layers, including a rich `SpatialTransformer` layer for N-D (dense and affine) spatial transforms, a vector integration layer `VecInt`, sparse operations (e.g. `SpatiallySparse_Dense`), and `LocallyConnected3D` currently not included in `keras`  
- [utils](neuron/utils.py): various utilities, including `interpn`: N-D gridded interpolation, `transform`: warp images, `integrate_vec`: vector field integration, `stack_models`: keras model stacking  
- [models](neuron/models.py): flexible models (many parameters to play with) particularly useful in medical image analysis, such as UNet/hourglass model, convolutional encoders and decoders   
- [generators](neuron/generators.py): generators for medical image volumes and various combinations of volumes, segmentation, categorical and other output  
- [callbacks](neuron/callbacks.py): a set of callbacks for `keras` training to help with understanding your fit, such as Dice measurements and volume-segmentation overlaps  
- [dataproc](neuron/dataproc.py): a set of tools for processing medical imaging data for preparation for training/testing  
- [metrics](neuron/metrics.py): metrics (most of which can be used as loss functions), such as Dice or weighted categorical crossentropy  
- [vae_tools](neuron/vae_tools.py): tools for analyzing (V)AE style models  
- [plot](neuron/plot.py): plotting tools, mostly for debugging models  


### Requirements:
- tensorflow, keras and all of their requirements (e.g. hyp5) 
- numpy, scipy, tqdm  
- [pytools lib](https://github.com/adalca/pytools-lib)
 
### Papers:
If you use this code, please cite:

**Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation**  
[Adrian V. Dalca](http://adalca.mit.edu), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
CVPR 2018.  
[ [PDF](http://www.mit.edu/~adalca/files/papers/cvpr2018_priors.pdf) | [arxiv](http://arxiv.org/abs/1903.03148) | [bibtex](bibtex.txt) ]

If you are using any of the sparse/imputation functions, please cite:  

**Unsupervised Data Imputation via Variational Inference of Deep Subspaces**  
[Adrian V. Dalca](http://adalca.mit.edu), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
Arxiv preprint 2019  
[ [arxiv](https://arxiv.org/abs/1903.03503) | [bibtex](bibtex.txt) ]


### Development:
Please open an [issue](https://github.com/adalca/neuron/issues) [preferred] or contact Adrian Dalca at adalca@csail.mit.edu for question related to `neuron`.


### Use/demos:
Parts of `neuron` were used in [VoxelMorph](http://voxelmorph.mit.edu) and [brainstorm](https://github.com/xamyzhao/brainstorm/), which we encourage you to check out!
