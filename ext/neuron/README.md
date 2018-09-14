# neuron
A Neural networks toolbox for anatomical  image analysis

This toolbox is **currently in development**, with the goal providing a set of tools with infrastructure for medical image analysis with neural network. While the tools are somewhat general, `neuron` will generally run with `keras` on top of `tensorflow`.

### Main tools
`callbacks`: a set of callbacks during keras training to help with understanding your fit, such as Dice measurements and volume-segmentation overlaps  
`generators`: generators for medical image volumes and various combinations of volumes, segmentation, categorical and other output  
`dataproc`: a set of tools for processing medical imaging data for preparation for training/testing  
`metrics`: metrics (most of which can be used as loss functions), such as dice or weighted categorical crossentropy.  
`models`: a set of flexible models (many parameters to play with...) particularly useful in medical image analysis, such as a U-net/hourglass model and a standard classifier. 
`layers`: a few simple layers
`plot`: plotting tools, mostly for debugging models
`utils`: various utilities useful in debugging.

Other utilities and a few `jupyter` notebooks are also provided.

### Requirements:
- tensorflow  
- keras and all of its requirements (e.g. hyp5) 
- numpy, scipy  
- tqdm  
- [python libraries](https://github.com/search?q=user%3Aadalca+topic%3Apython) from @adalca github account  
 
### Development:
Please contact Adrian Dalca, adalca@csail.mit.edu for question related to `neuron`

### Papers:
**Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation**  
AV Dalca, J Guttag, MR Sabuncu  
*Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.*

**Spatial Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation**  
A.V. Dalca, J. Guttag, and M. R. Sabuncu  
*NIPS ML4H: Machine Learning for Health. 2017.* 
