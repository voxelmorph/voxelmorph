# Unified Segmentation

We recently published a method on deep learning methods for unsupervised segmentation that makes use of voxelmorph infrastructure. We provide the atlases used during training. 

# Instructions

## Setup
It might be useful to have each folder inside the `ext` folder on your python path. 
assuming voxelmorph is setup at `/path/to/voxelmorph/`:

```
export PYTHONPATH=$PYTHONPATH:/path/to/voxelmorph/ext/neuron/:/path/to/voxelmorph/ext/pynd-lib/:/path/to/voxelmorph/ext/pytools-lib/
```

If you would like to train/test your own model, you will likely need to write some of the data loading code in 'datagenerator.py' for your own datasets and data formats. There are several hard-coded elements related to data preprocessing and format. 


## Usage:

```
python train_unsupervised_segmentation.py /path/to/your/volume/data/
python test_unsupervised_segmentation.py input_file.nii output_seg_filename.nii
```

## Paper:

If you use the code, please cite (see [bibtex](../citations.bib)):

-  **Unsupervised deep learning for Bayesian brain MRI segmentation**  
[Adrian V. Dalca](http://adalca.mit.edu), [Evan Yu](https://www.bme.cornell.edu/research/grad-students/evan-yu), [Polina Golland](https://people.csail.mit.edu/polina/), [Bruce Fischl](https://www.martinos.org/user/5499), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [Juan E. Iglesias](https://sites.google.com/site/jeiglesias/)  
Under Review. [eprint arXiv:1904.11319](https://arxiv.org/abs/1904.11319)

# Contact:
For any problems or questions please [open an issue](https://github.com/voxelmorph/voxelmorph/issues/new?labels=unifiedseg) in github (preferred).  
Alternatively, please contact us at unifiedseg@mit.edu, but our response might be slower through this route.
