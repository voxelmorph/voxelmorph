# commands to run voxelmorph: Learning-Based Image Registration  

**voxelmorph** is a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations.

# Command to train the network with atlas.
```py
/home/students/yogeshappa/miniconda3/bin/python3 /home/students/yogeshappa/repo/Masterarbeit/voxelmorph/scripts/tf/train.py
--img-list /home/students/yogeshappa/repo/Masterarbeit/src/linux/list.txt
--atlas /home/students/yogeshappa/repo/Masterarbeit/dataset/atlas/np_atlas_scaled.npz
--model-dir /work/scratch/yogeshappa/tensorflow_out/model
--epochs 900
--steps-per-epoch 59

# steps_per_epoch = len(training_data) / batch_size.
# len(training_data) = 59.
```

# Command to register an image with atlas.
```py
D:\Harsha\repo\Masterarbeit\voxelmorph\scripts\tf\register.py
--moving I:\masterarbeit_dataset\data\npz\np_brain3_scaled.npz
--fixed I:\masterarbeit_dataset\atlas\np_atlas_scaled.npz
--moved I:\tensorflow_out\out\moved.npz
--model I:\tensorflow_out\model\0223.h5
--gpu 0
```
