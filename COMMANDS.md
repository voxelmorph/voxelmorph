# commands to run voxelmorph: Learning-Based Image Registration  

**voxelmorph** is a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations.

# Command to train the network with atlas.
```py
/home/students/yogeshappa/miniconda3/bin/python3 /home/students/yogeshappa/repo/Masterarbeit/voxelmorph/scripts/tf/train.py
--img-list /home/students/yogeshappa/repo/Masterarbeit/src/linux/list.txt
--atlas /home/students/yogeshappa/repo/Masterarbeit/dataset/atlas/np_atlas_scaled.npz
--model-dir /home/students/yogeshappa/repo/Masterarbeit/out/model
--epochs 900
--steps-per-epoch 118

# steps_per_epoch = len(training_data) / batch_size.
```
