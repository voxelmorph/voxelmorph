# commands to run voxelmorph: Learning-Based Image Registration  

**voxelmorph** is a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations.

# Command to train the network with atlas.

From the voxelmorph/scripts/ directory.

python train.py --img-list D:\Harsha\Masterarbeit\src\list.txt --atlas D:\Harsha\Masterarbeit\dataset\atlas\np\scaled\atlas\atlas.npz --model-dir D:\Harsha\Masterarbeit\out\model --load-weights D:\Harsha\Masterarbeit\dataset\pre_trained-weights\pre_trained-weights\vxm_dense_brain_T1_3D_mse.h5