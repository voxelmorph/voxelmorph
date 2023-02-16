#!/bin/sh
#
#SBATCH --job-name="voxelmorph-group"
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G

module load 2022r2 openmpi miniconda3

conda activate regist

cd $HOME/scripts/voxelmorph-test


srun python scripts/inference.py hydra.verbose=__main__ \
            weight=0.001 image_loss=ncc norm=smooth  \
            dataset=MOLLI_post


path=results/MOLLI_pre/group/rank_5_5_5_5_5_5_5/ncc/smooth/image_loss_weight1/cycle_loss_weight0.01/weight0.001/bspline/cps4_svfsteps7_svfscale1/e80/test_MOLLI_post
srun python scripts/nii2mat.py --source ${path}/round1/moved --mat ${path}/round1/moved_mat
srun python scripts/nii2mat.py --source ${path}/round2/moved --mat ${path}/round2/moved_mat
srun python scripts/nii2mat.py --source ${path}/round3/moved --mat ${path}/round3/moved_mat
srun python scripts/nii2mat.py --source ${path}/round4/moved --mat ${path}/round4/moved_mat
srun python scripts/nii2mat.py --source ${path}/round5/moved --mat ${path}/round5/moved_mat
srun python scripts/nii2mat.py --source ${path}/round6/moved --mat ${path}/round6/moved_mat
srun python scripts/nii2mat.py --source ${path}/round7/moved --mat ${path}/round7/moved_mat
# Measure GPU usage of your job (result)


