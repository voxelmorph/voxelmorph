#!/bin/sh
#
#SBATCH --job-name="voxelmorph-group"
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

module load 2022r2 openmpi miniconda3

conda activate regist

cd $HOME/scripts/voxelmorph-test


srun python scripts/inference.py hydra.verbose=__main__ \
            weight=0.001 image_loss=nmi norm=smooth  \
            rpca_rank.rank1=11 rpca_rank.rank2=0 rpca_rank.rank3=0 \
            rpca_rank.rank4=0 rpca_rank.rank5=0 rpca_rank.rank6=0 \
            rpca_rank.rank7=0 epochs=240 gpu=0 dataset=MOLLI_post


path=results/MOLLI_pre/Group/rank_11_0_0_0_0_0_0/nmi/smooth/image_loss_weight1/cycle_loss_weight0.01/weight0.001/bspline/cps4_svfsteps7_svfscale1/e240/test_MOLLI_post
srun python scripts/nii2mat.py --source ${path}/round1/moved --mat ${path}/round1/moved_mat
# srun python scripts/nii2mat.py --source ${path}/round2/moved --mat ${path}/round2/moved_mat
# srun python scripts/nii2mat.py --source ${path}/round3/moved --mat ${path}/round3/moved_mat
# srun python scripts/nii2mat.py --source ${path}/round4/moved --mat ${path}/round4/moved_mat
# srun python scripts/nii2mat.py --source ${path}/round5/moved --mat ${path}/round5/moved_mat
# srun python scripts/nii2mat.py --source ${path}/round6/moved --mat ${path}/round6/moved_mat
# srun python scripts/nii2mat.py --source ${path}/round7/moved --mat ${path}/round7/moved_mat
# # Measure GPU usage of your job (result)


