#!/bin/sh
#
#SBATCH --job-name="voxelmorph-group"
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G

module load 2022r2 openmpi miniconda3

conda activate regist

cd $HOME/scripts/voxelmorph-test


srun python scripts/inference.py hydra.verbose=__main__ hydra.verbose=__main__ \
                    weight=0.3 rpca_rank.rank1=10 rpca_rank.rank2=5 rpca_rank.rank3=3

rank=10_5_3
path=results/MOLLI_pre/group/rank_${rank}/jointcorrelation/l2/image_loss_weight1/weight0.3/bspline/cps4_svfsteps7_svfscale1/e80/test_MOLLI_pre
srun python scripts/nii2mat.py --source ${path}/round1/moved --mat ${path}/round1/moved_mat
srun python scripts/nii2mat.py --source ${path}/round2/moved --mat ${path}/round2/moved_mat
srun python scripts/nii2mat.py --source ${path}/round3/moved --mat ${path}/round3/moved_mat
# Measure GPU usage of your job (result)


