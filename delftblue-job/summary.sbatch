#!/bin/sh
#
#SBATCH --job-name="voxelmorph-group"
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16G

module load 2022r2 openmpi miniconda3

conda activate regist

cd $HOME/scripts/voxelmorph-test


srun srun python scripts/inference.py hydra.verbose=__main__ hydra.verbose=__main__ \
                    weight=0.3 rpca_rank.rank1=5 rpca_rank.rank2=10 rpca_rank.rank3=20 \
                    dataset=MOLLI_post
# Measure GPU usage of your job (result)


