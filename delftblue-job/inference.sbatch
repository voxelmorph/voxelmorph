#!/bin/sh
#
#SBATCH --job-name="voxelmorph-group"
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G

module load 2022r2 openmpi miniconda3

conda activate regist

cd $HOME/scripts/voxelmorph-test


srun python scripts/inference.py weight=0.3 rpca_rank.rank1=10 rpca_rank.rank2=10 rpca_rank.rank3=10 hydra.verbose=__main__ 
# Measure GPU usage of your job (result)


