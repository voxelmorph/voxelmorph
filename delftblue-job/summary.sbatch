#!/bin/sh
#
#SBATCH --job-name="voxelmorph-group"
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

module load 2022r2 openmpi miniconda3

conda activate regist

cd $HOME/scripts/voxelmorph-test


srun srun python scripts/inference.py hydra.verbose=__main__ weight=0.3 rpca_rank.rank1=10

# Measure GPU usage of your job (result)


