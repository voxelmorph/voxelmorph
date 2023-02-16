#!/bin/bash

#SBATCH --job-name="pforTest"
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=compute

module load matlab

cd $HOME/scripts/voxelmorph-test/matlab_t1fitting
matlab -r molli_mona