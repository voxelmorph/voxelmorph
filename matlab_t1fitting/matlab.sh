#!/bin/bash

#SBATCH --job-name="pforTest"
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=compute

module load matlab
matlab -r parEigenLocal

cd $HOME/scripts/voxelmorph-test/matlab_t1fitting
srun matlab -batch "run('molli_mona.m'); exi