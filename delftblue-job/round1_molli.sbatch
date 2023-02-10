#!/bin/sh
#
#SBATCH --job-name="matlab_molli"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem-per-cpu=2G

module load matlab

cd $HOME/scripts/voxelmorph-test/matlab_t1fitting

matlab -r molli_round123

