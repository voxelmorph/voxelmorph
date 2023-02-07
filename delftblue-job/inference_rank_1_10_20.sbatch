#!/bin/sh
#
#SBATCH --job-name="voxelmorph-group"
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

module load 2022r2 openmpi miniconda3

conda activate regist

cd $HOME/scripts/voxelmorph-test

rank=1_10_20
path=results/MOLLI_pre/group/rank_${rank}/jointcorrelation/l2/image_loss_weight1/weight0.3/bspline/cps4_svfsteps7_svfscale1/e100/test_MOLLI_pre/round1
echo "python scripts/calc_t1map_py.py --input ${path}/moved --output ${path}/t1map --inversiontime data/MOLLI_pre_TI.csv"
srun python scripts/calc_t1map_py.py --input ${path}/moved --output ${path}/t1map --inversiontime data/MOLLI_pre_TI.csv

path=results/MOLLI_pre/group/rank_${rank}/jointcorrelation/l2/image_loss_weight1/weight0.3/bspline/cps4_svfsteps7_svfscale1/e100/test_MOLLI_pre/round2
srun python scripts/calc_t1map_py.py --input ${path}/moved --output ${path}/t1map --inversiontime data/MOLLI_pre_TI.csv

path=results/MOLLI_pre/group/rank_${rank}/jointcorrelation/l2/image_loss_weight1/weight0.3/bspline/cps4_svfsteps7_svfscale1/e100/test_MOLLI_pre/round3
srun python scripts/calc_t1map_py.py --input ${path}/moved --output ${path}/t1map --inversiontime data/MOLLI_pre_TI.csv

