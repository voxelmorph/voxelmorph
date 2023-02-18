#!/bin/sh
#
#SBATCH --job-name="voxelmorph-group"
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G

module load 2022r2 openmpi miniconda3

conda activate regist

cd $HOME/scripts/voxelmorph-test

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi

srun python scripts/general_run.py hydra.verbose=__main__ \
            weight=0.001 image_loss=nmi norm=smooth  \
            rpca_rank.rank1=11 rpca_rank.rank2=0 rpca_rank.rank3=0 \
            rpca_rank.rank4=0 rpca_rank.rank5=0 rpca_rank.rank6=0 \
            rpca_rank.rank7=0 epochs=240

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"


