#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short

# The default run (wall-clock) time is 1 minute
#SBATCH --time=0:30:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1 (note: CPUs are always allocated to jobs per 2)
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
#SBATCH --cpus-per-task=1

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=8G

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

# Measure GPU usage of your job (initialization)

# Your job commands go below here

# Uncomment these lines and adapt them to load the software that your job requires
module use /opt/insy/modulefiles
module load cuda/11.6 openmpi/4.0.1 miniconda/3.8
pip install torch torchvision torchaudio
pip install packaging==21.3
cd $HOME/script/voxelmorph-test-new
python -m pip install --user -e .

inf_epochs=100
name=MOLLI_post
total_epochs=700
srun python scripts/inference.py hydra.verbose=__main__ \
            weight=0.001 image_loss=ncc norm=smooth \
            rpca_rank.rank1=11 rpca_rank.rank2=0 \
            epochs=$total_epochs +resize=True \
            dataset=$name gpu=0 +inference_epochs=$inf_epochs


path=results/MOLLI_pre/Group/rank_11_0_5_5_5_5_5/ncc/smooth/image_loss_weight1/cycle_loss_weight0.01/weight0.001/bspline/cps4_svfsteps7_svfscale1/e$total_epochs/test_$name_$inf_epochs
srun python scripts/nii2mat.py --source ${path}/round1/moved --mat ${path}/round1/moved_mat
srun python scripts/nii2mat.py --source ${path}/round2/moved --mat ${path}/round2/moved_mat
srun python scripts/nii2mat.py --source ${path}/round3/moved --mat ${path}/round3/moved_mat
srun python scripts/nii2mat.py --source ${path}/round4/moved --mat ${path}/round4/moved_mat
srun python scripts/nii2mat.py --source ${path}/round5/moved --mat ${path}/round5/moved_mat
srun python scripts/nii2mat.py --source ${path}/round6/moved --mat ${path}/round6/moved_mat
srun python scripts/nii2mat.py --source ${path}/round7/moved --mat ${path}/round7/moved_mat
