#!/bin/bash
#SBATCH --job-name=dchan_hiucd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --partition=jean-zellou

#SBATCH -e /var/data/shared/dchan/slurm/%x-%j.err
#SBATCH -o /var/data/shared/dchan/slurm/%x-%j.out

# Compulsory
# NOTE: Beginning with 22.05, srun will not inherit the --cpus-per-task value requested by salloc or sbatch. 
# It must be requested again with the call to srun or set with the SRUN_CPUS_PER_TASK environment variable if desired for the task(s).
export SRUN_CPUS_PER_TASK=12 # = NB CPU per task

# For debug purposes (optional)
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=TRAC
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1

# echo launched commands
set -x

export ODEON_PATH="/mnt/stores/store-DAI/equipiers/ngonthier/odeon"
export ENVS_PATH="/envs/dchan_ng"
export PYTHONPATH=$PYTHONPATH:$ODEON_PATH:$ENVS_PATH
echo "PYTHONPATH ${PYTHONPATH}"

export BATCH_SIZE=16
export TRAIN_LR=0.0001
export TRAIN_ROOT_DIR="/mnt/stores/store-DAI/datasrc/dchan/hiucd_mini"
export TRAIN_MODEL_TYPE=odeon
export TRAIN_ENCODER_NAME=resnet34
export CLEARML_EXPERIMENT_NAME=dchan_hiucd
export TRAIN_MODEL=fc_siam_conc
export TRAIN_OPTIMIZER=adamw
export TENSORBOARD_LOCATION="/mnt/stores/store-DAI/pocs/slurm/dchan/tensorboard_ngonthier"
export TRAIN_TRANSFORM=True
export TRAIN_TRANSFORM_TYPE=4
export TRAIN_DATASETS=train,val,test
export TRAIN_LOG=True
export TRAIN_MAX_EPOCHS=100
export TRAIN_ENCODER_WEIGHTS=imagenet
export TRAIN_SEED=42
export TRAIN_LOSS=bce
export TRAIN_IMG_SIZE=512
export CLEARML_CACHE_DIR="/var/data/shared/clearml/cache"
export CLEARML_CONFIG_FILE="/var/data/shared/clearml/clearml_dchan.conf"
export CLEARML_API_DEFAULT_REQ_METHOD=POST

# Activate desired env
eval "$(conda shell.bash hook)"
conda activate dchan_ng

cd $ODEON_PATH
srun -vv python xp/dchan_hiucd.py

echo "Job complete"
