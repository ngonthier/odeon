#!/bin/bash
#SBATCH --job-name=dchan_siam_30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --partition=jean-zellou

#SBATCH -e /var/data/shared/dchan/slurm/%x-%j.err
#SBATCH -o /var/data/shared/dchan/slurm/%x-%j.out

# Compulsory
# NOTE: Beginning with 22.05, srun will not inherit the --cpus-per-task value requested by salloc or sbatch. 
# It must be requested again with the call to srun or set with the SRUN_CPUS_PER_TASK environment variable if desired for the task(s).
export SRUN_CPUS_PER_TASK=16 # = NB CPU per task

# For debug purposes (optional)
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=TRACE
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1

# echo launched commands
set -x

export PYTHONPATH=/mnt/stores/store-DAI/equipiers/pvoitot/dchan/odeon
echo "PYTHONPATH ${PYTHONPATH}"

export BATCH_SIZE=20
export TRAIN_LR=0.0001
export TRAIN_ENCODER_NAME=resnet50
export MLFLOW_EXPERIMENT_NAME=dchan_siam
export TRAIN_OPTIMIZER=adam
#export MLFLOW_TRACKING_URI="http://smlpslurmmft1.ign.fr:8000"
#export MLFLOW_ARTIFACT_LOCATION="file:/mnt/stores/store-DAI/pocs/slurm/dchan/mlruns/models"
export MLFLOW_SAVE_DIR="/mnt/stores/store-DAI/pocs/slurm/dchan/mlruns"
export TENSORBOARD_LOCATION="/mnt/stores/store-DAI/pocs/slurm/dchan/tensorboard"
export TRAIN_TRANSFORM=True
export TRAIN_DATASETS=train,val,test
export TRAIN_LOG=True
#export TRAIN_MAX_EPOCHS=5
export TRAIN_ENCODER_WEIGHTS=imagenet
export TRAIN_SEED=42
export TRAIN_LOSS=focal

cd $PYTHONPATH
srun -vv mamba run --no-capture-output -n dchan_odeon python xp/dchan_siam.py
