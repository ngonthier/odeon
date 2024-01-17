#!/bin/bash
#SBATCH --job-name=dchan_levir_15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --partition=jean-zellou
#SBATCH --mem=100G

#SBATCH -e /var/data/shared/dchan/slurm/%x-%j.err
#SBATCH -o /var/data/shared/dchan/slurm/%x-%j.out

# Compulsory
# NOTE: Beginning with 22.05, srun will not inherit the --cpus-per-task value requested by salloc or sbatch. 
# It must be requested again with the call to srun or set with the SRUN_CPUS_PER_TASK environment variable if desired for the task(s).
export SRUN_CPUS_PER_TASK=12 # = NB CPU per task

# For debug purposes (optional)
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=TRACE
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1

# echo launched commands
set -x

export ODEON_PATH="/mnt/stores/store-DAI/equipiers/pvoitot/dchan/odeon"
export OPENCD_PATH="/mnt/stores/store-DAI/equipiers/pvoitot/dchan/open-cd"
export PYTHONPATH=$PYTHONPATH:$ODEON_PATH:$OPENCD_PATH
echo "PYTHONPATH ${PYTHONPATH}"

export BATCH_SIZE=12
export TRAIN_LR=0.001
export TRAIN_DATASET_NAME=levir-cd
export TRAIN_ROOT_DIR="/var/data/datasets/levir-cd"
export TRAIN_MODEL_TYPE=odeon
export TRAIN_MODEL=fc_siam_conc

# export TRAIN_ENCODER_NAME=resnet34
export MLFLOW_EXPERIMENT_NAME=dchan_levir
export TRAIN_OPTIMIZER=adamw
#export MLFLOW_TRACKING_URI="http://smlpslurmmft1.ign.fr:8000"
#export MLFLOW_ARTIFACT_LOCATION="file:/mnt/stores/store-DAI/pocs/slurm/dchan/mlruns/models"
export MLFLOW_SAVE_DIR="/mnt/stores/store-DAI/pocs/slurm/dchan/mlruns"
export TENSORBOARD_LOCATION="/mnt/stores/store-DAI/pocs/slurm/dchan/tensorboard"
export TRAIN_TRANSFORM=True
export TRAIN_TRANSFORM_TYPE=4
export TRAIN_DATASETS=train,val,test
export TRAIN_LOG=True
export TRAIN_MAX_EPOCHS=800
# export TRAIN_ENCODER_WEIGHTS=imagenet
export TRAIN_SEED=42
export TRAIN_LOSS=bce
export TRAIN_IMG_SIZE=1024
export TRAIN_CROP_SIZE=512
export TRAIN_LR_SCHEDULER="LinearLR+PolyLR"
export CUDNN_BENCHMARK=True
export TRAIN_USE_SYNCBN=True

cd $ODEON_PATH
srun -vv mamba run --no-capture-output -n dchan_odeon python xp/dchan.py
