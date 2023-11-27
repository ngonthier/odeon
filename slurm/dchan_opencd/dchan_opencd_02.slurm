#!/bin/bash
#SBATCH --job-name=dchan_opencd_02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --partition=jean-zellou
#SBATCH --mem=100G

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

export ODEON_PATH="/mnt/stores/store-DAI/equipiers/pvoitot/dchan/odeon"
export OPENCD_PATH="/mnt/stores/store-DAI/equipiers/pvoitot/dchan/open-cd"
export PYTHONPATH=$PYTHONPATH:$ODEON_PATH:$OPENCD_PATH
echo "PYTHONPATH ${PYTHONPATH}"

export BATCH_SIZE=10
export TRAIN_LR=0.0001
export TRAIN_ROOT_DIR="/mnt/stores/store-DAI/datasrc/dchan/levir-cd-repatched-256x256"
#export TRAIN_ENCODER_NAME=resnet34
export MLFLOW_EXPERIMENT_NAME=dchan_opencd
export OPENCD_CONFIG_FILE="/mnt/stores/store-DAI/equipiers/pvoitot/dchan/open-cd/configs/changer/changer_ex_mit-b0_512x512_40k_levircd.py"
export TRAIN_OPTIMIZER=adamw
#export MLFLOW_TRACKING_URI="http://smlpslurmmft1.ign.fr:8000"
#export MLFLOW_ARTIFACT_LOCATION="file:/mnt/stores/store-DAI/pocs/slurm/dchan/mlruns/models"
export MLFLOW_SAVE_DIR="/mnt/stores/store-DAI/pocs/slurm/dchan/mlruns"
export TENSORBOARD_LOCATION="/mnt/stores/store-DAI/pocs/slurm/dchan/tensorboard"
export TRAIN_TRANSFORM=True
export TRAIN_TRANSFORM_TYPE=2
export TRAIN_DATASETS=train,val,test
export TRAIN_LOG=True
export TRAIN_MAX_EPOCHS=200
export TRAIN_SEED=42
export TRAIN_LOSS=ce
export TRAIN_IMG_SIZE=512

cd $ODEON_PATH
srun -vv mamba run --no-capture-output -n dchan_odeon python xp/dchan_opencd_levir.py
