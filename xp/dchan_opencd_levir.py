import os
import torch
from typing import Any, Dict, Optional
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything

from albumentations.core.transforms_interface import DualTransform
from odeon.data.data_module import Input
from odeon.models.change.module.change_unet_levir import ChangeUnetLevir
from tb_image_logger import TensorboardGenerativeModelImageSampler
from odeon.models.opencd import OpenCDPlug
import albumentations as A
# from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers.logger import DummyLogger
from mlflow_logger import MLFlowLogger
import mlflow

import logging

def build_params(stage: str, root_dir: str, batch_size: int, num_workers: int, transform: Optional[DualTransform] = None, nb_samples: int = 0, sample_seed = 0) -> Dict:
    dataset: str = os.path.join(root_dir, f'{stage}.csv')

    res = {
        'input_fields': {
            "T0": {"name": "A", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3], "mean": [0.485, 0.456, 0.406], "std": [0.228, 0.224, 0.225]},
            "T1": {"name": "B", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3], "mean": [0.485, 0.456, 0.406], "std": [0.228, 0.224, 0.225]},
            "mask": {"name": "label", "type": "mask", "post_process": "floor_div", "dtype": "uint8"},            
        },
        'dataloader_options' : {"batch_size": batch_size, "num_workers": num_workers},
        'input_file': dataset,
        'root_dir': root_dir,
        'input_files_has_header': 'infer',
        'nb_samples': nb_samples,
        'sample_seed': sample_seed
    }

    if transform:
        res["transform"] = transform
    return res


def main():

    # root: str = "/mnt/stores/store-DAI/datasrc/dchan"
    root: str = "/var/data/datasets"
    root_dir: str = os.path.join(root, 'levir-cd')

    gpus = 0

    if 'TRAIN_ROOT_DIR' in os.environ:
        root_dir = os.environ['TRAIN_ROOT_DIR']

    job_name = "dchan_levir"
    if 'SLURM_JOB_NAME' in os.environ:
        job_name = os.environ['SLURM_JOB_NAME']

    job_id = "0"
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ['SLURM_JOB_ID']

    name = f"{job_name}_{job_id}"
    print(f"Starting Job {name}")

    if 'SLURM_GPUS_ON_NODE' in os.environ:
        gpus = int(os.environ['SLURM_GPUS_ON_NODE'])
    
    nodes = 1
    if 'SLURM_NNODES' in os.environ:
        nodes = int(os.environ['SLURM_NNODES'])

    num_workers = 8
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    
    batch_size = 5
    if 'BATCH_SIZE' in os.environ:
        batch_size = int(os.environ['BATCH_SIZE'])

    train_lr = 0.0001
    if 'TRAIN_LR' in os.environ:
        train_lr = float(os.environ['TRAIN_LR'])

    config_file = os.environ["OPENCD_CONFIG_FILE"]

    # train_model = "fc_siam_conc"
    # if 'TRAIN_MODEL' in os.environ:
    #     train_model = os.environ['TRAIN_MODEL']        

    # encoder_name = "resnet34"
    # if 'TRAIN_ENCODER_NAME' in os.environ:
    #     encoder_name = os.environ['TRAIN_ENCODER_NAME']        

    # encoder_weights = None
    # if 'TRAIN_ENCODER_WEIGHTS' in os.environ:
    #     encoder_weights = os.environ['TRAIN_ENCODER_WEIGHTS']

    train_datasets = ["train", "val", "test"]
    if 'TRAIN_DATASETS' in os.environ:
        train_datasets = os.environ['TRAIN_DATASETS'].split(",")

    train_img_size = 512
    if 'TRAIN_IMG_SIZE' in os.environ:
        train_img_size = int(os.environ['TRAIN_IMG_SIZE'])


    mlflow_experiment_name = "dchan_opencd_levir"
    if 'MLFLOW_EXPERIMENT_NAME' in os.environ:
        mlflow_experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']        

    print("Initializing mlflow connection...")
    # mlflow_tracking_uri = "http://smlpmlftenap1.ign.fr:5000"
    # mlflow_tracking_uri = "http://smlpslurmmft1.ign.fr:8000"
    mlflow_tracking_uri = None
    if 'MLFLOW_TRACKING_URI' in os.environ:
        mlflow_tracking_uri = os.environ['MLFLOW_TRACKING_URI']       

    mlflow_save_dir = None
    if 'MLFLOW_SAVE_DIR' in os.environ:
        mlflow_save_dir = os.environ['MLFLOW_SAVE_DIR']       

    artifact_location = None #"file:/mnt/stores/store-DAI/pocs/slurm/dchan/mlruns"
    if 'MLFLOW_ARTIFACT_LOCATION' in os.environ:
        artifact_location = os.environ['MLFLOW_ARTIFACT_LOCATION']       


    tensorboard_location = "/mnt/stores/store-DAI/pocs/slurm/dchan/tensorboard"
    if 'TENSORBOARD_LOCATION' in os.environ:
        tensorboard_location = os.environ['TENSORBOARD_LOCATION']       


    has_transform = False
    if 'TRAIN_TRANSFORM' in os.environ:
        has_transform = eval(os.environ['TRAIN_TRANSFORM'])

    transform_type = 0
    if 'TRAIN_TRANSFORM_TYPE' in os.environ:
        transform_type = eval(os.environ['TRAIN_TRANSFORM_TYPE'])


    has_log = True
    if 'TRAIN_LOG' in os.environ:
        has_log = eval(os.environ['TRAIN_LOG'])

    max_epochs = 150
    if 'TRAIN_MAX_EPOCHS' in os.environ:
        max_epochs = int(os.environ['TRAIN_MAX_EPOCHS'])

    overfit_batches = 0.0
    if 'TRAIN_OVERFIT_BATCHES' in os.environ:
        overfit_batches = int(os.environ['TRAIN_OVERFIT_BATCHES'])

    lr_scheduler: Optional[str] = None
    if 'TRAIN_LR_SCHEDULER' in os.environ:
        lr_scheduler = os.environ['TRAIN_LR_SCHEDULER']

    optimizer: str = "adam"
    if 'TRAIN_OPTIMIZER' in os.environ:
        optimizer = os.environ['TRAIN_OPTIMIZER']


    loss: str = "bce"
    if 'TRAIN_LOSS' in os.environ:
        loss = os.environ['TRAIN_LOSS']

    seed: int = 42
    if 'TRAIN_SEED' in os.environ:
        seed = int(os.environ['TRAIN_SEED'])

    print("mlflow_experiment_name", mlflow_experiment_name)
    print("mlflow_run_name", name)
    print("lr", train_lr)
    # print("train_model", train_model)
    print("batch_size", batch_size)
    # print("encoder name", encoder_name)
    print("mlflow_tracking_uri", mlflow_tracking_uri)
    print("mlflow_save_dir", mlflow_save_dir)
    print("artifact_location", artifact_location)
    print("tensorboard_location", tensorboard_location)
    print("has_transform", has_transform)
    print("has_log", has_log)
    print("max_epochs", max_epochs)
    print("lr_scheduler", lr_scheduler)
    print("optimizer", optimizer)
    print("loss", loss)
    print("train_datasets", train_datasets)
    print("seed", seed)
    print("transform_type", transform_type)


    path_model_checkpoint = '/var/data/shared/dchan/checkpoints'
    save_top_k_models = 5
    path_tb_log = os.path.join(tensorboard_location, 'logs')

    loggers = [DummyLogger()]
    tags = {}
    mlflow_logger = None
    if has_log:        
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # from mlflow.tracking.context import registry as context_registry

        # tags = context_registry.resolve_tags({})
        # print("Tags", tags)            

        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_tracking_uri,
            save_dir=mlflow_save_dir,
            run_name=name,
            log_model=True,
            # tags=tags,
            artifact_location=artifact_location
        )
        
        os.makedirs(os.path.join(path_tb_log, mlflow_experiment_name, name), exist_ok=True)
        logger = pl_loggers.TensorBoardLogger(save_dir=path_tb_log, name=os.path.join(mlflow_experiment_name, name))

        loggers = [logger, mlflow_logger]

    print("Initializing Seeds...")

    seed_everything(seed)    

    print("Initializing datasets...")

    transform = None
    if has_transform:
        if transform_type == 0:
            transform = [
                A.RandomRotate90(p=0.5),
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.75)
            ]
        elif transform_type == 1:
            transform = [
                A.OneOf([A.RandomResizedCrop(height=train_img_size,
                                             width=train_img_size,
                                             scale=(0.5, 1.5), p=1.0),
                        A.RandomCrop(height=train_img_size,
                                     width=train_img_size)], p=1.0),
                A.RandomRotate90(p=0.5),
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.75),
                A.Transpose(p=0.5),
                A.GaussianBlur(p=0.5, blur_limit=(3, 3)),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5)
            ]
        elif transform_type == 2:
            transform = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomResizedCrop(height=train_img_size,
                                    width=train_img_size,
                                    scale=(0.8, 1.2), p=1.0),
                A.GaussianBlur(p=0.5, blur_limit=(3, 3)),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5)
            ]


    nb_samples = 10
    fit_params = build_params(train_datasets[0], root_dir, batch_size, num_workers, transform, nb_samples, seed)
    val_params = build_params(train_datasets[1], root_dir, batch_size, num_workers, None, nb_samples, seed)
    test_params = build_params(train_datasets[2], root_dir, batch_size, num_workers, None, nb_samples, seed)

    print("train_datasets", train_datasets)
    print("fit_params", fit_params)
    print("val_params", val_params)
    print("test_params", test_params)

    input = Input(
        fit_params=fit_params,
        validate_params=val_params,
        test_params=test_params
    )

    # model_params = {}
    # if train_model != "change_former":
    #     model_params = {
    #         "encoder_weights": encoder_weights,
    #         "encoder_name": encoder_name
    #     }

    # model = ChangeUnetLevir(
    #     model=train_model,
    #     lr=train_lr,
    #     scheduler=lr_scheduler,
    #     optimizer=optimizer,
    #     loss=loss,
    #     model_params = model_params
    # )

    model = OpenCDPlug(
        config_file=config_file,
        loss="ce"
    )

    # early_stopping = EarlyStopping(monitor="val_bin_iou", min_delta=0.0, patience=200)

    # accelerator = 'cpu'
    # if torch.cuda.is_available():
    #     accelerator = 'gpu'

    trainer_args: Dict[str, Any] = {}
    if gpus == 0:
        trainer_args = {
            'accelerator': 'cpu',
        }
    elif gpus > 1:
        trainer_args = {
            'accelerator': 'gpu',
            'devices': gpus,
            'num_nodes': nodes,
            'strategy': 'ddp'
        }
    else:
        trainer_args = {
            'accelerator': 'gpu',
            'devices': gpus
        }
    trainer_args["max_epochs"] = max_epochs
    trainer_args["overfit_batches"] = overfit_batches
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_path = os.path.join(path_model_checkpoint, name)
    os.makedirs(ckpt_path, exist_ok=True)
    model_checkpoint = ModelCheckpoint(dirpath=ckpt_path,
                                       save_top_k=save_top_k_models,
                                       save_last=True,
                                       filename='{name}-epoch-{epoch}-loss-{val_bin_iou:.2f}',
                                       mode="max",
                                       monitor='val_bin_iou')
    
    tb_sampler = TensorboardGenerativeModelImageSampler(input)
    callbacks = [tb_sampler, lr_monitor, model_checkpoint]

    trainer = Trainer(logger=loggers, callbacks=callbacks, **trainer_args)

    if trainer.global_rank == 0 and has_log and mlflow_logger is not None:
        mlflow_logger.log_hyperparams({"batch_size": batch_size})
        
        for k,v in tags.items():
            mlflow_logger.experiment.set_tag(mlflow_logger._run_id, k, v)

        for k,v in trainer_args.items():
            mlflow_logger.experiment.set_tag(mlflow_logger._run_id, k, v)
        
        print("Trainer Args", trainer_args)

    trainer.fit(model=model, datamodule=input)
    trainer.validate(model=model, datamodule=input) # Where are stored the values ?
    trainer.test(model=model, datamodule=input)

if __name__ == '__main__':

    main()
