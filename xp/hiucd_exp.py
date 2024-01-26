# Code pour lancer et charger Hi UCD mini dataset
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything
import albumentations as A
from typing import Dict

import os
import sys
sys.path.append('../odeon')

from odeon.data.data_module import Input
from odeon.models.change.module.change_unet import ChangeUnet

root: str = r'\\store\store-DAI\datasrc\dchan\hiucd_mini' #\train\image\2017\9"
if not os.path.exists(root):
    root: str = r"C:\Users\NGonthier\Documents\Detection_changement\data\hiucd_mini"
root_dir: str = root

dataset: str = os.path.join(root, 'train.csv')
batch_size = 2
input_fields : Dict = {"T0": {"name": "T0", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                       "T1": {"name": "T1", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                       "mask": {"name": "change", "type": "mask", "encoding": "integer", "band_indices": [3], "post_process": "add", "added_value": -1}, 
                       "sem_T0": {"name": "change", "type": "mask", "encoding": "integer", "band_indices": [1], "post_process": "add", "added_value": -1}, 
                       "sem_T1": {"name": "change", "type": "mask", "encoding": "integer", "band_indices": [2], "post_process": "add", "added_value": -1}} 
# On HiUCD : The first channel of change is  T1 land cover labels, the second is T2 land cover labels and the last is change labels.

train_img_size = 128

transform = [A.RandomResizedCrop(height=train_img_size,
                                    width=train_img_size,
                                    scale=(0.8, 1.2), p=1.0),
             A.RandomRotate90(p=0.5),
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.75)]
fit_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": batch_size, "num_workers": 8},
                               'input_file': dataset,
                               'root_dir': root_dir,
                               'transform': transform
              } # add transform for data augment
val_dataset: str = os.path.join(root, 'val.csv')
val_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": batch_size, "num_workers": 8},
                               'input_file': val_dataset,
                               'root_dir': root_dir
              }
test_dataset: str = os.path.join(root, 'test.csv')
test_params = {'input_fields': input_fields,
                               'dataloader_options' : {"batch_size": batch_size, "num_workers": 8},
                               'input_file': test_dataset,
                               'root_dir': root_dir
              }

input = Input(fit_params=fit_params,
              validate_params=val_params,
              test_params=test_params)
encoder_name = "resnet18"
model = ChangeUnet(model='fc_siam_conc', loss='bce', ignore_index=-1, scheduler='ExponentialLR', lr=0.001, model_params={"encoder_weights": None, "encoder_name": encoder_name})#"imagenet"
# need to replace the value by -1 et avoir ignore_index=-1 ? TODO
path_model_checkpoint = 'ckpt' # Need to specify by run, no ?
save_top_k_models = 1
path_model_log = ''
accelerator = 'gpu' # 'gpu'
limit_train_batches = 2
limit_val_batches = 2
limit_test_batches = 2
max_epochs = 2
check_val_every_n_epoch = 1
log_every_n_steps = 1
def main():
    seed_everything(42, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(dirpath=path_model_checkpoint,
                                       save_top_k=save_top_k_models,
                                       filename='epoch-{epoch}-loss-{val_bin_iou:.2f}',
                                       mode="max",
                                       monitor='val_bin_iou')
    callbacks = [lr_monitor, model_checkpoint]
    logger = pl_loggers.TensorBoardLogger(save_dir=path_model_log, version='test_unit')
    trainer = Trainer(logger=logger, callbacks=callbacks, accelerator=accelerator, max_epochs=max_epochs,
                      limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
                      limit_test_batches=limit_test_batches,
                      log_every_n_steps=log_every_n_steps) 
    trainer.fit(model=model, datamodule=input)
    trainer.validate(model=model, datamodule=input)
    trainer.test(model=model, datamodule=input)


if __name__ == '__main__':

    main()
