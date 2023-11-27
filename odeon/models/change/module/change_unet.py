"""Segmentation tasks"""

# import warnings
from functools import partial
import random
from typing import Any, Callable, Dict, Optional, Tuple, cast, List

# import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor
# from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, LinearLR, PolynomialLR, ChainedScheduler
from torch.utils.data import DataLoader
# from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics.classification import (  # type: ignore[attr-defined]
    BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision,
    BinaryRecall, BinarySpecificity, BinaryMatthewsCorrCoef)
from pytorch_lightning.loggers import TensorBoardLogger
import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

from odeon.core.types import OdnMetric
from odeon.losses.dice import DiceLoss
from odeon.models.change.arch.change_unet import FCSiamConc, FCSiamDiff
from odeon.models.change.arch.changeformer.ChangeFormer import ChangeFormerV6
# from odeon.models.change.arch.fc_siam_conc_original import FCSiamConcOriginal
# from odeon.models.change.arch.fc_siam_conc import OrigFCSiamConc
from odeon.models.core.models import ModelRegistry
# from mmseg.models.utils import resize


import warnings

import torch.nn as nn
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"  # Sphinx bug


@ModelRegistry.register(name='change_unet', aliases=['c_unet'])
class ChangeUnet(pl.LightningModule):
    """

    """
    def __init__(self,
                 model: str = 'fc_siam_conc',
                 model_params: Optional[Dict] = None,
                 loss: str | List[str] = 'bce',
                 lr: float = 0.0001,
                 threshold: float = 0.5,
                 scheduler='ReduceLROnPlateau',
                 optimizer='adam',
                 weight: Optional[List] = None,
                 random_swap: float = 0.0,
                 use_syncbn: bool = False,
                 **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.model = self.configure_model(model=model, model_params=model_params)
        if use_syncbn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if isinstance(loss, str):
            self.losses = [self.configure_loss(loss=loss, weight=weight)]
        elif(loss, List[str]):
            self.losses = self.configure_losses(loss, weight)

        self.train_metrics, self.val_metrics, self.test_metrics = self.configure_metrics(metric_params={})
        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.lr = lr
        self.activation: Callable[[Tensor], Tensor] = torch.sigmoid # Warning if you use something else than bce for this model !!!
        self.threshold: float = threshold
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.random_swap = random_swap
        """"
        if not isinstance(kwargs["ignore_index"], (int, type(None))):
            raise ValueError("ignore_index must be an int or None")
        if (kwargs["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )
        self.ignore_index = kwargs["ignore_index"]
        """

    def configure_model(self,
                        model: str = 'fc_siam_conc',
                        model_params: Optional[Dict] = None) -> nn.Module:
        """
        Configures the task based on kwargs parameters passed to the constructor.
        Parameters
        ----------
        model
        model_params

        Returns
        -------

        """

        if model_params is None:
            model_params = {}
        if model == "fc_siam_diff":
            return FCSiamDiff(**model_params)
        elif model == "fc_siam_conc":
            return FCSiamConc(**model_params)
        # elif model == "fc_siam_conc_original":
        #     return FCSiamConcOriginal(**model_params)
        elif model == "change_former":
            ps = {k: v for k, v in model_params.items() if k != "encoder_weights" }
            m = ChangeFormerV6(**ps)
            if "encoder_weights" in model_params:
                ckpt_path = str(model_params["encoder_weights"])
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                m.load_state_dict(checkpoint['model_G_state_dict'])
            return m
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                f"Currently, only supports 'unet'."
            )

    def configure_loss(self,
                       loss: str, weight: Optional[List] =  None) -> nn.Module:
        wt: Optional[Tensor] = None
        if weight is not None:
            wt = torch.Tensor(weight)
        if loss == "bce":
            # ignore_value = -1000 if self.ignore_index is None else self.ignore_index
            self.need_apply_sigmoid = True
            return nn.BCEWithLogitsLoss(reduction='mean', pos_weight=wt) # This loss combines a Sigmoid layer and the BCELoss in one single class.
        elif loss == "focal":
            return smp.losses.FocalLoss("binary", normalized=True)
        elif loss == "ce":
            return nn.CrossEntropyLoss(reduction='mean', ignore_index=255, weight=wt)        
        elif loss == "dice":
            return DiceLoss()
        else:
            raise ValueError(f"Loss type '{loss}' is not valid. "
                             f"Currently, supports 'bce', or 'focal' loss.")

    def configure_losses(self,
                       losses: List[str], weight: Optional[List] =  None) -> List[nn.Module]:
        ls: List[nn.Module] = []
        for l in losses:
            loss = self.configure_loss(l, weight).to(self.device)
            ls.append(loss)

        return ls
        

    def configure_lr(self,
                     lr: float,
                     optimizer: str = 'sgd',
                     scheduler: str = '',
                     differential: Optional[Dict[str, float]] = None):
        ...

    def configure_metrics(self, metric_params: Dict) -> Tuple[OdnMetric, OdnMetric, OdnMetric]:

        train_metrics = MetricCollection(
            {"bin_acc": BinaryAccuracy(),
             "bin_iou": BinaryJaccardIndex(),
             "bin_rec": BinaryRecall(),
             "bin_spec": BinarySpecificity(),
             "bin_pre": BinaryPrecision(),
             "bin_f1": BinaryF1Score(),
            #  "bin_mcc": BinaryMatthewsCorrCoef()
             },
            prefix="train_")
        val_metrics = train_metrics.clone(prefix="val_")
        test_metrics = train_metrics.clone(prefix="test_")
        return train_metrics, val_metrics, test_metrics

    def forward(self, T0: Tensor, T1: Tensor, *args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        T0
        T1
        args
        kwargs

        Returns
        -------

        """
        x = torch.stack(tensors=(T0, T1), dim=1)
        return self.model(x)

    def configure_activation(self, activation: str, dim=1) -> Callable[[Tensor], Tensor]:
        match activation:
            case 'softmax':
                return partial(torch.softmax, dim=1)
            case 'sigmoid':
                return torch.sigmoid
            case _:
                raise RuntimeError('something went in configuration activation')

    def step(self, batch: Dict, rnd: bool = False) -> Any:
        T0 = batch['T0']
        T1 = batch['T1']
        if rnd and random.random() < self.random_swap:
            tmp = T1
            T1 = T0
            T0 = tmp
        y = batch['mask']
        y_hat = self(T0=T0, T1=T1)

        return y_hat, y

    def post_compute(self, y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if y_hat.shape[2] != y.shape[2] or y_hat.shape[3] != y.shape[3]:
            y_hat = resize(
                input=y_hat,
                size=y.shape[2:],
                mode='bilinear',
                align_corners=False)
            
        if y_hat.shape[1] > 1:
            y_hat_hard = torch.argmax(y_hat, dim=1)
            y_out = y.squeeze(1).long()
        else:
            # no need of preliminary sigmoid, in binary case, we use BCEWithLogits
            y_hat_hard = (y_hat > self.threshold).to(y.device)
            y_out = y.float()

        # trick to not care about loss tensor init
        loss = self.losses[0](y_hat, y_out)

        if len(self.losses) > 1:
            for loss_fn in self.losses[1:]:
                loss += loss_fn(y_hat, y_out)

        return (loss, y_hat_hard, y_out)

    def training_step(self, batch: Dict[str, Any], batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        batch
        args
        kwargs

        Returns
        -------

        """
        # multi-class output
        y_hat, y = self.step(batch=batch, rnd=True)
        loss, y_hat_hard, y_out = self.post_compute(y, y_hat)
        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y_out)
        # debug = False
        # if debug:
        #     if batch_idx < 6: # Only on batch 0 TODO : need random samples but still the same
        #         y_hat = self.activation(y_hat)
        #         self.log_tb_images((batch['T0'], batch['T1'], y, y_hat, [batch_idx]*len(y)), step=self.global_step, set='train')
        return {'loss': loss}

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Parameters
        ----------
        outputs: list of items returned by training_step

        Returns
        -------

        """

        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        batch
        args
        kwargs

        Returns
        -------

        """
        y_hat, y = self.step(batch=batch)
        loss, y_hat_hard, y_out = self.post_compute(y, y_hat)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y_out)
        # if batch_idx == 0: # Only on batch 0 TODO : need random samples but still the same
        #     y_hat = self.activation(y_hat)
        #     self.log_tb_images((batch['T0'], batch['T1'], y, y_hat, [batch_idx]*len(y)), step=self.global_step, set='val')
        return {'val_loss': cast(Tensor, loss)}

    def log_tb_images(self, viz_batch, step, set='') -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            # raise ValueError('TensorBoard Logger not found')
            return

        for img_idx, (T0, T1, y_true, y_pred, batch_idx) in enumerate(zip(*viz_batch)):
            # Create one single image with the 4 elements
            figure = self.image_line([T0, T1, y_true, y_pred])
            image = self.plot_to_image(figure)
            tb_logger.add_image(f"Image {batch_idx}_{img_idx}_{set}", image, step)

    def image_line(self, list_images):
        """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
        # Create a figure to contain the plot.
        figure = plt.figure(figsize=(8, 3))
        list_titles = ['T0', 'T1', 'GroundTruth', 'Prediction']
        for i, title in enumerate(list_titles):
            # Start next subplot.
            plt.subplot(1, 4, i + 1, title=title)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            img = list_images[i].detach().cpu().permute(1, 2, 0).numpy()
            plt.imshow(img, interpolation='nearest')
            plt.tight_layout()

        return figure

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image) #.unsqueeze(0)
        return image

    def validation_epoch_end(self, outputs: Any) -> None:
        """

        Parameters
        ----------
        outputs

        Returns
        -------

        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        batch
        args
        kwargs

        Returns
        -------

        """
        y_hat, y = self.step(batch=batch)
        loss, y_hat_hard, y_out = self.post_compute(y, y_hat)
        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y_out)
        # if batch_idx == 0: # Only on batch 0 TODO : need random samples
        #     y_hat = self.activation(y_hat)
        #     self.log_tb_images((batch['T0'], batch['T1'], y, y_hat, [batch_idx]*len(y)), step=self.global_step, set='test')
        return {'test_loss': cast(Tensor, loss)}

    def test_epoch_end(self, outputs: Any) -> None:
        """
        Parameters
        ----------
        outputs

        Returns
        -------

        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def get_lr_scheduler(self, optimizer):
        if self.scheduler == 'ReduceLROnPlateau':
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer, patience=20),
                "monitor": "val_loss",
            }
        elif self.scheduler == 'ExponentialLR':
            lr_scheduler = {"scheduler": ExponentialLR(optimizer, 0.95),
                            "monitor": "val_loss",
                            }
        elif self.scheduler == 'CosineAnnealingLR':
            lr_scheduler = {"scheduler": CosineAnnealingLR(optimizer, T_max=10),
                            "monitor": "val_loss",
                            }
        elif self.scheduler == 'LinearLR+PolyLR':
            sch = ChainedScheduler([
                LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=1000),
                PolynomialLR(optimizer, power=1.0, total_iters=39000)
            ])              
            lr_scheduler = {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "step"
            }
        elif self.scheduler is None:
            lr_scheduler = None
        else:
            raise ValueError(
                f"LR Scheduler '{self.scheduler}' is unknown."
            )

        return lr_scheduler

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.
        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=1e-4
            )
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")        
        # else:
        #     optimizer = torch.optim.Adam(
        #         self.model.parameters(), lr=self.lr, weight_decay=1e-4
        #     )
        
        sch = self.get_lr_scheduler(optimizer)
        if sch is None:
            return {
                "optimizer": optimizer
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": sch
            }
