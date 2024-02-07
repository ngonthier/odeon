import io
from typing import Dict, List, Union, TypeVar, cast
import PIL.Image
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt
from matplotlib.colors import hex2color
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from odeon.data.data_module import Data, Input

T = TypeVar('T', torch.Tensor, Dict, List)

def move_to(obj: T, device: torch.device) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return cast(T, res)
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return cast(T, res)
    else:
        raise TypeError("Invalid type for move_to")
    
lut_colors = {
0   : '#FF00FF',
1   : '#db0e9a',
2   : '#938e7b',
3   : '#f80c00',
4   : '#a97101',
5   : '#1553ae',
6   : '#194a26',
7   : '#46e483',
8   : '#f3a60d',
9   : '#660082',
}

def convert_to_color(arr_2d: np.ndarray, palette: dict = lut_colors) -> np.ndarray:
    rgb_palette = {k: tuple(int(i * 255) for i in hex2color(v)) for k, v in palette.items()}
    arr_3d = np.zeros((arr_2d.shape[1], arr_2d.shape[2], 3), dtype=np.uint8)
    for c, i in rgb_palette.items():
        m = arr_2d == c
        arr_3d[m[0]] = i
    return arr_3d

class TensorboardGenerativeModelImageSampler(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation

    Example::

        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler(input)])
    """
    
    def __init__(
        self,
        input: Input,
        multitask: False,
        ignore_index: None
    ) -> None:
        """
        Args:
            input: Input Data Module.
        """

        super().__init__()
        self.input = input
        self.multitask = multitask
        self.ignore_index = ignore_index

    def on_train_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        self.log_image("train", self.input.fit, self.input.fit_params, trainer, model)

    def on_validation_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        self.log_image("val", self.input.validate, self.input.validate_params, trainer, model)

    def on_test_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        self.log_image("test", self.input.test, self.input.test_params, trainer, model)

    def denormalize_img_as_tensor(self, image: torch.Tensor, mean, std):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)

        return image

    def log_image(self, stage: str, data: Data, params: Dict, trainer: Trainer, model: LightningModule) -> None:
        df = data.dataframe
        samples = df[df["sampled"] == True]
        samples_t = []
        mean_T0 = [0.0, 0.0, 0.0]
        std_T0 = [1.0, 1.0, 1.0]
        mean_T1 = [0.0, 0.0, 0.0]
        std_T1 = [1.0, 1.0, 1.0]
        if "mean" in params["input_fields"]["T0"]:
            mean_T0 = params["input_fields"]["T0"]["mean"]
        if "std" in params["input_fields"]["T0"]:
            std_T0 = params["input_fields"]["T0"]["std"]
        if "mean" in params["input_fields"]["T1"]:
            mean_T1 = params["input_fields"]["T1"]["mean"]
        if "std" in params["input_fields"]["T1"]:
            std_T1 = params["input_fields"]["T1"]["std"]

        for i in samples.index.values: # type: ignore
            sample = data.dataset[i]
            sample["idx"] = i
            samples_t.append(sample)

        samples_d: Dict[str, torch.Tensor] = torch.utils.data.default_collate(samples_t) # type: ignore
        samples_d = move_to(samples_d, model.device)

        with torch.no_grad():
            model.eval()
            T0 = samples_d['T0']
            T1 = samples_d['T1']
            y = samples_d['mask']
            y_hat = model(T0, T1)
            if (y_hat.shape[1] > 1):
                y_hat = torch.argmax(y_hat, dim=1, keepdim=True)
            else:
                y_hat = torch.sigmoid(y_hat) # WARNING: how to know activation in general?
            if self.multitask:
                self.log_tb_images_multitask(
                trainer,
                (samples_d['T0'], samples_d['T1'], y, y_hat, samples_d["idx"]),
                ['T0', 'T1', 'GroundTruth', 'Prediction'],
                step=model.global_step,
                mean_T0=mean_T0, std_T0=std_T0,
                mean_T1=mean_T1, std_T1=std_T1,
                set=stage)
            else:
                self.log_tb_images(
                    trainer,
                    (samples_d['T0'], samples_d['T1'], y, y_hat, samples_d["idx"]),
                    ['T0', 'T1', 'GroundTruth', 'Prediction'],
                    step=model.global_step,
                    mean_T0=mean_T0, std_T0=std_T0,
                    mean_T1=mean_T1, std_T1=std_T1,
                    set=stage
                )
            model.train()

    def log_tb_images(self, trainer, viz_batch, col_titles, step, mean_T0, std_T0, mean_T1, std_T1, set='') -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            # raise ValueError('TensorBoard Logger not found')
            return

        for img_idx, (T0, T1, y_true, y_pred, idx) in enumerate(zip(*viz_batch)):
            # Create one single image with the 4 elements
            if y_true.max().item() <= 1.0:
                y_true = y_true * 255.0
                y_true = y_true.long()

            if y_pred.max().item() <= 1.0:
                y_pred = y_pred * 255.0
                y_pred = y_pred.long()

            T0 = self.denormalize_img_as_tensor(T0, mean_T0, std_T0)
            T1 = self.denormalize_img_as_tensor(T1, mean_T1, std_T1)

            figure = self.image_line([T0, T1, y_true, y_pred], col_titles)
            image = self.plot_to_image(figure)
            tb_logger.add_image(f"img_{set}_{idx}", image, step)     

    def log_tb_images_multitask(self, trainer, viz_batch, col_titles, step, mean_T0, std_T0, mean_T1, std_T1, set='') -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            # raise ValueError('TensorBoard Logger not found')
            return

        for img_idx, (T0, T1, y_true, y_pred, idx) in enumerate(zip(*viz_batch)):
            # Create one single image with the 4 elements
            y_true = convert_to_color(y_true, palette={0:'#000000', 1:'#FFFFFF', self.ignore_index:'#FF00FF'})
            #y_pred = convert_to_color(y_pred, palette={0:'#000000',1:'#FFFFFF'})

            if y_pred.max().item() <= 1.0:
                y_pred = y_pred * 255.0
                y_pred = y_pred.long()

            T0 = self.denormalize_img_as_tensor(T0, mean_T0, std_T0)
            T1 = self.denormalize_img_as_tensor(T1, mean_T1, std_T1)

            figure = self.image_line([T0, T1, y_true, y_pred], col_titles)
            image = self.plot_to_image(figure)
            tb_logger.add_image(f"img_{set}_{idx}", image, step)      

    def log_tb_images_grid(self, trainer, viz_batch, row_titles, col_titles, step, set='') -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            # raise ValueError('TensorBoard Logger not found')
            return

        T0s, T1s, y_true, y_pred, idxs = viz_batch
        
        figure = self.image_lines(
            list(zip(T0s, T1s, y_true, y_pred)),
            row_titles,
            col_titles
        )
        image = self.plot_to_image(figure)

        tb_logger.add_image(f"img_{set}", image, step)


        
        
    def image_lines(self, rows, row_titles, col_titles):
        # Create a figure to contain the plot.
        # titles = ['T0', 'T1', 'GroundTruth', 'Prediction']
        figure = plt.figure(figsize=(2 * len(col_titles), 2 * len(row_titles)))
        gs = figure.add_gridspec(len(row_titles), len(col_titles)) #, hspace=0, wspace=0)
        axes = gs.subplots(sharex='col', sharey='row')
        plt.yticks([])
        plt.grid(False)

        for ax, col in zip(axes[0], col_titles):
            ax.set_title(col)
        for ax, row in zip(axes[:,0], row_titles):
            ax.set_ylabel(row, rotation=0, size='large')
        for i, row in enumerate(rows):
            for j, img in enumerate(row):
                img = img.detach().cpu().permute(1, 2, 0).numpy()
                axes[i, j].imshow(img, interpolation='nearest')
        plt.tight_layout()

        return figure

    def image_line(self, list_images, col_titles):
        # Create a figure to contain the plot.
        figure = plt.figure(figsize=(2 * len(col_titles), 3))
        for i, title in enumerate(col_titles):
            # Start next subplot.
            plt.subplot(1, 4, i + 1, title=title)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            if isinstance(list_images[i], np.ndarray):
                img = list_images[i]
            else:
                img = list_images[i].detach().cpu().permute(1, 2, 0).numpy()
            plt.imshow(img, interpolation='nearest', vmin=0, vmax=255, cmap='gray') # cmap='gray'
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