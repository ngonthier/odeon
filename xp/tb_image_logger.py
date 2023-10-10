import io
from typing import Dict, List, Union, TypeVar, cast
import PIL.Image
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from odeon.data.data_module import Input

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
    

class TensorboardGenerativeModelImageSampler(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation

    Example::

        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler(input)])
    """
    
    def __init__(
        self,
        input: Input, # type: ignore
    ) -> None:
        """
        Args:
            input: Input Data Module.
        """

        super().__init__()
        self.input = input

    def on_train_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        self.log_image("train", self.input.fit, trainer, model)

    def on_validation_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        self.log_image("val", self.input.validate, trainer, model)

    def on_test_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        self.log_image("test", self.input.test, trainer, model)

    def log_image(self, stage: str, data: Input, trainer: Trainer, model: LightningModule) -> None:
        df = data.dataframe
        samples = df[df["sampled"] == True]
        samples_t = []
        for i in samples.index.values:
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
            y_hat = torch.sigmoid(y_hat) # WARNING: how to know activation in general?
            self.log_tb_images(
                trainer,
                (samples_d['T0'], samples_d['T1'], y, y_hat, samples_d["idx"]),
                ['T0', 'T1', 'GroundTruth', 'Prediction'],
                step=model.global_step,
                set=stage
            )
            model.train()

    def log_tb_images(self, trainer, viz_batch, col_titles, step, set='') -> None:
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
            figure = self.image_line([T0, T1, y_true * 255.0, y_pred * 255.0], col_titles)
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
        """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
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
            img = list_images[i].detach().cpu().permute(1, 2, 0).numpy()
            plt.imshow(img, interpolation='nearest', vmin=0, vmax=255)
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