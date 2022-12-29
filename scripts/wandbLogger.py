from logging.config import dictConfig
import wandb
import numpy as np
import torchvision
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from utils import *


class WandbLogger(object):
    def __init__(self, project_name=None, cfg=None, sweep=False) -> None:

        try:
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run
        if not sweep and self._wandb.run is None:
            self._wandb.init(
                project='Voxel Morph',
                name=project_name,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        elif sweep:
            self._wandb.init(allow_val_change=True)


    def log_config(self, args):
        """save the config of this training

        Args:
            config (dict): epochs, batchsize, learning rate
        """
        def namespace_to_dict(namespace):
            return {
                k: namespace_to_dict(v) if isinstance(v, args) else v
                for k, v in vars(namespace).items()
            }
        print(type(vars(args)))
        self._wandb.config.update(vars(args), allow_val_change=True)

    def log_step_metric(self, step, losses, NMI, MSE, NCC, folding_ratio_pos, mag_det_jac_det_pos, l1, l2, d2):
        self._wandb.log({
            "Step": step,
            "Step Loss": losses,
            "Step NMI": NMI,
            "Step MSE": MSE,
            "Step NCC": NCC,
            "Step Folding Ratio pos": folding_ratio_pos,
            "Step Mag Det Jac Det pos": mag_det_jac_det_pos,
            "Step L1": l1,
            "Step L2": l2,
            "Step D2": d2,
        })


    def log_epoch_metric(self, epoch, losses, epoch_loss):
        # morph = y_img_pred[1].transpose()
        self._wandb.log({
            "Epoch": epoch,
            "Epoch Loss": losses,
            "Epoch Similarity": epoch_loss[0],
            "Epoch Regularization": epoch_loss[1]
        })

    def log_morph_field(self, step, pred, fixed, atlas, new_atlas, warp, label):
        # print(f"The shape of the morph field is {input.shape}")
        pred = pred.detach().cpu().numpy()
        fixed = fixed.detach().cpu().numpy()
        warp = warp.detach().cpu().numpy()
        atlas = atlas.detach().cpu().numpy()
        new_atlas = new_atlas.detach().cpu().numpy()
        fig = plot_validation_fig(fixed, pred, new_atlas, atlas, warp)
        self._wandb.log({
            "Step": step,
            label: wandb.Image(fig)
            # label: fig
        })
        plt.close(fig)

    def watchModel(self, model):
        self._wandb.watch(model, 'all')

    def log_register_gifs(self, path, label):
        self._wandb.log({
            label: wandb.Video(path, fps=4, format="gif")
        })

    def log_dataframe(self, df, label):
        self._wandb.log({
            label: wandb.Table(dataframe=df)
        })
    
    def log_img(self, img, label):
        self._wandb.log({
            label: img
        })
