from logging.config import dictConfig
import wandb
import numpy as np
import torchvision
from omegaconf import OmegaConf
import matplotlib.pyplot as plt


class WandbLogger(object):
    def __init__(self, project_name: str, cfg) -> None:

        try:
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(
                project='Voxel Morph',
                name=project_name,
                config=OmegaConf.to_container(cfg, resolve=True)
            )

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
        self._wandb.config.update(vars(args))

    def log_step_metric(self, step, losses, loss_1, loss_2, MI, x_img, y_img_pred, y_img_true):
        # morph = y_img_pred[1].transpose()
        # print(f"The shape of the morph field is {y_img_pred[1].shape} and the shape of the image is {x_img[0].shape}")
        self.log_morph_field(np.squeeze(
            y_img_pred[1][0, :, :, :].detach().cpu().numpy()), step)
        self._wandb.log({
            "Step Loss": losses,
            "Step Loss_1": loss_1,
            "Step Loss_2": loss_2,
            "input images 1": wandb.Image(np.squeeze(x_img[0][0, :, :, :])),
            "output images 1": wandb.Image(np.squeeze(y_img_pred[0][0, :, :, :])),
            "ground truth 1": wandb.Image(np.squeeze(y_img_true[0][0, :, :, :])),
            "record MI": MI
        })

    def log_epoch_metric(self, epoch, losses, mu, l2):
        # morph = y_img_pred[1].transpose()
        self._wandb.log({
            "Epoch Loss": losses,
            "Epoch Loss_1": mu,
            "Epoch Loss_2": l2,
        })

    def log_morph_field(self, input, step):
        # print(f"The shape of the morph field is {input.shape}")
        _, rows, cols = input.shape
        x, y = np.meshgrid(np.arange(0, rows, 1), np.arange(0, cols, 1))
        u, v = input[0, :, :], input[1, :, :]
        # print(f"The shape of the u and v are {u.shape} and {v.shape}, the shape of x and y are {x.shape} and {y.shape}")
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.quiver(x, y, u, v, units='width')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')

        self._wandb.log({
            "Motion Field": wandb.Image(fig),
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
