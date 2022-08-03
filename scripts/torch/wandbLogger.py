import wandb
import numpy as np
import torchvision
class WandbLogger(object):
    def __init__(self, project_name) -> None:

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
                name=project_name
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
    

    def log_epoch_metric(self, step, losses, x_img, y_img_pred, y_img_true):

        self._wandb.log({
            "step": step,
            "losses": losses,
            "input images 1": wandb.Image(np.squeeze(x_img[0])),
            "output images 1": wandb.Image(np.squeeze(y_img_pred[0])),
            "ground truth 1": wandb.Image(np.squeeze(y_img_true[0])),
            "input images 2": wandb.Image(np.squeeze(x_img[1])),
            "output images 2": wandb.Image(np.squeeze(y_img_pred[1])),
            "ground truth 2": wandb.Image(np.squeeze(y_img_true[1])),
        })
    
    def watch_model(self, model):
        self._wandb.watch(model)

