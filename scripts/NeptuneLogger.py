from logging.config import dictConfig
import neptune.new as neptune
from neptune.new.types import File
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from utils import *


class NeptuneLogger(object):
    def __init__(self, project_name=None, cfg=None, sweep=False) -> None:

        try:
            # Initialize a W&B run
            self.run = neptune.init_run(
                project="lixinqi98/Registration",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNzg1YzBmZi1jYzk3LTQxMTEtOTU1Mi0yYjk2NmMxNGI5NTYifQ==",
                name=project_name,
            )  # your credentials
            self.log_config(cfg)
        except ImportError:
            raise ImportError(
                "To use the Neptune Logger please install wandb."
                "Run `pip install neptune-client` to install it."
            )

    def log_config(self, args):
        """save the config of this training

        Args:
            config (dict): epochs, batchsize, learning rate
        """
        
        print(type(vars(args)))
        self.run["params"] = args


    def log_epoch_metric(self, epoch, losses, epoch_loss):

        self.run["train/epoch/Loss"].append(losses)
        self.run["train/epoch/Similarity"].append(epoch_loss[0])
        self.run["train/epoch/Regularization"].append(epoch_loss[1])

    def log_morph_field(self, step, pred, fixed, atlas, new_atlas, warp, label):
        # print(f"The shape of the morph field is {input.shape}")
        pred = pred.detach().cpu().numpy()
        fixed = fixed.detach().cpu().numpy()
        warp = warp.detach().cpu().numpy()
        atlas = atlas.detach().cpu().numpy()
        new_atlas = new_atlas.detach().cpu().numpy()
        fig = plot_validation_fig(fixed, pred, new_atlas, atlas, warp)
        
        self.run["train/epoch/validation_img"].append(fig, step=step)
        plt.close(fig)


    def log_gifs(self, path, label):
        
        self.run[label].append(File(path))

    def log_dataframe(self, df, label):
        self.run[label].upload(File.as_html(df))
    
    def log_img(self, img, label):
        self.run[label].append(img)

    def log_metric(self, epoch, label, value):
        self.run[label].append(value)