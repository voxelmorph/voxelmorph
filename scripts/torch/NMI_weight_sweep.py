import wandb
import argparse
import tempfile
from omegaconf import OmegaConf
from wandbLogger import WandbLogger

from train import train
from utils import *
import shutil

sweep_configuration = {
    'method': 'grid',
    'name': 'NMI weight Sweep l1',
    'metric': {
        'goal': 'minimize', 
        'name': 'Epoch Loss'
		},
    'parameters': {
        'weight': {'values': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
     }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/MOLLI_nmi.yaml', help='config file')
    args = parser.parse_args()

    # load the config file
    cfg = OmegaConf.load(args.config)
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))

    if conf.wandb:
        wandb_logger = WandbLogger(sweep=True)
    # name = f"{conf\["dataset"\]}_{conf["image_loss"]}_{conf["weight"]}_{conf["norm"]}"
    
    conf.model_dir = f"{conf['model_dir']}/weight_{wandb_logger._wandb.config['weight']}"
    conf.inference = f"{conf['model_dir']}"
    conf.weight = wandb_logger._wandb.config['weight']
    wandb_logger._wandb.config.update(conf)
    print(f"Mona debug - conf: {conf} and type: {type(conf)}")

    # run the training
    print(f"{'---'*10} Start Training {'---'*10}")
    train(conf, wandb_logger)
    print(f"{'---'*10} End of Training {'---'*10}")
    config_path = f"{conf['model_dir']}/config.yaml"
    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=conf, f=fp.name)
        shutil.copy(fp.name, config_path)

if __name__ == '__main__':
    

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Voxel Morph')

    wandb.agent(sweep_id, function=main, count=15)