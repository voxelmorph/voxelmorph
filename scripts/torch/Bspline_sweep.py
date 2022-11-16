import wandb
import argparse
import tempfile
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from wandbLogger import WandbLogger

from train import train
from register_single import register_single
from utils import *
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/MOLLI_nmi_bspline.yaml', help='config file')
    args = parser.parse_args()

    # load the config file
    cfg = OmegaConf.load(args.config)
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))

    if conf.wandb:
        wandb_logger = WandbLogger(sweep=True)
    # name = f"{conf\["dataset"\]}_{conf["image_loss"]}_{conf["weight"]}_{conf["norm"]}"
    
    conf.model_dir = f"{conf['model_dir']}/weight_{wandb_logger._wandb.config['weight']}"
    conf.inference = f"{conf['inference']}/weight_{wandb_logger._wandb.config['weight']}"
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


    print(f"{'---'*10} Start Testing {'---'*10}")
    conf.model_path = os.path.join(conf.model_dir, '%04d.pt' % conf.epochs)
    conf.moved = os.path.join(conf.inference, 'moved')
    conf.warp = os.path.join(conf.inference, 'warp')
    conf.result = os.path.join(conf.inference, 'summary')

    os.makedirs(conf.moved, exist_ok=True)
    os.makedirs(conf.warp, exist_ok=True)
    os.makedirs(conf.result, exist_ok=True)

    source_files = os.listdir(conf.moving)
    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA', 'registered PCA']
    df = pd.DataFrame(columns=col)
    for subject in tqdm(source_files):
        name, loss_org, org_pca, loss_rig, rig_pca = register_single(conf, subject, wandb_logger)
        df = pd.concat([df, pd.DataFrame([[name, loss_org, loss_rig, org_pca, rig_pca]], columns=col)], ignore_index=True)
        # df.append(name, loss_org, org_dis, loss_rig, rig_dis)
    # convert the registered images to gif and compute the results

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df.to_csv(os.path.join(conf.result, 'results.csv'), index=False)

    wandb_logger.log_dataframe(df, 'Results')
    print(f"{'---'*10} End of Testing {'---'*10}")

if __name__ == '__main__':

    sweep_configuration = {
        'method': 'grid',
        'name': 'Bspline Config Sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'Epoch NMI'
            },
        'parameters': {
            'cps': {'values': [1, 2, 4, 8, 16, 32, 64]},
            'svf_steps': {'values': [1, 4]},
            'svf_scale': {'values': [1, 4]},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Voxel Morph')

    wandb.agent(sweep_id, function=main, count=70)