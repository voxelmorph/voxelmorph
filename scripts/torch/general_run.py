import argparse
import os
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from wandbLogger import WandbLogger

from train import train
from register_single import register_single
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/ncc_b1.yaml', help='config file')
    args = parser.parse_args()

    # load the config file
    cfg = OmegaConf.load(args.config)
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    print(f"Mona debug - conf: {conf} and type: {type(conf)}")

    if conf.wandb:
        wandb_logger = WandbLogger(project_name=conf.wandb_project, cfg=conf)

    # run the training
    print(f"{'---'*10} Start Training {'---'*10}")
    train(conf, wandb_logger)
    print(f"{'---'*10} End of Training {'---'*10}")

    # register the model
    print(f"{'---'*10} Start Testing {'---'*10}")
    conf.model_path = os.path.join(conf.model_dir, '%04d.pt' % conf.epochs)

    os.makedirs(conf.moved, exist_ok=True)
    os.makedirs(conf.warp, exist_ok=True)
    os.makedirs(conf.result, exist_ok=True)

    source_files = os.listdir(conf.moving)
    col =  ['Cases', 'raw MSE', 'registered MSE', 'raw PCA', 'registered PCA']
    df = pd.DataFrame(columns=col)
    for subject in tqdm(source_files):
        name, loss_org, org_dis, loss_rig, rig_dis = register_single(
            conf, subject, wandb_logger)
        df = pd.concat([df, pd.DataFrame([[name, loss_org, org_dis, loss_rig, rig_dis]], columns=col)], ignore_index=True)
        # df.append(name, loss_org, org_dis, loss_rig, rig_dis)
    # convert the registered images to gif and compute the results

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df.to_csv(os.path.join(conf.result, 'results.csv'), index=False)

    wandb_logger.log_dataframe(df, 'Results')
    print(f"{'---'*10} End of Testing {'---'*10}")
