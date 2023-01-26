import argparse
import os
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import warnings

from register_single import register_single
from utils import *
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    wandb_logger = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='model/fbMOLLI_post_nmi_l2/', help='root path')
    parser.add_argument('--weight', 
                        default=0.001, help='weight')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()

    # load the config file
    conf_path = f"{args.root}/config.yaml"
    cfg = OmegaConf.load(conf_path)
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))

    print(f"{'---'*10} Start Testing {'---'*10}")
    conf.model_path = os.path.join(conf.model_dir_round, '%04d.pt' % conf.epochs)
    conf.moved = os.path.join(conf.inference, 'moved')
    conf.warp = os.path.join(conf.inference, 'warp')
    conf.result = os.path.join(conf.inference, 'summary')
    print(f"Mona debug - conf: {conf}")
    
    os.makedirs(conf.moved, exist_ok=True)
    os.makedirs(conf.warp, exist_ok=True)
    os.makedirs(conf.result, exist_ok=True)

    source_files = os.listdir(conf.moving)
    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA', 'registered PCA']
    df = pd.DataFrame(columns=col)
    for subject in tqdm(source_files):
        name, loss_org, org_dis, loss_rig, rig_dis = register_single(
            conf, subject, wandb_logger)
        df = pd.concat([df, pd.DataFrame(
            [[name, loss_org, org_dis, loss_rig, rig_dis]], columns=col)], ignore_index=True)

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df.to_csv(os.path.join(conf.result, 'results.csv'), index=False)
    if wandb_logger:
        wandb_logger.log_dataframe(df, 'Results')
    print(f"{'---'*10} End of Testing {'---'*10}")