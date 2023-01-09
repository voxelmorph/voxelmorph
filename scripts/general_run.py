import argparse
import os
import shutil
import tempfile
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from NeptuneLogger import NeptuneLogger

from train import train
from register_single import register_single
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/MOLLI_jointcorrelation_group.yaml', help='config file')
    args = parser.parse_args()

    # load the config file
    cfg = OmegaConf.load(args.config)
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    print(f"Mona debug - conf: {conf} and type: {type(conf)}")

    if conf.wandb:
        logger = NeptuneLogger(project_name=conf.wandb_project, cfg=conf)

    # run the training
    print(f"{'---'*10} Start Training {'---'*10}")
    train(conf, logger)
    config_path = f"{conf['model_dir']}/config.yaml"
    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=conf, f=fp.name)
        shutil.copy(fp.name, config_path)
    print(f"{'---'*10} End of Training {'---'*10}")

    # register the model
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
        name, loss_org, org_dis, loss_rig, rig_dis = register_single(conf, subject, logger)
        df = pd.concat([df, pd.DataFrame([[name, loss_org, loss_rig, org_dis, rig_dis]], columns=col)], ignore_index=True)
        # df.append(name, loss_org, org_dis, loss_rig, rig_dis)
    # convert the registered images to gif and compute the results

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df.to_csv(os.path.join(conf.result, 'results.csv'), index=False)

    logger.log_dataframe(df, 'Results')
    print(f"{'---'*10} End of Testing {'---'*10}")
