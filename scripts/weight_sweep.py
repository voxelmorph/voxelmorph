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
                        default='configs/MOLLI_ngf_group.yaml', help='config file')
    args = parser.parse_args()

    # load the config file
    cfg = OmegaConf.load(args.config)

    if cfg.log == 'wandb':
        logger = WandbLogger(sweep=True)

    cfg.weight = logger._wandb.config['weight']
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    
    
    conf.model_path = os.path.join(conf.model_dir, '%04d.pt' % conf.epochs)
    conf.moved = os.path.join(conf.inference, 'moved')
    conf.warp = os.path.join(conf.inference, 'warp')
    conf.result = os.path.join(conf.inference, 'summary')
    conf.val = os.path.join(conf.inference, 'val')
    
    os.makedirs(conf.moved, exist_ok=True)
    os.makedirs(conf.warp, exist_ok=True)
    os.makedirs(conf.result, exist_ok=True)
    os.makedirs(conf.val, exist_ok=True)

    logger._wandb.config.update(conf)
    print(f"Mona debug - conf: {conf} and type: {type(conf)}")

    # run the training
    print(f"{'---'*10} Start Training {'---'*10}")
    train(conf, logger)
    print(f"{'---'*10} End of Training {'---'*10}")
    config_path = f"{conf['model_dir']}/config.yaml"
    try:
        with tempfile.NamedTemporaryFile() as fp:
            OmegaConf.save(config=conf, f=fp.name)
            shutil.copy(fp.name, config_path)
    except:
        print("Unable to copy the config")


    print(f"{'---'*10} Start Testing {'---'*10}")

    source_files = os.listdir(conf.moving)
    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA', 'registered PCA']
    df = pd.DataFrame(columns=col)
    for subject in tqdm(source_files):
        name, loss_org, org_pca, loss_rig, rig_pca = register_single(conf, subject, logger)
        df = pd.concat([df, pd.DataFrame([[name, loss_org, loss_rig, org_pca, rig_pca]], columns=col)], ignore_index=True)
        # df.append(name, loss_org, org_dis, loss_rig, rig_dis)
    # convert the registered images to gif and compute the results

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df.to_csv(os.path.join(conf.result, 'results.csv'), index=False)

    logger.log_dataframe(df, 'Results')
    print(f"{'---'*10} End of Testing {'---'*10}")

if __name__ == '__main__':

    sweep_configuration = {
        'method': 'grid',
        'name': 'NGF finetune weight sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'Epoch Loss'
            },
        'parameters': {
            'weight': {'values': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Group Registration')

    wandb.agent(sweep_id, function=main, count=10)