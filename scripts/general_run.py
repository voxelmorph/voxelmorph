import os
import hydra
import logging
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from wandbLogger import WandbLogger

from train import train
from register_single import register_single
from utils import *

hydralog = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg:DictConfig):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str,
    #                     default='conf/config.yaml', help='config file')
    # args = parser.parse_args()

    # # load the config file
    # cfg = OmegaConf.load(args.config)
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    hydralog.debug(f"Conf: {conf} and type: {type(conf)}")

    conf.model_path = os.path.join(conf.model_dir, '%04d.pt' % conf.epochs)
    conf.moved = os.path.join(conf.inference, 'moved')
    conf.warp = os.path.join(conf.inference, 'warp')
    conf.result = os.path.join(conf.inference, 'summary')
    conf.val = os.path.join(conf.inference, 'val')
    
    os.makedirs(conf.moved, exist_ok=True)
    os.makedirs(conf.warp, exist_ok=True)
    os.makedirs(conf.result, exist_ok=True)
    os.makedirs(conf.val, exist_ok=True)

    if conf.log == 'wandb':
        logger = WandbLogger(project_name=conf.wandb_project, cfg=conf)
    elif conf.log == 'neptune':
        from NeptuneLogger import NeptuneLogger
        logger = NeptuneLogger(project_name=conf.wandb_project, cfg=conf)

    # run the training
    hydralog.info(f"{'---'*10} Start Training {'---'*10}")
    train(conf, logger)
    config_path = f"{conf['model_dir']}/config.yaml"
    try:
        with open(config_path, 'w') as fp:
            OmegaConf.save(config=conf, f=fp.name)
    except:
        hydralog.warning("Unable to copy the config")
    hydralog.info(f"{'---'*10} End of Training {'---'*10}")

    # register the model
    hydralog.info(f"{'---'*10} Start Testing {'---'*10}")

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
    
    logger.log_dataframe(df, 'Results', path=os.path.join(conf.result, 'results.csv'))
    hydralog.info(f"{'---'*10} End of Testing {'---'*10}")


if __name__ == '__main__':
    main()