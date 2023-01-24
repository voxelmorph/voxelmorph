import glob
import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import scipy.io
from omegaconf import DictConfig, OmegaConf
from register_single import register_single
from tqdm import tqdm
from train import train
from utils import *
from wandbLogger import WandbLogger

hydralog = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    hydralog.debug(f"Conf: {conf} and type: {type(conf)}")

    conf.model_path = os.path.join(conf.model_dir, '%04d.pt' % conf.epochs)

    if conf.log == 'wandb':
        logger = WandbLogger(project_name=conf.wandb_project, cfg=conf)
    elif conf.log == 'neptune':
        from NeptuneLogger import NeptuneLogger
        logger = NeptuneLogger(project_name=conf.wandb_project, cfg=conf)

    # save the config
    config_path = f"{conf['model_dir']}/config.yaml"
    try:
        with open(config_path, 'w') as fp:
            OmegaConf.save(config=conf, f=fp.name)
    except:
        hydralog.warning("Unable to copy the config")

    pipeline(conf, logger)


    logger._wandb.finish()
    assert logger._wandb.run is None

def createdir(conf):
    conf.moved = os.path.join(conf.inference, f"round{conf.round}", 'moved')
    conf.warp = os.path.join(conf.inference, f"round{conf.round}", 'warp')
    conf.result = os.path.join(conf.inference, f"round{conf.round}", 'summary')
    conf.val = os.path.join(conf.inference, f"round{conf.round}", 'val')

    os.makedirs(conf.moved, exist_ok=True)
    os.makedirs(conf.warp, exist_ok=True)
    os.makedirs(conf.result, exist_ok=True)
    os.makedirs(conf.val, exist_ok=True)


def generate_input(conf):
    if not conf.final:
        files = glob.glob(os.path.join(conf.moved, '*.npy'))
        txt_string = "\n".join(files)
        with open(f"{conf.moved}/MOLLI_post_input.txt", "w") as f:
            f.write(txt_string) 
        conf.img_list = f"{conf.moved}/MOLLI_post_input.txt"
    return conf

def pipeline(conf, logger=None):
    # first train the model with rpca rank=5?
    conf.final = False
    hydralog.info(f"{'---'*10} Round 1 {'---'*10}")
    conf.rank = 5
    conf.round = 1
    createdir(conf)
    train(conf, logger)
    validate(conf, logger)
    conf = generate_input(conf)
    hydralog.info(f"{'---'*10} Round 1 {'---'*10}")

    hydralog.info(f"{'---'*10} Round 2 {'---'*10}")
    conf.rank = 5
    conf.round = 2
    createdir(conf)
    
    train(conf, logger)
    validate(conf, logger)
    conf = generate_input(conf)
    hydralog.info(f"{'---'*10} Round 2 {'---'*10}")

    hydralog.info(f"{'---'*10} Round 3 {'---'*10}")
    conf.final = True
    conf.rank = 5
    conf.round = 2
    createdir(conf)
    
    train(conf, logger)
    validate(conf, logger)
    hydralog.info(f"{'---'*10} Round 2 {'---'*10}")

def validate(conf, logger=None):

    source_files = os.listdir(conf.moving)
    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA', 'registered PCA', 'raw T1err', 'registered T1err']
    df = pd.DataFrame(columns=col)

        # load the TI for all subjects
    if conf.TI_json:
        import json
        with open(f"{conf.TI_json}") as json_file:
            TI_dict = json.load(json_file)
        hydralog.debug(f"Loading TI from json {TI_dict}")

    for subject in source_files:
        name, loss_org, org_dis, t1err_org, loss_rig, rig_dis, t1err_rig = register_single(
            conf, subject, TI_dict[Path(subject).stem], logger)
        df = pd.concat([df, pd.DataFrame(
            [[name, loss_org, loss_rig, org_dis, rig_dis, t1err_org, t1err_rig]], columns=col)], ignore_index=True)
    # convert the registered images to gif and compute the results

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df['T1err changes percentage'] = percentage_change(
        df['raw T1err'], df['registered T1err'])
    df.to_csv(os.path.join(conf.result, 'results.csv'), index=False)
    hydralog.info(f"The summary is \n {df.describe()}")

    logger.log_dataframe(df, 'Results', path=os.path.join(
        conf.result, 'results.csv'))


def save2mat(conf):

    def nii2mat(nii_path, mat_path):
        img = sitk.ReadImage(nii_path)
        img_array = sitk.GetArrayFromImage(img)
        scipy.io.savemat(mat_path, {'img': img_array})

    folder = conf.inference
    output_folder = os.path.join(folder, 'moved_mat')
    os.makedirs(output_folder, exist_ok=True)
    registed_subjects = glob.glob(os.path.join(conf.moved, '*.nii'))
    for file in registed_subjects:
        name = Path(file).stem + '.mat'
        hydralog.debug(f"The file name is {name}")
        nii2mat(file, os.path.join(output_folder, name))


if __name__ == '__main__':
    main()
