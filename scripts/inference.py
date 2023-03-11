import glob
import logging
import multiprocessing
import os
from pathlib import Path

import hydra
import pandas as pd
import scipy.io
import torch
from omegaconf import DictConfig, OmegaConf
from register_single import register_single
from tqdm import tqdm
from train import train
from utils import *
from wandbLogger import WandbLogger

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph_group as vxm  # nopep8

hydralog = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    rounds = conf.rpca_rank.n_ranks

    for round in range(rounds):
        conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
        logger = None

        conf.round = round + 1
        conf.rank = conf.rpca_rank[f"rank{round+1}"]
        if conf.rank == 0:
            hydralog.info("No rpca, Finish")
            return
        conf.model_dir_round = os.path.join(conf.model_dir, f"round{conf.round}")
        # conf.inference = f"{conf.inference}/test_{conf.dataset}_{conf.inference_epochs}"
        conf.inference = f"{conf.inference}/test_{conf.dataset}"
        conf.final = True

        createdir(conf)
        if conf.round > 1:
            conf.moving = os.path.join(conf.inference, f"round{conf.round-1}", 'moved')
        hydralog.debug(f"Round {round} - Conf: {conf}")
        validate(conf, logger)
        
def createdir(conf):
    conf.moved = os.path.join(conf.inference, f"round{conf.round}", 'moved')
    conf.warp = os.path.join(conf.inference, f"round{conf.round}", 'warp')
    conf.result = os.path.join(conf.inference, f"round{conf.round}", 'summary')
    conf.val = os.path.join(conf.inference, f"round{conf.round}", 'val')

    conf.model_dir_round = os.path.join(conf.model_dir, f"round{conf.round}")

    os.makedirs(conf.moved, exist_ok=True)
    os.makedirs(conf.warp, exist_ok=True)
    os.makedirs(conf.result, exist_ok=True)
    os.makedirs(conf.val, exist_ok=True)
    os.makedirs(conf.model_dir_round, exist_ok=True)


def validate(conf, logger):
    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA',
           'registered PCA', 'raw T1err', 'registered T1err']
    df = pd.DataFrame(columns=col)

    device = 'cuda' if conf.gpu > 0 else 'cpu'

    num_cores = multiprocessing.cpu_count()
    conf.num_cores = num_cores if num_cores < 64 else 64
    hydralog.info(f"Existing {num_cores}, Using {conf.num_cores} cores")

    conf.model_path = os.path.join(
        conf.model_dir_round, '%04d.pt' % conf.inference_epochs)
    checkpoint = torch.load(conf.model_path, map_location=torch.device(device))
    model_conf = checkpoint['config']
    hydralog.debug(f"Load the model from {conf.model_path}, model config: {model_conf}")

    if conf.transformation == 'Dense':
        model = vxm.networks.VxmDense(
            inshape=model_conf['inshape'],
            nb_unet_features=model_conf['nb_unet_features'],
            bidir=model_conf['bidir'],
            int_steps=model_conf['int_steps'],
            int_downsize=model_conf['int_downsize']
        )
    elif conf.transformation == 'bspline':
        if conf.register == 'Group':
            model = vxm.networks.GroupVxmDenseBspline(
                inshape=model_conf['inshape'],
                nb_unet_features=model_conf['nb_unet_features'],
                bidir=model_conf['bidir'],
                int_steps=model_conf['int_steps'],
                int_downsize=model_conf['int_downsize'],
                src_feats=model_conf['src_feats'],
                trg_feats=model_conf['trg_feats'],
                cps=model_conf['cps'],
                svf=model_conf['svf'],
                svf_steps=model_conf['svf_steps'],
                svf_scale=model_conf['svf_scale'],
                resize_channels=model_conf['resize_channels'],
                method=model_conf['method']
            )
            hydralog.debug("Mona: use the Group bspline model")
    else:
        hydralog.error("Mona: the register type is not supported")
        raise NotImplementedError   
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        hydralog.info("Load the model from the new version")
    except:
        model = vxm.networks.GroupVxmDenseBspline.load(conf.model_path, device)
        hydralog.info("Load the model from the old version")
    model.eval()

    hydralog.info("Registering Samples:")
    
    source_files = glob.glob(os.path.join(conf.moving, "*.npy"))
    for idx, subject in enumerate(tqdm(source_files, desc="Registering Samples:")):
        name = Path(subject).stem
        if os.path.exists(os.path.join(conf.moved, f"{name}.nii")):
            hydralog.debug(f"Already registered {name}")
        else:
            name, loss_org, org_dis, t1err_org, loss_rig, rig_dis, t1err_rig = register_single(
                idx, conf, subject, device, model, logger)
            df = pd.concat([df, pd.DataFrame(
                [[name, loss_org, loss_rig, org_dis, rig_dis, t1err_org, t1err_rig]], columns=col)], ignore_index=True)

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df['T1err changes percentage'] = percentage_change(
        df['raw T1err'], df['registered T1err'])
    df.to_csv(os.path.join(conf.result, f"{conf.round}_summary.csv"), index=False)
    hydralog.info(
        f"The summary is \n {df[['MSE changes percentage', 'PCA changes percentage', 'T1err changes percentage']].describe()}")

    return

if __name__ == '__main__':
    main()