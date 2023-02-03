import glob
import logging
import multiprocessing
import os
import time
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

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph_group as vxm  # nopep8

hydralog = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    hydralog.debug(f"Conf: {conf} and type: {type(conf)}")
    # conf.model_path = os.path.join(conf.model_dir, '%04d.pt' % conf.epochs)
    conf.model_path = '/Users/mona/Documents/repo/voxelmorph-test/model/MOLLI_post/group/rank_5_5_5/jointcorrelation/l2/image_loss_weight1/weight0.3/bspline/cps4_svfsteps7_svfscale1/e100/round1/0100.pt'
    
    logger = None

    # save the config
    config_path = f"{conf['model_dir']}/config.yaml"
    os.makedirs(conf['model_dir'], exist_ok=True)
    try:
        with open(config_path, 'w') as fp:
            OmegaConf.save(config=conf, f=fp.name)
    except:
        hydralog.warning("Unable to copy the config")

    # createdir(conf)
    validate(conf, logger)
    hydralog.info("Done")

    logger._wandb.finish()
    assert logger._wandb.run is None


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
    os.makedirs(conf.val, exist_ok=True)


def validate(conf, logger=None):
    # if os.path.exists(os.path.join(conf.result, 'results.csv')):
        # return

    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA',
           'registered PCA', 'raw T1err', 'registered T1err']
    df = pd.DataFrame(columns=col)

    # load the TI for all subjects
    if conf.TI_json:
        import json
        with open(f"{conf.TI_json}") as json_file:
            TI_dict = json.load(json_file)

    device = 'cpu'
    num_cores = multiprocessing.cpu_count()
    conf.num_cores = num_cores if num_cores < 64 else 64
    hydralog.info(f"Existing {num_cores}, Using {conf.num_cores} cores")

    # conf.model_path = os.path.join(
    #     conf.model_dir_round, '%04d.pt' % conf.epochs)
    if conf.transformation == 'Dense':
        model = vxm.networks.VxmDense.load(conf.model_path, device)
    elif conf.transformation == 'bspline':
        model = vxm.networks.GroupVxmDenseBspline.load(conf.model_path, device)
    else:
        raise ValueError('transformation must be dense or bspline')

    model.to(device)
    model.eval()

    hydralog.info("Registering Samples:")
    
    idx = 0
    if os.path.exists(os.path.join(conf.moved, f"{Path(conf.subject).stem}.nii")):
        hydralog.debug(f"Already registered {Path(conf.subject).stem}")
    else:
        name, loss_org, org_dis, t1err_org, loss_rig, rig_dis, t1err_rig = register_single(
            idx, conf, conf.subject, TI_dict[Path(conf.subject).stem], device, model, logger)
        if t1err_org is not None:
            df = pd.concat([df, pd.DataFrame(
                [[name, loss_org, loss_rig, org_dis, rig_dis, t1err_org, t1err_rig]], columns=col)], ignore_index=True)

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df['T1err changes percentage'] = percentage_change(
        df['raw T1err'], df['registered T1err'])
    df.to_csv(os.path.join(conf.result, 'results.csv'), index=False)
    hydralog.info(f"The summary is \n {df.describe()}")

    logger.log_dataframe(df, f"{conf.round}_summary", path=os.path.join(
        conf.result, 'results.csv'))
    return

if __name__ == '__main__':
    main()