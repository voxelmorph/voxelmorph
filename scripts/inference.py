import glob
import logging
import multiprocessing
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

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph_group as vxm  # nopep8

hydralog = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    rounds = [1, 2, 3]
    for round in rounds:
        conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
        # conf.model_path = os.path.join(conf.model_dir, '%04d.pt' % conf.epochs)

        logger = None

        if conf.TI_csv:
            TI_dict = csv_to_dict(conf.TI_csv)

        conf.round = round
        conf.rank = conf.rpca_rank[f"rank{conf.round}"]
        conf.model_dir_round = os.path.join(conf.model_dir, f"round{conf.round}")
        conf.inference = f"{conf.inference}/test_{conf.dataset}"
        conf.final = True

        createdir(conf)
        if conf.round > 1:
            conf.moving = os.path.join(conf.inference, f"round{conf.round-1}", 'moved')
        hydralog.debug(f"Round {round} - Conf: {conf}")
        validate(conf, TI_dict, logger)
        
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


def validate(conf, TI_dict, logger):
    if os.path.exists(os.path.join(conf.result, f"{conf.round}_summary.csv")):
        return
    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA',
           'registered PCA', 'raw T1err', 'registered T1err']
    df = pd.DataFrame(columns=col)

    device = 'cuda' if conf.gpu > 0 else 'cpu'

    num_cores = multiprocessing.cpu_count()
    conf.num_cores = num_cores if num_cores < 64 else 64
    hydralog.info(f"Existing {num_cores}, Using {conf.num_cores} cores")

    conf.model_path = os.path.join(
        conf.model_dir_round, '%04d.pt' % conf.epochs)
    if conf.transformation == 'Dense':
        model = vxm.networks.VxmDense.load(conf.model_path, device)
    elif conf.transformation == 'bspline':
        model = vxm.networks.GroupVxmDenseBspline.load(conf.model_path, device)
    else:
        raise ValueError('transformation must be dense or bspline')

    model.to(device)
    model.eval()

    hydralog.info("Registering Samples:")
    
    source_files = glob.glob(os.path.join(conf.moving, "*.npy"))
    for idx, subject in enumerate(tqdm(source_files, desc="Registering Samples:")):
        name = Path(subject).stem
        if os.path.exists(os.path.join(conf.moved, f"{name}.nii")):
            hydralog.debug(f"Already registered {name}")
        else:
            tvec = np.array(list(TI_dict[name].values())[1:], dtype=np.float32)
            name, loss_org, org_dis, t1err_org, loss_rig, rig_dis, t1err_rig = register_single(
                idx, conf, subject, tvec, device, model, logger)
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