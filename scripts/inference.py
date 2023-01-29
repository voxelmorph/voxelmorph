import argparse
import os
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import warnings

import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from register_single import register_single
from utils import *

import voxelmorph_group as vxm  # nopep8

warnings.filterwarnings("ignore")


hydralog = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    config_path = f"{conf['model_dir']}/config.yaml"
    if os.path.exists(config_path):
        conf = OmegaConf.load(config_path)
        hydralog.info(f"Loaded config from {config_path}")
    else:
        hydralog.warning(f"The config file {config_path} does not exist. Using the current config")
    
    source_files = os.listdir(conf.moving)
    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA',
           'registered PCA', 'raw T1err', 'registered T1err']
    df = pd.DataFrame(columns=col)

    # load the TI for all subjects
    if conf.TI_json:
        import json
        with open(f"{conf.TI_json}") as json_file:
            TI_dict = json.load(json_file)

    device = 'cuda' if conf.gpu > 0 else 'cpu'

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
    for idx, subject in enumerate(tqdm(source_files, desc="Registering Samples:")):
        if os.path.exists(os.path.join(conf.moved, f"{Path(subject).stem}.nii")):
            hydralog.debug(f"Already registered {Path(subject).stem}")
        else:
            name, loss_org, org_dis, t1err_org, loss_rig, rig_dis, t1err_rig = register_single(
                idx, conf, subject, TI_dict[Path(subject).stem], device, model, logger)
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

    logger.log_dataframe(df, 'Results', path=os.path.join(
        conf.result, 'results.csv'))


if __name__ == '__main__':
    main()