from pathlib import Path
import time
import os
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import warnings

import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from register_single import *
from utils import *

import voxelmorph_group as vxm  # nopep8
from wandbLogger import WandbLogger

warnings.filterwarnings("ignore")


hydralog = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))

    conf.round = 3
    conf.moved = os.path.join(conf.inference, f"round{conf.round}", 'moved')
    conf.result = os.path.join(conf.inference, "final_summary")
    os.makedirs(conf.result, exist_ok=True)

    hydralog.debug(f"Conf: {conf} and type: {type(conf)}")

    if conf.log == 'wandb' and conf.run_id is not None:
        logger = WandbLogger(project_name=conf.wandb_project, cfg=conf)

    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA',
           'registered PCA', 'raw T1err', 'registered T1err']
    df = pd.DataFrame(columns=col)

    # load the TI for all subjects
    if conf.TI_json:
        import json
        with open(f"{conf.TI_json}") as json_file:
            TI_dict = json.load(json_file)

    device = 'cpu'

    train_files = vxm.py.utils.read_file_list(conf.img_list, prefix=conf.img_prefix,
                                              suffix=conf.img_suffix)
    hydralog.info(f"Number of samples: {len(train_files)}")
    t1map = vxm.groupwise.t1map.MOLLIT1mapParallel()
            
    for idx, subject in enumerate(tqdm(train_files, desc="Registering Samples:")):
        if idx % 10 != 0:
            continue
        name = Path(subject).stem
        start = time.time()
        tvec = TI_dict[Path(subject).stem]
        orig_vols= vxm.py.utils.load_volfile(subject, add_feat_axis=False, ret_affine=False)
        rigs_vols = vxm.py.utils.load_volfile(os.path.join(
            conf.moved, f"{name}.npy"), add_feat_axis=False, ret_affine=False)
        
        orig_T1err = map(t1map, orig_vols.transpose(1, 2, 0), tvec=tvec, conf=conf, name=name, label='original')
        rigs_T1err = map(t1map, rigs_vols.transpose(1, 2, 0), tvec=tvec, conf=conf, name=name, label='register')
        et = time.time()

        mean_orig_T1err = np.mean(orig_T1err)
        mean_rigs_T1err = np.mean(rigs_T1err)
        saveT1err(orig_T1err, rigs_T1err, conf, name, None)
        hydralog.info(
            f"{name}, Time elapsed: {(et - start)/60} mins, T1 error orig {mean_orig_T1err} and rigs {mean_rigs_T1err}")

        orig = orig_vols.transpose(1, 2, 0)
        moved = rigs_vols.transpose(1, 2, 0)

        org_mse, rig_mse = 0, 0
        for j in range(1, moved.shape[-1]):
            rig_mse += mse(moved[:, :, j-1], moved[:, :, j])
            org_mse += mse(orig[:, :, j-1], orig[:, :, j])

        eig_org, org_K, org_dis = pca(orig, topk=1)
        eig_rig, rig_K, rig_dis = pca(moved, topk=1)

        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        sns.barplot(x=np.arange(len(eig_org)),
                    y=np.around(eig_org, 2), palette="rocket", ax=ax1)
        sns.barplot(x=np.arange(len(eig_rig)),
                    y=np.around(eig_rig, 2), palette="rocket", ax=ax2)
        ax1.bar_label(ax1.containers[0])
        ax2.bar_label(ax2.containers[0])
        ax1.set_title(f"Eigenvalues of original image {name}")
        ax2.set_title(f"Eigenvalues of registered image {name}")
        plt.savefig(os.path.join(conf.result, f"{name}_pca_barplot.png"))
        plt.close()
        df = pd.concat([df, pd.DataFrame(
            [[name, org_mse, rig_mse, org_dis, rig_dis, mean_orig_T1err, mean_rigs_T1err]], columns=col)], ignore_index=True)

    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df['T1err changes percentage'] = percentage_change(
        df['raw T1err'], df['registered T1err'])
    df.to_csv(os.path.join(conf.result, 'results.csv'), index=False)
    hydralog.info(
        f"The summary is \n {df[['MSE changes percentage', 'PCA changes percentage', 'T1err changes percentage']].describe()}")
    logger.log_dataframe(df, 'Final Results', path=os.path.join(conf.result, 'results.csv'))

def map(t1map, data, tvec, conf, name, label):
    filename = os.path.join(conf.result, f"{name}_{label}_T1map.pickle")
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            re = pickle.load(handle)
        pmap = re['pmap']
        sdmap = re['sdmap']
        null_index = re['null_index']
        S = re['S']
        orig_T1err = S[None, None, :, :]
    else:
        inversion_img, pmap, sdmap, null_index, S = t1map.mestimation_abs(-1, np.array(tvec), data)
        re = {}
        re['pmap'] = pmap
        re['sdmap'] = sdmap
        re['null_index'] = null_index
        re['S'] = S
        orig_T1err = S[None, None, :, :]
        with open(filename, 'wb') as handle:
            pickle.dump(re, handle)
    return orig_T1err

if __name__ == '__main__':
    main()
