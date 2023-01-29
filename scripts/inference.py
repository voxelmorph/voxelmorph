from pathlib import Path
import time
import os
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

    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA',
           'registered PCA', 'raw T1err', 'registered T1err']
    df = pd.DataFrame(columns=col)

    # load the TI for all subjects
    if conf.TI_json:
        import json
        with open(f"{conf.TI_json}") as json_file:
            TI_dict = json.load(json_file)

    device = 'cpu'

    train_files = glover(conf.moving, "*.npy")
    add_feat_axis = not conf.multichannel

    for idx, subject in enumerate(tqdm(train_files, desc="Registering Samples:")):
        name = Path(subject).stem
        start = time.time()
        tvec = TI_dict[Path(subject).stem]
        orig_vols, fixed_affine = vxm.py.utils.load_volfile(os.path.join(
            conf.moving, f"{name}.npy"), add_feat_axis=add_feat_axis, ret_affine=True)
        rigs_vols, fixed_affine = vxm.py.utils.load_volfile(os.path.join(
            conf.moved, f"{name}.npy"), add_feat_axis=add_feat_axis, ret_affine=True)
        orig_vols = torch.from_numpy(orig_vols).float().permute(0, 3, 1, 2).to(device)
        rigs_vols = torch.from_numpy(rigs_vols).float().permute(0, 3, 1, 2).to(device)
        orig_T1err = vxm.groupwise.utils.update_atlas(
            orig_vols, -1, 't1map', tvec=tvec)
        rigs_T1err = vxm.groupwise.utils.update_atlas(
            rigs_vols, -1, 't1map', tvec=tvec)
        et = time.time()
        mean_orig_T1err = np.mean(orig_T1err)
        mean_rigs_T1err = np.mean(rigs_T1err)
        saveT1err(orig_T1err, rigs_T1err, conf, name, None)
        hydralog.info(
            f"{name}, Time elapsed: {(et - start)/60} mins, T1 error orig {mean_orig_T1err} and rigs {mean_rigs_T1err}")

        orig = np.squeeze(orig_vols.detach().cpu().numpy()).transpose(1, 2, 0)
        moved = np.squeeze(rigs_vols.detach().cpu().numpy()).transpose(1, 2, 0)

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
        plt.savefig(os.path.join(conf.result, f"{name[:-4]}_pca_barplot.png"))
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


if __name__ == '__main__':
    main()
