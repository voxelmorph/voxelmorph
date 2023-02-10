import glob
import logging
import multiprocessing
import os
import time
from pathlib import Path
import hydra
import pandas as pd
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
    conf.model_path = os.path.join(conf.model_dir, '%04d.pt' % conf.epochs)

    if conf.log == 'wandb':
        logger = WandbLogger(project_name=conf.wandb_project, cfg=conf)

    # save the config
    config_path = f"{conf['model_dir']}/config.yaml"
    os.makedirs(conf['model_dir'], exist_ok=True)
    try:
        with open(config_path, 'w') as fp:
            OmegaConf.save(config=conf, f=fp.name)
    except:
        hydralog.warning("Unable to copy the config")

    pipeline(conf, logger)
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


def generate_input(conf):
    if not conf.final:
        files = glob.glob(os.path.join(conf.moved, '*.npy'))
        txt_string = "\n".join(files)
        with open(f"{conf.moved}/{conf.dataset}_input.txt", "w") as f:
            f.write(txt_string)
        conf.img_list = f"{conf.moved}/{conf.dataset}_input.txt"


def pipeline(conf, logger=None):
    # first train the model with rpca rank=5?
    conf.final = False
    st = time.time()
    hydralog.info(f"{'---'*10} Round 1 {'---'*10}")
    conf.rank = conf.rpca_rank.rank1
    conf.round = 1
    conf.moving = f"data/{conf.dataset}_dataset/train"
    createdir(conf)
    train(conf, logger)
    train_time = time.time() - st
    validate(conf, logger)
    generate_input(conf)
    round_time = time.time() - st
    hydralog.info(
        f"{'---'*10} Round 1 train_t {train_time/60} mins and total_t {round_time/60} mins")

    hydralog.info(f"{'---'*10} Round 2 {'---'*10}")
    conf.rank = conf.rpca_rank.rank2
    conf.round = 2
    conf.moving = os.path.join(conf.inference, f"round{conf.round-1}", 'moved')
    createdir(conf)
    train(conf, logger)
    train_time = time.time() - st
    validate(conf, logger)
    generate_input(conf)
    round_time = time.time() - st
    hydralog.info(
        f"{'---'*10} Round 2 train_t {train_time/60} mins and total_t {round_time/60} mins")

    hydralog.info(f"{'---'*10} Round 3 {'---'*10}")
    conf.final = True
    conf.rank = conf.rpca_rank.rank3
    conf.round = 3
    conf.moving = os.path.join(conf.inference, f"round{conf.round-1}", 'moved')
    createdir(conf)
    train(conf, logger)
    train_time = time.time() - st
    validate(conf, logger)
    generate_input(conf)
    round_time = time.time() - st
    hydralog.info(
        f"{'---'*10} Round 3 train_t {train_time/60} mins and total_t {round_time/60} mins")


def validate(conf, logger=None):

    col = ['Cases', 'raw MSE', 'registered MSE', 'raw PCA',
           'registered PCA', 'raw T1err', 'registered T1err']
    df = pd.DataFrame(columns=col)

    # load the TI for all subjects
    if conf.TI_csv:
        TI_dict = csv_to_dict(conf.TI_csv)

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

    source_files = glob.glob(os.path.join(conf.moving, "*.npy"))
    hydralog.info("Registering Samples:")
    for idx, subject in enumerate(tqdm(source_files, desc="Registering Samples:")):
        if os.path.exists(os.path.join(conf.moved, f"{Path(subject).stem}.nii")):
            hydralog.debug(f"Already registered {Path(subject).stem}")
        else:
            name, loss_org, org_dis, t1err_org, loss_rig, rig_dis, t1err_rig = register_single(
                idx, conf, subject, TI_dict[Path(subject).stem], device, model, logger)
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

    logger.log_dataframe(df, f"{conf.round}_summary", path=os.path.join(
        conf.result, f"{conf.round}_summary"))
    return


if __name__ == '__main__':
    main()
