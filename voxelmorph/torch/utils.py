import torch
import os


def setup_device(gpu_id=None):
    """
    Configures the appropriate device from a cuda device string.
    Returns the device id and total number of devices.
    """

    if gpu_id is not None and not isinstance(gpu_id, str):
        gpu_id = str(gpu_id)

    if gpu_id is not None:
        nb_devices = len(gpu_id.split(','))
    else:
        nb_devices = 1

    if gpu_id is not None and (gpu_id != '-1'):
        # device = 'cuda:' + gpu_id
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    return device, nb_devices
