import os
import argparse
import sys
import numpy as np
import nibabel as nib

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving-folder', required=True, help='moving image (source) folder')

    parser.add_argument('--moved-folder', required=True, help='warped image output folder')
    parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('--warp-folder', help='output warp deformation folder')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')
    args = parser.parse_args()

    moving_folder = args.moving_folder
    moved_folder = args.moved_folder
    warp_folder = args.warp_folder

    os.makedirs(moved_folder, exist_ok=True)
    os.makedirs(warp_folder, exist_ok=True)
    
    source_files = os.listdir(moving_folder)
    for subject in source_files:
        name = subject.split(".")[0] + '.nii'
        cmd = f"python scripts/torch/register_3dvol.py --moving {moving_folder}/{subject} --fixed {moving_folder}/{subject} --model {args.model} --moved {moved_folder}/{name} --warp {warp_folder}/{name}"
        print(cmd)
        os.system(cmd)