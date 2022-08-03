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
    if not os.path.exists(moving_folder):
        os.makedirs(moving_folder)
    if not os.path.exists(moved_folder):
        os.makedirs(moved_folder)
    if not os.path.exists(warp_folder):
        os.makedirs(warp_folder)
    
    source_files = os.listdir(moving_folder)
    subject_ids = [(x.split("_")[0], x.split("_")[3]) for x in source_files]
    unique_subjects = set(subject_ids)
    for subject in unique_subjects:
        output_files = []
        files = sorted([source_files[idx] for (idx, id) in enumerate(subject_ids) if id == subject])
        for file in files[1:]:
            new_file = file.rsplit(".", 1)[0] + ".nii"
            cmd = f"python scripts/torch/register.py --moving {moving_folder}/{file} --fixed {moving_folder}/{files[0]} --model {args.model} --moved {moved_folder}/{new_file} --warp {warp_folder}/{new_file}"
            print(cmd)
            os.system(cmd)
            output_files.append(new_file)
        timestamps = [int((x.split(".")[0]).split("_")[-1]) for x in output_files]
        index = np.argsort(timestamps)
        moved_vols = []
        for id in index:
            moved_vols.append((nib.load(f"{moved_folder}/{output_files[id]}")).get_data().squeeze())
        moved_vols = np.stack(moved_vols)
        vxm.py.utils.save_volfile(moved_vols, f"{moved_folder}/{subject[0]}_slice_{subject[1]}_aggregate.nii", None)
        print(f"Finish the register of {subject[0]}_slice_{subject[1]}")



