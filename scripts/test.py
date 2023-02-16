import os
import glob
import argparse
import scipy.io
import numpy as np
from pathlib import Path
import SimpleITK as sitk

import voxelmorph_group as vxm  # nopep8

def npy2nii(npy_path, nii_path):
    img_array = np.load(npy_path)
    img_array = img_array.transpose(0, 2, 1)
    img = sitk.GetImageFromArray(img_array)
    sitk.WriteImage(img, nii_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Nifti path')
    parser.add_argument('--output', default=None, help='Mat path')
    args = parser.parse_args()
    input = args.source
    
    if os.path.isdir(input):
        files = glob.glob(os.path.join(input, '*.npy'))
        print(files)
        for file in files:
            format = Path(file).suffix
            name = Path(file).stem
            if args.output is not None:
                os.makedirs(args.output, exist_ok=True)
                if format == ".nii":
                    print("nii")
                elif format == ".npy":
                    data = np.load(file)
                    low_matrix, sparse_matrix = vxm.py.utils.rpca(np.squeeze(data).transpose(1, 2, 0), rank=5)  # (H, W, N)
                    np.save(f"{args.output}/{name}_low.npy", low_matrix.transpose(2, 0, 1))
