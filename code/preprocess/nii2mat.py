import argparse
import scipy.io
import numpy as np
from pathlib import Path
import SimpleITK as sitk

def nii2mat(nii_path, mat_path):
    img = sitk.ReadImage(nii_path)
    img_array = sitk.GetArrayFromImage(img)
    img_array = img_array.transpose(1, 2, 0)
    scipy.io.savemat(mat_path, {'img': img_array})


def npy2mat(npy_path, mat_path):
    img_array = np.load(npy_path)
    img_array = img_array.transpose(1, 2, 0)
    print(img_array.shape)
    scipy.io.savemat(mat_path, {'img': img_array})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Nifti path')
    parser.add_argument('--mat', help='Mat path')
    args = parser.parse_args()
    input = args.source
    format = Path(input).suffix
    print(format)
    if format == ".nii":
        print("nii")
        nii2mat(input, args.mat)
    elif format == ".npy":
        npy2mat(input, args.mat)

