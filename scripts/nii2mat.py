import os
import glob
import argparse
import scipy.io
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib

def nii2mat(nii_path, mat_path):
    img = sitk.ReadImage(nii_path)
    img_array = sitk.GetArrayFromImage(img)
    img_array = img_array.transpose(1, 2, 0)
    scipy.io.savemat(mat_path, {'img': img_array})

def mat2nii(mat_path, nii_path):
    img_array = scipy.io.loadmat(mat_path)['img']
    # read your own space setting
    PixelSpacing = np.array([1, 1, 1])

    affine = np.diag(np.concatenate([PixelSpacing, [1]]))
    nii_img = nib.Nifti1Image(img_array, affine)
    nib.save(nii_img, nii_path)


def npy2mat(npy_path, mat_path):
    img_array = np.load(npy_path)
    img_array = img_array.transpose(1, 2, 0)
    print(img_array.shape)
    scipy.io.savemat(mat_path, {'img': img_array})


def npy2nii(npy_path, nii_path):
    img_array = np.load(npy_path)
    img_array = img_array.transpose(0, 2, 1)
    img = sitk.GetImageFromArray(img_array)
    sitk.WriteImage(img, nii_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Nifti path')
    parser.add_argument('--mat', default=None, help='Mat path')
    parser.add_argument('--nii', default=None, help='Nifti path')
    args = parser.parse_args()
    input = args.source
    
    if os.path.isdir(input):
        files = glob.glob(os.path.join(input, '*.npy'))
        print(files)
        for file in files:
            format = Path(file).suffix
            name = Path(file).stem
            if args.mat is not None:
                os.makedirs(args.mat, exist_ok=True)
                if format == ".nii":
                    print("nii")
                    nii2mat(file, f"{args.mat}/{name}.mat")
                elif format == ".npy":
                    npy2mat(file, f"{args.mat}/{name}.mat")
            if args.nii is not None:
                os.makedirs(args.nii, exist_ok=True)
                if format == ".npy":
                    npy2nii(file, f"{args.nii}/{name}.nii")
                if format == ".mat":
                    mat2nii(file, f"{args.nii}/{name}.nii")

