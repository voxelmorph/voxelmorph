import argparse
import scipy.io
import numpy as np
import SimpleITK as sitk

def nii2mat(nii_path, mat_path):
    img = sitk.ReadImage(nii_path)
    img_array = sitk.GetArrayFromImage(img)
    scipy.io.savemat(mat_path, {'img': img_array})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nii', help='Nifti path')
    parser.add_argument('--mat', help='Mat path')
    args = parser.parse_args()
    nii2mat(args.nii, args.mat)

