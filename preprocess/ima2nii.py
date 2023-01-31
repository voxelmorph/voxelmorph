import argparse
import os
import glob
import scipy.io
import pydicom
import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='dataset path')
parser.add_argument('--output', required=True, help='output path')
args = parser.parse_args()
root = args.dataset
output = args.output

subjects = glob.glob(f"{root}/*")
os.makedirs(output, exist_ok=True)
os.makedirs(output+'_mat', exist_ok=True)

PixelSpacing = None
for subject in subjects:
    imas = glob.glob(os.path.join(root, subject, '*.IMA'))
    vols = []
    T1 = []
    for idx, ima in enumerate(imas):
        img = pydicom.read_file(ima)
        vols.append(img.pixel_array)

        T1.append(img.InversionTime)
        print(img.InversionTime)

        if PixelSpacing is None:
            pix = np.array(img.PixelSpacing).astype(np.float32) 
            sli = np.array(img.SliceThickness).astype(np.float32)
            PixelSpacing = np.array([pix[0], pix[1], sli])

    vols = np.stack(vols, axis=-1)
    affine = np.diag(np.concatenate([PixelSpacing, [1]]))
    ni_img = nib.Nifti1Image(vols, affine)
    name = subject.split("/")[-1]
    nib.save(ni_img, os.path.join(output, f"{name}_T1w.nii"))

    
    tvec = np.array(T1)
    scipy.io.savemat(os.path.join(output+'_mat', f"{name}_T1w.mat"), 
                    {'img': vols, 'tvec': tvec})