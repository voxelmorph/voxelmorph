import argparse
import glob
import os
import shutil

import numpy as np
import SimpleITK as sitk
from utils import *

np.random.seed(821)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--n_test', default=10,
                        type=float, help='test_ratio')
    args = parser.parse_args()

    basepath = f"data/{args.dataset}"
    train_output = f"data/{args.dataset}_dataset/train"
    val_output = f"data/{args.dataset}_dataset/test"

    input_file_paths = []
    output_file_paths = []
    image_shapes = []
    if os.path.exists(train_output):
        shutil.rmtree(train_output)
    os.makedirs(train_output)
    if os.path.exists(val_output):
        shutil.rmtree(val_output)
    os.makedirs(val_output)

    files = glob.glob(f"{basepath}/*.nii.gz")
    # print(files)
    for file in files:
        input_file_paths.append(file)
        name = file.split("/")[-1][:-7]
        output_file_path = f"{train_output}/{name}"
        output_file_paths.append(output_file_path)

        image = sitk.ReadImage(file)
        image_shapes.append(image.GetSize())
    median_shape = (np.median(np.vstack(image_shapes), 0)).astype(np.int)
    max_shape = np.max(np.vstack(image_shapes), 0)
    min_shape = np.min(np.vstack(image_shapes), 0)

    print(f"Total file number {len(input_file_paths)}")
    print(
        f"The max shape is {max_shape}, median shape is {median_shape}, min_shape is {min_shape}")

    idxs = np.arange(len(input_file_paths))
    np.random.shuffle(idxs)
    test_idx = idxs[:args.n_test]

    min_idx = np.argmin(np.vstack(image_shapes), 0)
    min_dim_0 = image_shapes[min_idx[0]][0]
    min_dim_1 = image_shapes[min_idx[1]][1]
    print(f"Minimal shape {(min_dim_0, min_dim_1)}")

    output_train_filenames = []
    output_test_filenames = []

    for idx, file in enumerate(input_file_paths):
        image = sitk.ReadImage(file)
        original_shape = image_shapes[idx]
        new_shape = (min_dim_0, min_dim_1, original_shape[-1])
        resize_img = resizeSitkImg(image, new_shape, sitk.sitkLinear)

        resize_img_array = sitk.GetArrayFromImage(resize_img)
        tmp = (output_file_paths[idx]).split("/")[-1]

        if idx in test_idx:
            output_name = f"{val_output}/{tmp}.npy"
            np.save(output_name, resize_img_array)
            output_test_filenames.append(output_name)
        else:
            output_name = f"{train_output}/{tmp}.npy"
            np.save(output_name, resize_img_array)
            output_train_filenames.append(output_name)
    print(f"In training set, number of images {len(output_train_filenames)}")
    txt_string = "\n".join(output_train_filenames)
    with open(f'data/{args.dataset}_input.txt', "w") as f:
        f.write(txt_string)

    print(f"In test set, number of images {len(output_test_filenames)}")
    txt_string = "\n".join(output_test_filenames)
    with open(f'data/{args.dataset}_test.txt', "w") as f:
        f.write(txt_string)
