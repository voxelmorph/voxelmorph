import argparse
import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt
from sewar.full_ref import mse
from sklearn.decomposition import PCA
from array2gif import write_gif
from skimage import exposure


def pca(array, name, output_dir, label, n_components=10):
    # calculate the eigenvalues of data(covariance matrix)
    x, y, z = array.shape
    M = array.reshape(x*y, z)
    Sigma = np.diag(np.std(M, axis=0))
    M_avg = np.tile(np.average(M, axis=0), (x*y, 1))
    K = np.dot(np.dot(np.dot(np.linalg.inv(Sigma), (M - M_avg).T),
               (M - M_avg)), np.linalg.inv(Sigma)) / (x*y - 1)

    eigenvalues = np.linalg.eigvals(K)

    dis = z - np.sum(eigenvalues[:n_components])

    return eigenvalues, K, dis


def percentage_change(col1, col2):
    return ((col2 - col1) / col1) * 100


def save_gif(data, name, output_dir, label):
    data = exposure.equalize_hist(data)*255
    data_list = [np.stack((data[:, :, i], data[:, :, i], data[:, :, i]))
                 for i in range(data.shape[-1])]
    os.makedirs(f"{output_dir}/gifs", exist_ok=True)
    write_gif(data_list, f"{output_dir}/gifs/{label}_{name}.gif")

def save_quiver(data, name, output_dir):
    os.makedirs(f"{output_dir}/quiver", exist_ok=True)
    _, slices, rows, cols = data.shape
    x, y = np.meshgrid(np.arange(0, rows, 1), np.arange(0, cols, 1))
    for slice in range(slices):
        u, v = data[0, slice, :, :], data[1, slice, :, :]

        fig, ax = plt.subplots(figsize=(9,9))
        ax.quiver(x, y, u, v, units='width')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')

        plt.savefig(f"{output_dir}/quiver/{name}_{slice}.png")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig', required=True, help='original image folder')
    parser.add_argument('--registered', required=True,
                        help='registered image folder')
    parser.add_argument('--output', required=True, help='output folder')
    parser.add_argument('--warp', required=True, help='warp folder')
    args = parser.parse_args()
    orig_path = args.orig
    reg_path = args.registered
    output_path = args.output
    warp_path = args.warp

    os.makedirs(output_path, exist_ok=True)

    orig = sorted(glob.glob(f"{orig_path}/*.npy"))
    registed = sorted(glob.glob(f"{reg_path}/*.nii"))
    warped = sorted(glob.glob(f"{warp_path}/*.nii"))

    tests = []
    org_mse = []
    rig_mse = []
    org_pca = []
    rig_pca = []
    # for i in range(1,2):
    for i in range(len(registed)):
        rig = sitk.GetArrayFromImage(sitk.ReadImage(registed[i]))
        org = np.transpose(np.load(orig[i]), (2, 1, 0))
        print(warped[i])
        warp = sitk.GetArrayFromImage(sitk.ReadImage(warped[i]))

        loss_rig, loss_org = 0, 0
        name = (registed[i]).split("/")[-1]
        for j in range(1, rig.shape[-1]):
            loss_rig += mse(rig[:, :, j-1], rig[:, :, j])
            loss_org += mse(org[:, :, j-1], org[:, :, j])

        org_mse.append(loss_org)
        rig_mse.append(loss_rig)
        tests.append(name)

        eig_org, org_K, org_dis = pca(
            org, name, output_path, "original", n_components=20)
        eig_rig, rig_K, rig_dis = pca(
            rig, name, output_path, "registered", n_components=20)

        org_pca.append(org_dis)
        rig_pca.append(rig_dis)

        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        sns.barplot(x=np.arange(len(eig_org)),
                    y=np.around(eig_org, 2), palette="rocket", ax=ax1)
        sns.barplot(x=np.arange(len(eig_rig)),
                    y=np.around(eig_rig, 2), palette="rocket", ax=ax2)
        ax1.bar_label(ax1.containers[0])
        ax2.bar_label(ax2.containers[0])
        ax1.set_title(f"Eigenvalues of original image {name}")
        ax2.set_title(f"Eigenvalues of registered image {name}")
        plt.savefig(os.path.join(output_path, f"{name[:-4]}_pca_barplot.png"))

        save_gif(rig, name, output_path, "registered")
        save_gif(org, name, output_path, "original")
        print(f"File {name}, original MSE - {loss_org:.5f} PCA - {org_dis:.5f}, registered MSE - {loss_rig:5f} PCA - {rig_dis:.5f}")

        save_quiver(warp, name, output_path)

    df = pd.DataFrame({'Cases': tests, 'raw MSE': org_mse,
                      'registered MSE': rig_mse, 'raw PCA': org_pca, 'registered PCA': rig_pca})
    df['MSE changes percentage'] = percentage_change(
        df['raw MSE'], df['registered MSE'])
    df['PCA changes percentage'] = percentage_change(
        df['raw PCA'], df['registered PCA'])
    df.to_csv(os.path.join(output_path, 'results.csv'), index=False)
