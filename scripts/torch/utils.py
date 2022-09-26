import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from array2gif import write_gif

def pca(array, name, output_dir, label, n_components=10):
    # calculate the eigenvalues of data(covariance matrix)
    x, y, z = array.shape
    # print(f"Shape of array is {array.shape} and x is {x} and y is {y} and z is {z}")
    M = array.reshape(x*y, z)
    Sigma = np.diag(np.std(M, axis=0))
    # print(f"Shape of Sigma is {Sigma.shape} and M is {M.shape} and sigma {Sigma}")
    M_avg = np.tile(np.average(M, axis=0), (x*y, 1))
    K = np.dot(np.dot(np.dot(np.linalg.inv(Sigma), (M - M_avg).T),
               (M - M_avg)), np.linalg.inv(Sigma)) / (x*y - 1)

    eigenvalues = np.linalg.eigvals(K)

    dis = z - np.sum(eigenvalues[:n_components])

    return eigenvalues, K, dis


def percentage_change(col1, col2):
    return ((col2 - col1) / col1) * 100


def save_gif(data, name, output_dir, label):
    # save gif and use hist equalization
    data = exposure.equalize_hist(data)*255
    data_list = [np.stack((data[:, :, i], data[:, :, i], data[:, :, i]))
                 for i in range(data.shape[-1])]
    os.makedirs(f"{output_dir}/gifs", exist_ok=True)
    path = f"{output_dir}/gifs/{label}_{name}.gif"
    write_gif(data_list, path)
    return path


def save_quiver(data, name, output_dir):
    os.makedirs(f"{output_dir}/quiver", exist_ok=True)
    _, slices, rows, cols = data.shape
    x, y = np.meshgrid(np.arange(0, rows, 1), np.arange(0, cols, 1))
    quiver_list = []
    for slice in range(slices):
        u, v = data[0, slice, :, :], data[1, slice, :, :]

        fig, ax = plt.subplots(figsize=(9,9))
        ax.quiver(x, y, u, v, units='width')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
        fig.canvas.draw()

# grab the pixel buffer and dump it into a numpy array
        X = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # print(image.shape)
        quiver_list.append(image)

        # plt.savefig(f"{output_dir}/quiver/{name}_{slice}.png")
        plt.close()
    path = f"{output_dir}/quiver/{name}.gif"
    write_gif(quiver_list, path)
    return path