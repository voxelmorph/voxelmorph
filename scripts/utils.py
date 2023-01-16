import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import gif


def resize_img(img, new_size, interpolator):
    # img = sitk.ReadImage(img)
    dimension = img.GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = new_size
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))

    # centered_transform = sitk.Transform(transform)
    # centered_transform.AddTransform(centering_transform)

    centered_transform = sitk.CompositeTransform([transform, centering_transform])

    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
    # no new labels are introduced.

    return sitk.Resample(img, reference_image, centered_transform, interpolator, 0.0)


def normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def pca(array, topk=1):
    """calculate the eigenvalues of data's(covariance matrix). And the top-K eigenvalue's percentage

    Args:
        array (_type_): input volumes. shape is (H, W, slice)
        topk (int, optional): The first K eigenvalues. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # 
    x, y, z = array.shape
    # print(f"Shape of array is {array.shape} and x is {x} and y is {y} and z is {z}")
    M = array.reshape(x*y, z)
    K = np.corrcoef(M.T)
    eigenvalues = np.linalg.eigvals(K)
    G = np.sum(eigenvalues)
    dis = np.sum(eigenvalues[:topk])/G

    return eigenvalues, K, dis


def percentage_change(col1, col2):
    return ((col2 - col1) / col1) * 100

@gif.frame
def help_mag_plot(data):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(data, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    
def save_gif(data, name, output_dir, label, duration=100):
    # save gif and use hist equalization
    rows, cols, slices = data.shape
    frames = []
    for slice in range(slices):

        # frames.append(np.stack((data[:, :, slice], data[:, :, slice], data[:, :, slice])))
        frames.append(help_mag_plot(data[:, :, slice]))
    os.makedirs(f"{output_dir}/gifs", exist_ok=True)
    path = f"{output_dir}/gifs/{label}_{name}.gif"
    gif.save(frames, path, duration=duration)
    # write_gif(frames, path)
    return path


# def save_quiver(data, name, output_dir):
#     os.makedirs(f"{output_dir}/quiver", exist_ok=True)
#     _, slices, rows, cols = data.shape
#     x, y = np.meshgrid(np.arange(0, rows, 1), np.arange(0, cols, 1))
#     quiver_list = []
#     for slice in range(slices):
#         u, v = data[0, slice, :, :], data[1, slice, :, :]

#         fig, ax = plt.subplots(figsize=(5,5))
#         ax.quiver(x, y, u, v, units='width')
#         ax.xaxis.set_ticks([])
#         ax.yaxis.set_ticks([])
#         ax.set_aspect('equal')
#         fig.canvas.draw()

# # grab the pixel buffer and dump it into a numpy array
#         X = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         image = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         # print(image.shape)
#         quiver_list.append(image)

#         # plt.savefig(f"{output_dir}/quiver/{name}_{slice}.png")
#         plt.close()
#     path = f"{output_dir}/quiver/{name}.gif"
#     write_gif(quiver_list, path)
#     return path


@gif.frame
def help_morph_plot(data, title_font_size=4):
    fig, ax = plt.subplots(figsize=(3,3))
    field = np.squeeze(data)
    bg_img = np.zeros_like(field[0, ...])
    plot_warped_grid(ax, field, bg_img, interval=3, title="$\phi_{pred}$", fontsize=title_font_size)


def save_morphField(data, name, output_dir, duration=100):
    _, slices, rows, cols = data.shape
    frames = []
    for slice in range(slices):
        frames.append(help_morph_plot(data[:, slice, :, :]))
    os.makedirs(f"{output_dir}/morph_field", exist_ok=True)
    path = f"{output_dir}/morph_field/{name}.gif"
    gif.save(frames, path, duration=duration)
    return path


def plot_result_fig(warp, pred, fixed, size=(8, 8), title_font_size=8):
    fig = plt.figure(figsize=size)
    title_pad = 10
    ax1 = fig.add_subplot(2, 2, 1)
    plt.imshow(fixed[0, 0, ...], cmap='gray')
    plt.axis('off')
    ax1.set_title('Target', fontsize=title_font_size, pad=title_pad)

    ax2 = fig.add_subplot(2, 2, 2)
    plt.imshow(pred[0, 0, ...], cmap='gray')
    plt.axis('off')
    ax2.set_title('Pred Target', fontsize=title_font_size, pad=title_pad)

    ax3 = fig.add_subplot(2, 2, 3)
    bg_img = np.zeros_like(warp[0, 0, ...])
    plot_warped_grid(ax3, warp[0, ...], bg_img, interval=3, title="$\phi_{pred}$", fontsize=title_font_size)

    ax4 = fig.add_subplot(2, 2, 4)
    error = pred - fixed
    plt.imshow(error[0, 0, ...], cmap='seismic')
    plt.axis('off')
    ax4.set_title('Difference', fontsize=title_font_size, pad=title_pad)
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.2)
    return fig


def plot_validation_fig(y_true, y_pred, new_atlas, atlas, disp, size=(8, 8), title_font_size=8):
    fig = plt.figure(figsize=size)
    title_pad = 10
    ax1 = fig.add_subplot(2, 3, 1)
    plt.imshow(y_true[0, 0, ...], cmap='gray')
    plt.axis('off')
    ax1.set_title('y_original', fontsize=title_font_size, pad=title_pad)

    ax2 = fig.add_subplot(2, 3, 2)
    plt.imshow(y_pred[0, 0, ...], cmap='gray')
    plt.axis('off')
    ax2.set_title('y_registered', fontsize=title_font_size, pad=title_pad)

    ax3 = fig.add_subplot(2, 3, 3)
    error = y_pred[0, 0, ...] - y_true[0, 0, ...]
    plt.imshow(error, cmap='seismic')
    # plt.colorbar(ax=ax3)
    plt.axis('off')
    ax3.set_title('Difference', fontsize=title_font_size, pad=title_pad)

    ax4 = fig.add_subplot(2, 3, 4)
    plt.imshow(new_atlas[0, 0, ...], cmap='gray')
    plt.axis('off')
    ax4.set_title('new_atlas', fontsize=title_font_size, pad=title_pad)

    ax5 = fig.add_subplot(2, 3, 5)
    error = new_atlas[0, 0, ...] - atlas[0, 0, ...]
    plt.imshow(error, cmap='seismic')
    plt.axis('off')
    ax5.set_title('atlas Difference', fontsize=title_font_size, pad=title_pad)


    ax6 = fig.add_subplot(2, 3, 6)
    bg_img = np.zeros_like(disp[0, 0, ...])
    plot_warped_grid(ax6, disp[0, ...], bg_img, interval=3, title="$\phi_{pred}$", fontsize=title_font_size)

    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.2)
    return fig



def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=30, color='c'):
    """disp shape (2, H, W)"""
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)