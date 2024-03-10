import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

import neurite as ne
from neurite.py.flow_color import flow_to_color, flow_key_map
from scripts.py.flow_utils import tform2dir_flow, dir2tform_flow
from voxelmorph.torch.layers import SpatialTransformer

# Compare these colormap schemes
cmaps = ['Baker', 'winter']  # use proposed Baker et al. color scheme



def flow_to_color_matplotlib(X, cmap_name):
    dy = X[:, :, 0]
    dx = X[:, :, 1]
    colors = np.arctan2(dy, dx)
    colors[np.isnan(colors)] = 0
    norm = Normalize()
    norm.autoscale(colors)

    cmap_show = matplotlib.colormaps[cmap_name]
    flow_rgb_show = cmap_show(norm(colors))
    return flow_rgb_show


def matplotlib_colorwheel(cmap_name, size=100):
    x = np.linspace(-1, 1, size)
    dy, dx = np.meshgrid(x, x)
    X = np.stack((dy, dx), axis=2)
    return flow_to_color_matplotlib(X, cmap_name)


flow_small_x = np.linspace(-1, 1, 15)
flow_small_dx, flow_small_dy = np.meshgrid(flow_small_x, flow_small_x)
flow_small_xy = np.stack([flow_small_dx, flow_small_dy], axis=2)


fig1, axes1 = plt.subplots(2, 2, figsize=(5, 5))

flow_wheel_big_baker = flow_key_map(100)
flow_wheel_big_winter = matplotlib_colorwheel(cmap_name='winter', size=100)

axes1[0, 0].imshow(flow_wheel_big_baker, extent=[-1, 1, 1, -1])
axes1[0, 0].set_title('Baker et al. Colorwheel\n(RGB Image)')

axes1[0, 1].imshow(flow_wheel_big_winter, extent=[-1, 1, 1, -1])
axes1[0, 1].set_title('Winter Colorwheel\n(RGB Image)')

ne.plot.flow_ax(flow_small_xy, ax=axes1[1, 0], scale=1.0, plot_block=False,
                mode='transformer', cmap='Baker',
                title=f'Baker et al. Colorwheel\n(arrows)',
                axis='on')
axes1[1, 0].set_xticks([])
axes1[1, 0].set_yticks([])

ne.plot.flow_ax(flow_small_xy, ax=axes1[1, 1], scale=1.0, plot_block=False,
                mode='transformer', cmap='winter', title=f'Winter \n(arrows)',
                axis='on')
axes1[1, 1].set_xticks([])
axes1[1, 1].set_yticks([])
plt.tight_layout()
a = 1
pass


for cmap in cmaps:
    # ne.plot.flow_legend()
    if cmap == 'Baker':
        flow_wheel_big = flow_key_map(100)
    else:
        flow_wheel_big = matplotlib_colorwheel(cmap_name=cmap, size=100)


    left_right_idx = 0
    up_down_idx = 1

    direction_flows_dict = {'Right (red)': {left_right_idx: 1, up_down_idx: 0},
                            # 'Left (turquoise)': {left_right_idx: -1, up_down_idx: 0},
                            'Up (blue)': {left_right_idx: 0, up_down_idx: -1},
                            'Down & Left (green)': {left_right_idx: -1, up_down_idx: 1},
                            'Left to center, Right down & right': {},  # define below
                            'Top to middle, Bottom down': {},  # define below
                            }

    # rightward flow (should be red)
    shape = (10, 15)
    src = np.zeros(shape)
    h, w = shape
    src[h // 4 + 1:  3 * h // 4,
        w // 4 + 1:  3 * w // 4] = 1

    flow_shape = shape + (2,)

    # Apply a scaling field from left to right to illustrate the effect of flows of
    # different magnitudes
    scaling_w = np.linspace(0.5, 1, w)
    scaling_w = np.tile(scaling_w.reshape([1, w, 1]), (h, 1, 2))


    def apply_transformer(tr: torch.nn.Module, x: np.ndarray, flow: np.ndarray):
        x_tensor = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
        flow_tensor = torch.Tensor(flow).unsqueeze(0)
        with torch.no_grad():
            y_tensor_out = tr.forward(x_tensor, flow_tensor)
        y_out = y_tensor_out.numpy()[0, 0]
        return y_out


    num_plots = len(direction_flows_dict) + 1
    transformer = SpatialTransformer(shape)

    plt_size = 3
    fig, axes = plt.subplots(3, num_plots, figsize=(plt_size * num_plots, plt_size * 3))
    line_ax = fig.add_axes([0.0, 0.0, 1, 1])  # [left, bottom, width, height]
    line_ax.axvline(x=1 / num_plots + .01, color='k', linestyle='--')
    line_ax.axis('off')

    axes[0, 0].imshow(flow_wheel_big)
    axes[0, 0].set_title('Flow Colorwheel\n(Image)')

    # axes[1, 0].imshow(flow_map_small)
    ne.plot.flow_ax(flow_small_xy, ax=axes[1, 0], scale=1.0, plot_block=False,
                    mode='transformer', cmap=cmap, title=f'Flow Colorwheel\n(arrows)',
                    axis='on')

    axes[2, 0].imshow(src)
    axes[2, 0].set_title('Source Image')

    all_flows = []
    half_h, half_w = shape[0] // 2, shape[1] // 2
    for i, (name, directions_vals) in enumerate(direction_flows_dict.items()):
        directional_flow_i = np.zeros(flow_shape)
        max_flow_val0 = 1.5

        if name == 'Top to middle, Bottom down':
            directional_flow_i[:half_h, :half_w, left_right_idx] = max_flow_val0
            directional_flow_i[:half_h, half_w:, left_right_idx] = -max_flow_val0
            directional_flow_i[half_h:, :, up_down_idx] = max_flow_val0
        elif name == 'Left to center, Right down & right':
            directional_flow_i[:half_h, :half_w, up_down_idx] = max_flow_val0
            directional_flow_i[half_h:, :half_w, up_down_idx] = -max_flow_val0
            directional_flow_i[:, half_w:, left_right_idx] = max_flow_val0
            directional_flow_i[:, half_w:, up_down_idx] = max_flow_val0

        else:
            for idx, val in directions_vals.items():
                directional_flow_i[:, :, idx] = val * max_flow_val0

        directional_flow_i = directional_flow_i * scaling_w

        directional_flow_i_plot = directional_flow_i / 2
        transformer_flow_i = dir2tform_flow(directional_flow_i)
        all_flows.append(directional_flow_i_plot)

        directional_flow_i2 = tform2dir_flow(transformer_flow_i)
        assert np.array_equal(directional_flow_i, directional_flow_i2)

        src_transformed = apply_transformer(transformer, src, transformer_flow_i)

        if cmap == 'Baker':
            flow_rgb = flow_to_color(directional_flow_i_plot)
        else:
            flow_rgb = flow_to_color_matplotlib(directional_flow_i_plot, cmap)

        col_i = int(i) + 1
        axes[0, col_i].imshow(flow_rgb)
        axes[0, col_i].set_title(f'{name}\n(RGB)')

        ne.plot.flow_ax(directional_flow_i_plot, ax=axes[1, col_i], scale=1.0,
                        plot_block=False, mode='transformer', cmap=cmap,
                        title=f'{name}\n(arrows)', axis='on')
        axes[1, col_i].axis('scaled')
        axes[1, col_i].set_xlim([-0.5, w - 0.5])
        axes[1, col_i].set_ylim([h - 0.5, -0.5])
        axes[2, col_i].imshow(src_transformed)
        axes[2, col_i].set_title(f'Transformer output')

    plt.suptitle(f'Flow fields using "{cmap}" color scheme', fontsize=20)
    plt.tight_layout()

    plt.show(block=False)

print('Done!')
