# -*- coding: utf-8 -*-
"""
voxelwise cardiac T1 mapping

contributors: Yoon-Chul Kim, Khu Rai Kim, Hyelee Lee
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import copy
import argparse
from scipy.optimize import curve_fit
from utils import *

def func_orig(x, a, b, c):
    return a*(1-np.exp(-b*x)) + c
    # return a - b * np.exp(-x / c)


def calc_t1value(j, ir_img, inversiontime):

    nx, ny, nti = ir_img.shape
    inversiontime = np.asarray(inversiontime)
    y = np.zeros(nti)
    r = int(j/ny)
    c = int(j%ny)

    p0_initial = [350, 0.005, -150]

    for tino in range(nti):
        y[tino] = ir_img[r,c,tino]

    yf = copy.copy(y)
    sq_err = 100000000.0
    curve_fit_success = False

    if nti == 3:
        iter = 1
    else:
        iter = 5
    for nsignflip in range(iter):
        if nsignflip == 0:
            yf[0] = -y[0]
        elif nsignflip == 1:
            yf[0] = -y[0]
            yf[1] = -y[1]
        elif nsignflip == 2:
            yf[0] = -y[0]
            yf[1] = -y[1]
            yf[2] = -y[2]
        elif nsignflip == 3:
            yf[0] = -y[0]
            yf[1] = -y[1]
            yf[2] = -y[2]
            yf[3] = -y[3]
        elif nsignflip == 4:
            yf[0] = -y[0]
            yf[1] = -y[1]
            yf[2] = -y[2]
            yf[3] = -y[3]
            yf[4] = -y[4]
        try:
            popt,pcov = curve_fit(func_orig, inversiontime, yf, p0=p0_initial)
        except RuntimeError:
            # print("Error - curve_fit failed")
            # curve_fit_success = False
            popt = p0_initial

        a1 = popt[0]
        b1 = popt[1]
        c1 = popt[2]

        yf_est = func_orig(inversiontime, a1, b1, c1)
        sq_err_curr = np.sum((yf_est - yf)**2, dtype=np.float32)

        if sq_err_curr < sq_err:
            curve_fit_success = True
            sq_err = sq_err_curr
            a1_opt = a1
            b1_opt = b1
            c1_opt = c1

    if not curve_fit_success:
        a1_opt = 0
        b1_opt = np.iinfo(np.float32).max
        c1_opt = 0

    return a1_opt, b1_opt, c1_opt


def calculate_T1map(ir_img, inversiontime):

    nx, ny, nti = ir_img.shape
    t1map = np.zeros([nx, ny, 3])
    ir_img = copy.copy(ir_img)
    if inversiontime[-1] == 0:
        inversiontime = inversiontime[0:-1]
        nTI = inversiontime.shape[0]
        if nti > nTI:
            ir_img = ir_img[:,:,0:nTI]

    for j in range(nx*ny):
        r = int(j / ny)
        c = int(j % ny)
        t1map[r, c, :] = calc_t1value(j, ir_img, inversiontime)

    return t1map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Nifti path')
    parser.add_argument('--output', help='Mat path')
    parser.add_argument('--inversiontime', help='Inversion time')
    args = parser.parse_args()
    input = args.input
    os.makedirs(args.output, exist_ok=True)
    TI_dict = csv_to_dict(args.inversiontime)
    if os.path.isdir(input):
        files = glob.glob(os.path.join(input, '*.npy'))
        for file in files:
            name = Path(file).stem
            frames = np.load(file).transpose(1, 2, 0)
            tvec = np.array(list(TI_dict[name].values())[1:], dtype=np.float32)
            print(name, tvec)
            t1_params_pre = calculate_T1map(frames, tvec)

            a = t1_params_pre[:, :, 0]
            b = t1_params_pre[:, :, 1]
            c = t1_params_pre[:, :, 2]
            t1 = (1 / b) * (a / (a + c) - 1)

            plt.imshow(t1, cmap='jet', vmin=0, vmax=2000)
            plt.colorbar()
            plt.axis('off')
            plt.savefig(os.path.join(args.output, f"{name}.png"))
            plt.close()