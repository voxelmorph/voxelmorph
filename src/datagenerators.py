import os
import numpy as np


def load_example_by_name(vol_name, seg_name):

    X = np.load(vol_name)['vol_data']
    X = np.reshape(X, (1,) + X.shape + (1,))

    return_vals = [X]

    X_seg = np.load(seg_name)['vol_data']
    X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
    return_vals.append(X_seg)

    return tuple(return_vals)


def example_gen(vol_names, batch_size=1, return_segs=False, seg_dir=None):
    #idx = 0
    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        for idx in idxes:
            X = np.load(vol_names[idx])['vol_data']
            X = np.reshape(X, (1,) + X.shape + (1,))
            X_data += [X]

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        if return_segs:
            X_data = []
            for idx in idxes:
                name = os.path.basename(vol_names[idx])
                X_seg = np.load(seg_dir + name[0:-8]+'aseg.npz')['vol_data']
                X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
                X_data += X_seg
            
            if batch_size > 1:
                return_vals.append(np.concatenate(X_data, 0))
            else:
                return_vals.append(X_data[0])

        # print vol_names[idx] + "," + seg_dir + name[0:-8]+'aseg.npz'

        yield tuple(return_vals)
