
# Third party inports
import tensorflow as tf
import numpy as np

# batch_sizexheightxwidthxdepthxchan


def diceLoss(y_true, y_pred):
    top = 2*tf.reduce_sum(y_true * y_pred, [1, 2, 3])
    bottom = tf.maximum(tf.reduce_sum(y_true+y_pred, [1, 2, 3]), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


def gradientLoss(penalty='l1'):
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        return d/3.0

    return loss


def gradientLoss2D():
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

        dy = dy * dy
        dx = dx * dx

        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)
        return d/2.0

    return loss


def cc3D(win=[9, 9, 9], voxel_weights=None):
    def loss(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights

        return -1.0*tf.reduce_mean(cc)

    return loss


def cc2D(win=[9, 9]):
    def loss(I, J):
        I2 = tf.multiply(I, I)
        J2 = tf.multiply(J, J)
        IJ = tf.multiply(I, J)

        sum_filter = tf.ones([win[0], win[1], 1, 1])

        I_sum = tf.nn.conv2d(I, sum_filter, [1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv2d(J, sum_filter, [1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv2d(I2, sum_filter, [1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv2d(J2, sum_filter, [1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv2d(IJ, sum_filter, [1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
        return -1.0*tf.reduce_mean(cc)
    return loss
