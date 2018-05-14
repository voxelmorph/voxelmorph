"""
voxelwarp

3d spatial transformer 
"""

# third party
from keras.layers.core import Layer
import tensorflow as tf


class Dense3DSpatialTransformer(Layer):

    def __init__(self, **kwargs):
        super(Dense3DSpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) > 3:
            raise Exception('Spatial Transformer must be called on a list of length 2 or 3. '
                            'First argument is the image, second is the offset field.')

        if len(input_shape[1]) != 5 or input_shape[1][4] != 3:
            raise Exception('Offset field must be one 5D tensor with 3 channels. '
                            'Got: ' + str(input_shape[1]))

        self.built = True

    def call(self, inputs):
        return self._transform(inputs[0], inputs[1][:, :, :, :, 0],
                               inputs[1][:, :, :, :, 1], inputs[1][:, :, :, :, 2])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _transform(self, I, dx, dy, dz):

        batch_size = tf.shape(dx)[0]
        height = tf.shape(dx)[1]
        width = tf.shape(dx)[2]
        depth = tf.shape(dx)[3]

        # Convert dx and dy to absolute locations
        x_mesh, y_mesh, z_mesh = self._meshgrid(height, width, depth)
        x_mesh = tf.expand_dims(x_mesh, 0)
        y_mesh = tf.expand_dims(y_mesh, 0)
        z_mesh = tf.expand_dims(z_mesh, 0)

        x_mesh = tf.tile(x_mesh, [batch_size, 1, 1, 1])
        y_mesh = tf.tile(y_mesh, [batch_size, 1, 1, 1])
        z_mesh = tf.tile(z_mesh, [batch_size, 1, 1, 1])
        x_new = dx + x_mesh
        y_new = dy + y_mesh
        z_new = dz + z_mesh

        return self._interpolate(I, x_new, y_new, z_new)

    def _repeat(self, x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, dtype='int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _meshgrid(self, height, width, depth):
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(0.0,
                                                                tf.cast(width, tf.float32)-1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0,
                                                   tf.cast(height, tf.float32)-1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t = tf.tile(tf.expand_dims(x_t, 2), [1, 1, depth])
        y_t = tf.tile(tf.expand_dims(y_t, 2), [1, 1, depth])

        z_t = tf.linspace(0.0, tf.cast(depth, tf.float32)-1.0, depth)
        z_t = tf.expand_dims(tf.expand_dims(z_t, 0), 0)
        z_t = tf.tile(z_t, [height, width, 1])

        return x_t, y_t, z_t

    def _interpolate(self, im, x, y, z):

        im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "CONSTANT")

        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        depth = tf.shape(im)[3]
        channels = tf.shape(im)[4]

        out_height = tf.shape(x)[1]
        out_width = tf.shape(x)[2]
        out_depth = tf.shape(x)[3]

        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        z = tf.reshape(z, [-1])

        x = tf.cast(x, 'float32')+1
        y = tf.cast(y, 'float32')+1
        z = tf.cast(z, 'float32')+1

        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        max_z = tf.cast(depth - 1, 'int32')

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), 'int32')
        z1 = z0 + 1

        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)
        z0 = tf.clip_by_value(z0, 0, max_z)
        z1 = tf.clip_by_value(z1, 0, max_z)

        dim3 = depth
        dim2 = depth*width
        dim1 = depth*width*height
        base = self._repeat(tf.range(num_batch)*dim1,
                            out_height*out_width*out_depth)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0*dim3 + z0
        idx_b = base_y1 + x0*dim3 + z0
        idx_c = base_y0 + x1*dim3 + z0
        idx_d = base_y1 + x1*dim3 + z0
        idx_e = base_y0 + x0*dim3 + z1
        idx_f = base_y1 + x0*dim3 + z1
        idx_g = base_y0 + x1*dim3 + z1
        idx_h = base_y1 + x1*dim3 + z1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)
        Ie = tf.gather(im_flat, idx_e)
        If = tf.gather(im_flat, idx_f)
        Ig = tf.gather(im_flat, idx_g)
        Ih = tf.gather(im_flat, idx_h)

        # and finally calculate interpolated values
        x1_f = tf.cast(x1, 'float32')
        y1_f = tf.cast(y1, 'float32')
        z1_f = tf.cast(z1, 'float32')

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = tf.expand_dims((dz * dx * dy), 1)
        wb = tf.expand_dims((dz * dx * (1-dy)), 1)
        wc = tf.expand_dims((dz * (1-dx) * dy), 1)
        wd = tf.expand_dims((dz * (1-dx) * (1-dy)), 1)
        we = tf.expand_dims(((1-dz) * dx * dy), 1)
        wf = tf.expand_dims(((1-dz) * dx * (1-dy)), 1)
        wg = tf.expand_dims(((1-dz) * (1-dx) * dy), 1)
        wh = tf.expand_dims(((1-dz) * (1-dx) * (1-dy)), 1)

        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id,
                           we*Ie, wf*If, wg*Ig, wh*Ih])
        output = tf.reshape(output, tf.stack(
            [-1, out_height, out_width, out_depth, channels]))
        return output
