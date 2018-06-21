from keras.layers.core import Layer
import tensorflow as tf
class Dense2DSpatialTransformer(Layer):
    """
    The layer takes an input image and an offset field and generates a transformed version of the image
    """

    def __init__(self, **kwargs):
        super(Dense2DSpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) > 3:
            raise Exception('Spatial Transformer must be called on a list of length 2 or 3. '
                            'First argument is the image, second is the offset field.')

        if len(input_shape[1]) != 4 or input_shape[1][3] != 2:
            raise Exception('Offset field must be one 4D tensor with 2 channels. '
                            'Got: ' + str(input_shape[1]))

        self.built = True

    def call(self, inputs):
        return self._transform(inputs[0], inputs[1][:, :, :, 0],
                               inputs[1][:, :, :, 1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _transform(self, I, dx, dy):

        batch_size = tf.shape(dx)[0]
        height = tf.shape(dx)[1]
        width = tf.shape(dx)[2]

        # Convert dx and dy to absolute locations
        x_mesh, y_mesh = self._meshgrid(height, width)
        x_mesh = tf.expand_dims(x_mesh, 0)
        y_mesh = tf.expand_dims(y_mesh, 0)

        x_mesh = tf.tile(x_mesh, [batch_size, 1, 1])
        y_mesh = tf.tile(y_mesh, [batch_size, 1, 1])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self._interpolate(I, x_new, y_new)

    def _repeat(self, x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, dtype='int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _meshgrid(self, height, width):
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(0.0,
                                                                tf.cast(width, tf.float32)-1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0,
                                                   tf.cast(height, tf.float32)-1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        return x_t, y_t

    def _interpolate(self, im, x, y):

        im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        out_height = tf.shape(x)[1]
        out_width = tf.shape(x)[2]

        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        x = tf.cast(x, 'float32')+1
        y = tf.cast(y, 'float32')+1

        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)
        
        dim3 = 1
        dim2 = width
        dim1 = width*height
        base = self._repeat(tf.range(num_batch)*dim1,
                            out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0*dim3 
        idx_b = base_y1 + x0*dim3
        idx_c = base_y0 + x1*dim3
        idx_d = base_y1 + x1*dim3
        idx_e = base_y0 + x0*dim3
        idx_f = base_y1 + x0*dim3
        idx_g = base_y0 + x1*dim3
        idx_h = base_y1 + x1*dim3

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

        dx = x1_f - x
        dy = y1_f - y

        wa = tf.expand_dims((dx * dy), 1)
        wb = tf.expand_dims((dx * (1-dy)), 1)
        wc = tf.expand_dims(((1-dx) * dy), 1)
        wd = tf.expand_dims(((1-dx) * (1-dy)), 1)
        we = tf.expand_dims((dx * dy), 1)
        wf = tf.expand_dims((dx * (1-dy)), 1)
        wg = tf.expand_dims(((1-dx) * dy), 1)
        wh = tf.expand_dims(((1-dx) * (1-dy)), 1)

        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id,
                           we*Ie, wf*If, wg*Ig, wh*Ih])
        output = tf.reshape(output, tf.stack(
            [-1, out_height, out_width, channels]))
        return output
