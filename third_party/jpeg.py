''' 
    The bit depth reduction defense method. 
    Code copied from https://github.com/thu-ml/ares/blob/main/ares/defense/jpeg_compression.py
'''

import tensorflow as tf

def jpeg_compress(xs, x_min, x_max, quality=95):
    ''' Run jpeg compress on xs.

    :param xs: A batch of images to compress.
    :param x_min: The minimum value of xs.
    :param x_max: The maximum value of xs.
    :param quality: Jpeg compress quality.
    :return: Compressed images tensor with same numerical scale to the input image.
    '''
    # due to tf.custom_gradient's limitation, we need a wrapper here
    @tf.custom_gradient
    def jpeg_compress_op(xs_tf):
        # batch_size x width x height x channel
        imgs = tf.cast((xs_tf - x_min) / ((x_max - x_min) / 255.0), tf.uint8)
        imgs_jpeg = tf.map_fn(lambda img: tf.image.decode_jpeg(tf.image.encode_jpeg(img, quality=quality)), imgs)
        imgs_jpeg.set_shape(xs_tf.shape)

        def jpeg_compress_grad(d_output):
            return d_output

        return tf.cast(imgs_jpeg, xs_tf.dtype) / (255.0 / (x_max - x_min)) + x_min, jpeg_compress_grad

    return jpeg_compress_op(xs)