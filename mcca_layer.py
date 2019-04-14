from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np


def my_eigen(x):
    return np.linalg.eigh(x)

def my_svd(x):
    return np.linalg.svd(x, compute_uv=False)

def my_inv(x):
    return np.linalg.inv(x)

class MCCA(Layer):
    '''MCCA layer is used compute the MCCA objective

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.


    # Arguments
        output_dim: output dimension, default 1, i.e., correlation coefficient
        use_all_singular_value: if use the top-k singular values
        cca_space_dim: the number of singular values, i.e., k

    '''

    def __init__(self, output_dim=1, use_all_singular_values=True, mcca_space_dim=10, **kwargs):
        
        self.output_dim = output_dim
        self.mcca_space_dim = mcca_space_dim
        self.use_all_singular_values = use_all_singular_values
        super(MCCA, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MCCA, self).build(input_shape)

    def call(self, x):
        '''
        r1 = tf.constant([1e-4])
        r2 = tf.constant([1e-4])
        r3 = tf.constant([1e-4])
        eps = tf.constant([1e-12])
        '''
        
        o1 = o2 = o3 = tf.shape(x)[1] // 3

        H1 = K.transpose(x[:, 0:o1])
        H2 = K.transpose(x[:, o1:(o1 + o2)])
        H3 = K.transpose(x[:, (o1 + o2):(o1 + o2 + o3)])

        one = tf.constant([1.0])
        samplenum = tf.shape(H1)[1]
        samplenum_float = tf.cast(samplenum, 'float')

        # minus the mean value
        partition = tf.divide(one, samplenum_float)
        H1bar = H1 - partition * tf.matmul(H1, tf.ones([samplenum, samplenum]))
        H2bar = H2 - partition * tf.matmul(H2, tf.ones([samplenum, samplenum]))
        H3bar = H3 - partition * tf.matmul(H3, tf.ones([samplenum, samplenum]))
        
        # calculate R and S
        sigma = tf.matmul(tf.transpose(H1bar), H2bar)
        RB = tf.matmul(tf.transpose(H1bar), H2bar) + tf.matmul(tf.transpose(H1bar), H3bar) + \
             tf.matmul(tf.transpose(H2bar), H1bar) + tf.matmul(tf.transpose(H2bar), H3bar) + \
             tf.matmul(tf.transpose(H3bar), H1bar) + tf.matmul(tf.transpose(H3bar), H2bar)
        RW = tf.matmul(tf.transpose(H1bar), H1bar) + tf.matmul(tf.transpose(H2bar), H2bar) + tf.matmul(tf.transpose(H3bar), H3bar)

        T = tf.matmul(tf.linalg.inv(RW + tf.constant([1e-4]) * tf.eye(tf.shape(sigma)[0])), RB)
    
        U, V = tf.linalg.eigh(T)
        U_sort, _ = tf.nn.top_k(U, 1)
        corr = K.sum(K.sqrt(U_sort))
        
        return -corr

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'cca_dim': self.cca_dim,
            'use_all_singular_values': self.use_all_singular_values,
        }
        base_config = super(MCCA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))