from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class MCCA(Layer):
    '''
    MCCA layer is used compute the MCCA objective
    
    '''

    def __init__(self, output_dim=1, feature_shape = 10, num_sets = 3, name='mcca_layer'):
        
        super(MCCA, self).__init__()
        self.output_dim = output_dim
        self.f = feature_shape
        self.N = num_sets
        self.name = name
        

    def build(self, input_shape):
        
        super(MCCA, self).build(input_shape)

    def call(self, x):
        
        # mean
        one = tf.constant([1.0])
        sample = tf.shape(x)[0]
        sample_float = tf.cast(sample, 'float')
        
        partition = tf.divide(one, sample_float)
        xbar = K.transpose(x) - partition * tf.matmul(K.transpose(x), tf.ones([sample, sample]))
        R = tf.matmul(xbar, tf.transpose(xbar))
        Rs = tf.Variable(tf.zeros_like(R))
        indices = []
        values = []
        for i in range(self.N):
            for j in range(self.f):
                for k in range(self.f):
                    indices.append([j + i * self.f, k + i * self.f])
                    values.append(R[j + i * self.f, k + i * self.f])
                    
        S = tf.scatter_nd_update(Rs, indices, values)
        T = tf.matmul(tf.linalg.inv(S + tf.constant([1e-6]) * tf.eye(self.f*self.N)), R - S)
    
        U, V = tf.linalg.eigh(T)
        U_sort, _ = tf.nn.top_k(U, 1)
        corr = K.sum(K.sqrt(U_sort))
        
        return -corr

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'sets_dim': self.N,
        }
        base_config = super(MCCA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))