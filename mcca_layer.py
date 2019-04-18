from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class MCCA(Layer):
    '''
    MCCA layer is used compute the MCCA objective
    
    '''

    def __init__(self, output_dim=1, feature_shape = 10, num_sets = 3, **kwargs):
        
        self.output_dim = output_dim
        self.f = feature_shape
        self.N = num_sets
        super(MCCA, self).__init__(**kwargs)

    def build(self, input_shape):
        
        super(MCCA, self).build(input_shape)

    def call(self, x):

        eps = tf.constant([1e-4])
        
        # mean
        one = tf.constant([1.0])
        sample = tf.shape(x)[0]
        sample_float = tf.cast(sample, 'float')
        
        partition = tf.divide(one, sample_float)
        xbar = K.transpose(x) - partition * tf.matmul(K.transpose(x), tf.ones([sample, sample]))
        R = tf.matmul(xbar, tf.transpose(xbar))
        S = tf.zeros_like(R)
        
        for i in range(self.N):
            S[i*self.f:(i+1)*self.f,i*self.f:(i+1)*self.f] = tf.assign_add(S[i*self.f:(i+1)*self.f,i*self.f:(i+1)*self.f], R[i*self.f:(i+1)*self.f,i*self.f:(i+1)*self.f])
        '''
        f = tf.shape(x)[1] // self.N
        Rw = tf.zeros([f,f])
        RT = tf.zeros([f,f])
        
        for k1 in range(self.N):
            Rw = Rw + sigma_xbar[k1*f:(k1+1)*f, k1*f:(k1+1)*f] / (self.N)
        
        for ki in range(self.N):
            for kj in range(self.N):
                RT = RT + sigma_xbar[ki*f:(ki+1)*f, kj*f:(kj+1)*f] / (self.N * (self.N - 1))
        '''
        T = tf.matmul(tf.linalg.inv(S + eps * tf.eye(self.f*self.N)), R - S)
    
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