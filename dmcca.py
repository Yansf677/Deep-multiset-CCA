
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, concatenate, normalization
from mcca_layer import MCCA

def constant_loss(y_true, y_pred):
    return y_pred

if __name__ == '__main__':

    # size of the input for view 1 and view 2
    input_shape = 52
    hidden_shape = 20
    feature_shape = 10
    activation1 = 'tanh'
    
    # network settings
    epoch_num  = 500
    batch_size = 100

    # load data
    train_x = np.load('train_data.npy')

    # 网络结构
    # net1
    input1 = Input(shape=(input_shape, ), name='input1')
    net1_1 = Dense(hidden_shape, activation=activation1, name='x1_1')(input1)
    net1_2 = Dense(hidden_shape, activation=activation1, name='x1_2')(net1_1)
    net1_3 = Dense(hidden_shape, activation=activation1,  name='x1_3')(net1_2)
    net1_out = Dense(feature_shape, activation='linear', name='x1_4')(net1_3)
    
    # net2
    input2 = Input(shape=(input_shape, ), name='input2')
    net2_1 = Dense(hidden_shape, activation=activation1,  name='x2_1')(input2)
    net2_2 = Dense(hidden_shape, activation=activation1,  name='x2_2')(net2_1)
    net2_3 = Dense(hidden_shape, activation=activation1, name='x2_3')(net2_2)
    net2_out = Dense(feature_shape, activation='linear', name='x2_4')(net2_3)
    
    # net3
    input3 = Input(shape=(input_shape, ), name='input3')
    net3_1 = Dense(hidden_shape, activation=activation1,  name='x3_1')(input3)
    net3_2 = Dense(hidden_shape, activation=activation1,  name='x3_2')(net3_1)
    net3_3 = Dense(hidden_shape, activation=activation1, name='x3_3')(net3_2)
    net3_out = Dense(feature_shape, activation='linear', name='x3_4')(net3_3)
    
    # net4
    input4 = Input(shape=(input_shape, ), name='input4')
    net4_1 = Dense(hidden_shape, activation=activation1,  name='x4_1')(input4)
    net4_2 = Dense(hidden_shape, activation=activation1,  name='x4_2')(net4_1)
    net4_3 = Dense(hidden_shape, activation=activation1, name='x4_3')(net4_2)
    net4_out = Dense(feature_shape, activation='linear', name='x4_4')(net4_3)

    # feature layer
    shared_layer = concatenate([net1_out, net2_out, net3_out, net4_out], name='shared_layer')
    
    BN = normalization.BatchNormalization()(shared_layer)
    
    mcca_layer = MCCA(1, feature_shape, 4, name='cca_layer')(BN)

    model = Model(inputs=[input1, input2, input3, input4], outputs=mcca_layer)
    
    model.compile(optimizer='sgd', loss=constant_loss)
    
    model.fit([train_x, train_x, train_x, train_x], np.zeros(len(train_x)), batch_size=batch_size, epochs=epoch_num, shuffle=True)
    
    #model.save('current_dcca.h5') # 保存模型
    


    
