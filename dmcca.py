
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, concatenate, normalization
from mcca_layer import MCCA

def constant_loss(y_true, y_pred):
    return y_pred

if __name__ == '__main__':

    # size of the input for view 1 and view 2
    input_shape1 = 52
    input_shape2 = 52
    input_shape3 = 52
    
    feature_shape1 = 10
    feature_shape2 = 10
    feature_shape3 = 10
    
    # network settings
    epoch_num  = 500
    batch_size = 200

    # load data
    train_x = np.load('train_data.npy')

    # 网络结构
    # net1
    input1 = Input(shape=(input_shape1, ), name='input1')
    net1_1 = Dense(10, activation='relu', name='x1_1')(input1)
    net1_2 = Dense(10, activation='relu', name='x1_2')(net1_1)
    net1_3 = Dense(10, activation='relu',  name='x1_3')(net1_2)
    net1_out = Dense(feature_shape1, activation='linear', name='x1_4')(net1_3)
    
    # net2
    input2 = Input(shape=(input_shape2, ), name='input2')
    net2_1 = Dense(10, activation='relu',  name='x2_1')(input2)
    net2_2 = Dense(10, activation='relu',  name='x2_2')(net2_1)
    net2_3 = Dense(10, activation='relu', name='x2_3')(net2_2)
    net2_out = Dense(feature_shape2, activation='linear', name='x2_4')(net2_3)
    
    # net3
    input3 = Input(shape=(input_shape3, ), name='input3')
    net3_1 = Dense(10, activation='relu',  name='x3_1')(input3)
    net3_2 = Dense(10, activation='relu',  name='x3_2')(net3_1)
    net3_3 = Dense(10, activation='relu', name='x3_3')(net3_2)
    net3_out = Dense(feature_shape3, activation='linear', name='x3_4')(net3_3)

    # feature layer
    shared_layer = concatenate([net1_out, net2_out, net3_out], name='shared_layer')
    
    BN = normalization.BatchNormalization()(shared_layer)
    
    mcca_layer = MCCA(1, 3, name='cca_layer')(BN)

    model = Model(inputs=[input1, input2, input3], outputs=mcca_layer)
    
    model.compile(optimizer='sgd', loss=constant_loss)
    
    model.fit([train_x, train_x, train_x], np.zeros(len(train_x)), batch_size=batch_size, epochs=epoch_num, shuffle=True)
    
    model.save('current_dcca.h5') # 保存模型
    


    
