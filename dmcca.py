
import numpy as np
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Dense, concatenate, normalization
from keras.models import load_model
from keras.callbacks import  EarlyStopping
from mcca_layer import MCCA

from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats


def constant_loss(y_true, y_pred):
    return y_pred

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def diag_power(x, alpha):
    y = np.copy(x)
    for i in range(len(x)):
        y[i,i] = y[i,i] ** alpha
    
    return y

def MyPCA(x, percent):

    U, lamda, Ut = np.linalg.svd(np.dot(x.T, x) / (len(x)-1))

    for i in range(len(x.T)):
        if sum(lamda[0:(i+1)])/sum(lamda) >= percent:
            lamda_pc = np.diag(lamda[0:(i+1)])
            Px_pc = U[:, 0:(i+1)]
            break

    return lamda_pc, Px_pc

def MyCCA(x, y):
    
    sigma_x = diag_power(np.eye(len(x.T)) * (np.dot(x.T, x) / (len(x) - 1)), -0.5)
    sigma_y = diag_power(np.eye(len(y.T)) * (np.dot(y.T, y) / (len(x) - 1)), -0.5)
    sigma_xy = (np.dot(x.T, y) / (len(x) - 1))
    
    K = np.dot(np.dot(sigma_x, sigma_xy), sigma_y)
    
    R, sigma, VT = np.linalg.svd(K)
    
    J = np.dot(sigma_x, R)
    L = np.dot(sigma_y, VT.T)
    sig = np.zeros([len(x.T), len(y.T)])
    sig[0:(len(y.T)), :] = np.diag(sigma)
    
    return J, L, sig

def cal_threshold(x, alpha):

    kernel = stats.gaussian_kde(x)

    step = np.linspace(0,100,10000)
    pdf = kernel(step)
    for i in range(len(step)):

        if sum(pdf[0:(i+1)]) / sum(pdf) > alpha:
            
            break
    return step[i+1]

def calculateR(p, limit):
    count1 = 0; count2 = 0
    for i in range(960):
        if p[i] > limit:
            if i < 160:
                count1 = count1 + 1
            else:
                count2 = count2 + 1
    return count1 / 160, count2 / 800

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
    batch_size = 50

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
    
    #bn = normalization.BatchNormalization()
    
    cca_layer = MCCA(1, name='cca_layer')(shared_layer)

    model = Model(inputs=[input1, input2, input3], outputs=cca_layer)
    #x = model.predict([train_x, train_x, train_x])
    
    model.compile(optimizer='sgd', loss=constant_loss)
    model.fit([train_x, train_x, train_x], np.zeros(len(train_x)), batch_size=batch_size, epochs=epoch_num, shuffle=True,
              validation_split = 0.10, callbacks=[EarlyStopping(monitor='val_loss', patience = 100000)])
    


    
