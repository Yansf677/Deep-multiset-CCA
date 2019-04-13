
import numpy as np
import pandas as pd
from sklearn import preprocessing

# 训练数据
d00 = np.array(pd.read_csv('./datasets/d00.csv', header=None))
data_scalar = preprocessing.StandardScaler().fit(d00)
d00 = data_scalar.transform(d00)#.astype(np.float32)

# 测试数据
test_data = np.zeros([21, 960, 52])
for i in range(21):
    if i < 9:
        test_data[i] = data_scalar.transform(
            np.array(pd.read_csv('./datasets/d0{}_te.csv'.format(i + 1), header=None)))#.astype(np.float32)
    else:
        test_data[i] = data_scalar.transform(
            np.array(pd.read_csv('./datasets/d{}_te.csv'.format(i + 1), header=None)))#.astype(np.float32)
        
np.save('train_data', d00)
np.save('test_data', test_data)


