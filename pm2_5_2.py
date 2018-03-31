# -*- coding=utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import mean_squared_error,r2_score
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv('./data/process_data_pm2_5.csv')
# data = data.sort_index(by='publish_time')
data['publish_time'] = pd.to_datetime(data['publish_time'])
data['hour']=data.publish_time.dt.hour
# print data.head(100)
# exit()
#测试集1000条
original_test_data = data[data['publish_time']>='2015-12-27 00:00:00']
original_test_data = original_test_data[original_test_data['hour'].isin([4,12])]
# print  original_test_data.describe()
#训练集66200
original_train_data = data.merge(original_test_data, indicator=True, how='outer')
original_train_data = original_train_data[original_train_data['_merge'] == 'left_only']

original_train_data.drop(['hour','_merge'], axis=1, inplace=True)
data.drop(['hour'], axis=1, inplace=True)
original_test_data.drop(['hour'], axis=1, inplace=True)

# print original_train_data.describe()
data = data.iloc[:,4:]
train_data = original_train_data.iloc[:,4:]
test_data = original_test_data.iloc[:,4:]
#
# print train_data.head(10)
# print test_data.head(10)
# exit()
# print data.describe()
# exit()
# print df_train.head(10)
# Training data features, skip the first column 'Survived'
# train_features = train_data[:, 1:]  # Fit the model to our training data
# train_features = train_data[:, [7,19,31]]  # Fit the model to our training data
# 'Survived' column values
# train_target = train_data[:, 0]
# print train_target.reshape(train_target.size,1)
# x_train, test_x, y_train, test_y = train_test_split(train_features,
#                                                     train_target,
#                                                     test_size=0.20)



feature = ['PM2_5_1','SO2_1','NO2_1','CO_1','O3_1','PM10_1','wind_power_1',
           'wind_direction_1','temperature_1','relative_humidity_1','precipitation_1',
           'PM2_5_2','SO2_2','NO2_2','CO_2','O3_2','PM10_2','wind_power_2',
           'wind_direction_2','temperature_2','relative_humidity_2','precipitation_2',
           'PM2_5_3','SO2_3','NO2_3','CO_3','O3_3','PM10_3','wind_power_3',
           'wind_direction_3','temperature_3','relative_humidity_3','precipitation_3'
           ] #影响因素11个
label = ['PM2_5'] #标签一个，即需要进行预测的值


#训练数据集
data_mean = train_data.mean()
data_std = train_data.std()
data_train = (train_data - data_mean)/data_std #数据标准化
data_train['PM2_5'] = train_data['PM2_5']

# print data_train
# exit()
x_train = data_train[feature].as_matrix() #特征数据
# print data_train[label]
# exit()
y_train = data_train[label].as_matrix() #标签数据

#测试数据集
test_mean = test_data.mean()
test_std = test_data.std()

# exit(0)

data_test = (test_data - test_mean)/test_std #数据标准化
data_test.fillna(0)
data_test['PM2_5'] = test_data['PM2_5']
# print data_train.head(10)
# print test_data.describe()
# print train_data.describe()
# # print data_test
# exit()


x_test = data_test[feature].as_matrix() #特征数据
# y_test = data_train[label].as_matrix() #标签数据


from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras import regularizers


model = Sequential()  #层次模型
model.add(Dense(10,input_dim=33,kernel_initializer='uniform')) #输入层，Dense表示BP层
model.add(Activation('sigmoid'))  #添加激活函数
# model.add(Dropout(0.2))
model.add(Dense(1,input_dim=10))  #输出层
# model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
model.compile(loss='mean_squared_error', optimizer='rmsprop') #编译模型
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,) #训练模型10次
# print(hist.history)

# model.save_weights(modelfile) #保存模型权重
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

#4 预测，并还原结果。
# x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
original_test_data['PM2_5_pred'] = model.predict(x_test)
# original_test_data[u'PM2_5_pred'] = model.predict(x_test)
# y= model.predict(x_test) * test_std['PM2_5'] + test_mean['PM2_5']
# data[data < 0] = 0
original_test_data[original_test_data['PM2_5_pred']<0]=0

# print original_test_data
# exit()

print "test mse:%f"% mean_squared_error(original_test_data['PM2_5'], original_test_data['PM2_5_pred'])
print "test R2:%f"% r2_score(original_test_data['PM2_5'], original_test_data['PM2_5_pred'])
outputfile = 'output.csv'
# print original_test_data.describe()
# print original_test_data[['PM2_5','PM2_5_pred']].tail(10)
# exit()
#5 导出结果
original_test_data.to_csv(outputfile)

#6 画出预测结果图
import matplotlib.pyplot as plt

original_test_data[['PM2_5','PM2_5_pred']].reset_index(drop=True).plot(style=['b-o','r-*'])
plt.show()