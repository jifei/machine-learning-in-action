# -*- coding=utf-8 -*-
import os
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv('./data/process_data_pm2_5.csv')
data['publish_time'] = pd.to_datetime(data['publish_time'])
data['hour']=data.publish_time.dt.hour
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

data = data.iloc[:,4:]
train_data = original_train_data.iloc[:,4:]
test_data = original_test_data.iloc[:,4:]




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
x_train = data_train[feature].as_matrix() #特征数据
y_train = data_train[label].as_matrix() #标签数据

#测试数据集
test_mean = test_data.mean()
test_std = test_data.std()
data_test = (test_data - test_mean)/test_std #数据标准化
data_test.fillna(0)
data_test['PM2_5'] = test_data['PM2_5']
x_test = data_test[feature].as_matrix() #特征数据


from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras import regularizers
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


def create_model(optimizer='rmsprop', init='glorot_uniform',neurons=10):
    model = Sequential()  #层次模型
    model.add(Dense(neurons,input_dim=33,kernel_initializer=init)) #输入层，Dense表示BP层,kernel_initializer='uniform'
    model.add(Activation('sigmoid'))  #添加激活函数
    # model.add(Dropout(0.2))
    model.add(Dense(1,input_dim=neurons))  #输出层
    # model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
    model.compile(loss='mean_squared_error', optimizer=optimizer) #编译模型
    return model

seed = 7
np.random.seed(seed)
estimator = KerasRegressor(build_fn=create_model, verbose=0)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, x_train, y_train, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# exit()

# print model.model.loss
# exit()
# define the grid search parameters
optimizers = ['rmsprop', 'adam']
# optimizers = ['rmsprop']
# init = ['glorot_uniform', 'normal', 'uniform']
init = ['glorot_uniform', 'normal', 'uniform']
# init = ['uniform']
neurons = [15]
epochs = [1500]
batches = [100]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init,neurons=neurons)
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1,verbose=2)
# grid_result = grid.fit(X, Y)
grid_result = grid.fit(x_train, y_train) #训练模型10次
print grid_result.best_estimator_
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# print model.get_weights()
# exit()
#预测，并还原结果。
# x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
original_test_data['PM2_5_pred'] = grid.predict(x_test)
# original_test_data[u'PM2_5_pred'] = model.predict(x_test)
# y= model.predict(x_test) * test_std['PM2_5'] + test_mean['PM2_5']
# data[data < 0] = 0
original_test_data[original_test_data['PM2_5_pred']<0]=0
print "test mse:%f"% mean_squared_error(original_test_data['PM2_5'], original_test_data['PM2_5_pred'])
print "test R2:%f"% r2_score(original_test_data['PM2_5'], original_test_data['PM2_5_pred'])

#画出预测结果图
import matplotlib.pyplot as plt

original_test_data[['PM2_5','PM2_5_pred']].reset_index(drop=True).plot(style=['b-o','r-*'])
plt.show()