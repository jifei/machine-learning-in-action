# -*- coding=utf-8 -*-
from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
# X = [[1,1,1],[1,1,2],[1,2,1]]
# y = [[6],[9],[8]]
# print(type(X))
# exit()

df=pd.read_csv('./data/water_quality_testing.csv')
df = df.drop(["site_code","monitor_time"], axis=1)
print(df.head(10))
print(df.describe())

sns.pairplot(df, x_vars=['PH','dissolved_oxygen','ammonia_nitrogen'], y_vars='total_organic_carbon', size=5, aspect=0.8)
# plt.show()
# exit()
# print(df["PH","dissolved_oxygen","ammonia_nitrogen"])
# exit(0)
data_X = df.values[:, :3]
data_y = df.values[:, 3]

model = LinearRegression()
model.fit(data_X, data_y)
from sklearn.metrics import mean_squared_error
y_pre = model.predict(data_X)

print(mean_squared_error(y_pre,data_y))

# print(data_X)

# print(model.predict(data_X[:4, :]))
# print(data_y[:4])
# n_samples=100表示100个样本点，n_features=1表示1个特征
# n_targets=1表示1个目标，noise=10噪音等级为10
# X, y = datasets.make_regression(n_samples=26226, n_features=3, n_targets=1, noise=10)
# # scatter表示用点描述
# plt.scatter(X, y)
# plt.show()