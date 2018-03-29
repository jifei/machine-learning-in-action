# -*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

ilf = IsolationForest(n_estimators=100,
                      n_jobs=-1,
                      max_features=3,
                      verbose=2,
    )
data = pd.read_csv('./data/water_quality_testing.csv')

#数据分析
print data.describe()
print data.head(10)
data = data.fillna(0)

plt.hist(data['PH'])
plt.title('PH')
plt.xlabel('PH')
plt.ylabel('count')
plt.show()

plt.hist(data['dissolved_oxygen'])
plt.title('dissolved_oxygen')
plt.xlabel('dissolved_oxygen')
plt.ylabel('count')
plt.show()

plt.hist(data['ammonia_nitrogen'])
plt.title('ammonia_nitrogen')
plt.xlabel('ammonia_nitrogen')
plt.ylabel('count')
plt.show()

plt.hist(data['total_organic_carbon'])
plt.title('total_organic_carbon')
plt.xlabel('total_organic_carbon')
plt.ylabel('count')
plt.show()

# 选取特征
X_cols = ["PH", "dissolved_oxygen", "ammonia_nitrogen","total_organic_carbon"]

# 训练
ilf.fit(data[X_cols])
shape = data.shape[0]
batch = 10**6




all_pred = []
for i in range(shape/batch+1):
    start = i * batch
    end = (i+1) * batch
    test = data[X_cols][start:end]
    # 预测
    pred = ilf.predict(test)
    all_pred.extend(pred)

data['pred'] = all_pred

print(data[data['pred']==1].describe())
print(data[data['pred']==-1].describe())

data.to_csv('out_water.csv')

