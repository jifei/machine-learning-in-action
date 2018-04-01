# -*- coding=utf-8 -*-
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
data1 = pd.read_csv('./output1.csv')
data2 = pd.read_csv('./output2.csv')

print "before mse:%f"% mean_squared_error(data1['PM2_5'], data1['PM2_5_pred'])
print "berore R2:%f"% r2_score(data1['PM2_5'], data1['PM2_5_pred'])


print "after mse:%f"% mean_squared_error(data2['PM2_5'], data2['PM2_5_pred'])
print "after R2:%f"% r2_score(data2['PM2_5'], data2['PM2_5_pred'])

import matplotlib.pyplot as plt
# from pylab import mpl
# from matplotlib.font_manager import _rebuild
# _rebuild()
# #防止中文乱码问题
# mpl.rcParams['font.sans-serif']=[u'SimHei']
# mpl.rcParams['axes.unicode_minus']=False

# data1[['PM2_5','PM2_5_pred']].reset_index(drop=True).plot(style=['b-o','r-*'])
# # plt.title('Before',color=u"优化前")
# plt.show()
# # exit()
# data2[['PM2_5','PM2_5_pred']].reset_index(drop=True).plot(style=['b-o','r-*'])
# # plt.title('After',color=u"优化后")
# plt.show()

#
# print data1

data2['PM2_5_pred2'] =data2['PM2_5_pred']
data2 = data2.drop(["PM2_5_pred"], axis=1)
data1 = data1[['site_code','publish_time','PM2_5','PM2_5_pred']]
data2 = data2[['site_code','publish_time','PM2_5','PM2_5_pred2']]
print data1.head(10)
print data2.head(10)
# data = pd.concat([data1,data2],keys=['site_code','publish_time'])
data = pd.merge(data1,data2,on=['site_code','publish_time'],how='left')

data['before']=data['PM2_5_pred']- data['PM2_5_x']
data['after']=data['PM2_5_pred2']- data['PM2_5_x']
data[['before','after']].reset_index(drop=True).plot(style=['b-o','r-*'])
plt.show()