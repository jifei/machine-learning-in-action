# -*- coding=utf-8 -*-
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

#原始数据
original_data = pd.read_csv('./data/pm2.5.csv')

#数据分析
print original_data.describe()
print original_data.groupby("site_code").size()
original_data['publish_time'] = pd.to_datetime(original_data['publish_time'])
print original_data.head(10)

#处理后的数据
data = original_data[original_data['publish_time']>='2015-12-04']
for num in range(1,4):
    tmp_data = original_data
    tmp_data['publish_time'] =original_data['publish_time']+datetime.timedelta(days=1)
    data = pd.merge(data,tmp_data[['site_code','publish_time','AQI','SO2','NO2','CO','O3','PM10','PM2_5','wind_power','wind_direction','temperature','relative_humidity','precipitation']],
                    on=['site_code', 'publish_time'], how='left',suffixes=['', '_'+str(num)])
    #
# data = data.fillna(0)
data.drop(['AQI','SO2','NO2','CO','O3','PM10','wind_power','wind_direction','temperature','relative_humidity','precipitation'], axis=1, inplace=True)
data.to_csv('./data/process_data_pm2_5.csv',index=False)
