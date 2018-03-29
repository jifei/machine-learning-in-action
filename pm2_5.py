# -*- coding=utf-8 -*-
# import tensorflow as tf
# import numpy as np
#
# def addLayer(inputData, inSize, outSize, activity_function=None):
#     Weights = tf.Variable(tf.random_normal([inSize, outSize]))
#     basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
#     weights_plus_b = tf.matmul(inputData, Weights) + basis
#     if activity_function is None:
#         ans = weights_plus_b
#     else:
#         ans = activity_function(weights_plus_b)
#     return ans
#
#
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 转为列向量
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) + 0.5 + noise
#
# xs = tf.placeholder(tf.float32, [None, 1])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
# ys = tf.placeholder(tf.float32, [None, 1])
#
# l1 = addLayer(xs, 1, 10, activity_function=tf.nn.relu)  # relu是激励函数的一种
# l2 = addLayer(l1, 10, 1, activity_function=None)
# loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l2)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
#
# train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 选择梯度下降法
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(10000):
#     sess.run(train, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         print sess.run(loss, feed_dict={xs: x_data, ys: y_data})

import tensorflow as tf
import os
import numpy as np
import pandas as pd
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv('./data/process_data_pm2_5.csv')
data = data.iloc[:,4:]
train_data = data.values
print data.head(10)
# print data.describe()
# exit()
# print df_train.head(10)
# Training data features, skip the first column 'Survived'
train_features = train_data[:, 1:]  # Fit the model to our training data
# train_features = train_data[:, [7,19,31]]  # Fit the model to our training data
# 'Survived' column values
train_target = train_data[:, 0]
# print train_target.reshape(train_target.size,1)
train_x, test_x, train_y, test_y = train_test_split(train_features,
                                                    train_target,
                                                    test_size=0.20)
# train_x = tf.nn.l2_normalize(train_x, axis = 0)
# print train_x
# exit()

train_y = train_y.reshape(train_y.size,1)
# print train_y
# print train_x.shape
# print train_y.shape
# exit()

xs = tf.placeholder(tf.float32, [None, 33])
ys = tf.placeholder(tf.float32, [None, 1])

# 构建网络模型

def add_layer(inputs, in_size, out_size, activation_function=None):
    # 构建权重 : in_size * out)_sieze 大小的矩阵
    weights = tf.Variable(tf.zeros([in_size, out_size]))
    # 构建偏置 : 1 * out_size 的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # print "111111111111111111111"
    # print inputs.shape
    # print weights.shape
    # print biases.shape
    inputs = tf.nn.l2_normalize(inputs, axis=0)

    # 矩阵相乘
    Wx_plus_b = tf.matmul(inputs, weights)+biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs  # 得到输出数据

neurons_size = 100
# 构建输入层到隐藏层,假设隐藏层有 hidden_layers 个神经元
h1 = add_layer(xs, 33, neurons_size, activation_function=tf.nn.sigmoid)
# 构建隐藏层到隐藏层
h2 = add_layer(h1, neurons_size, neurons_size, activation_function=tf.nn.relu)
#构建隐藏层到隐藏层
h3 = add_layer(h2, neurons_size, neurons_size, activation_function=tf.nn.sigmoid)

# 构建隐藏层到输出层
prediction = add_layer(h3, neurons_size, 1, activation_function=None)

# print train_y
# exit(0)

# 接下来构建损失函数: 计算输出层的预测值和真是值间的误差,对于两者差的平方求和,再取平均,得到损失函数.运用梯度下降法,以0.1的效率最小化损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 优化算法选取SGD,随机梯度下降

print('将计算图写入事件文件,在TensorBoard里查看')
writer = tf.summary.FileWriter(logdir='logs/8_2_BP', graph=tf.get_default_graph())
writer.close()

# 训练模型
# 我们让TensorFlow训练1000次,每500次输出训练的损失值:
with tf.Session() as sess:
    tf.global_variables_initializer().run()  # 初始化所有变量

    for i in range(10000):
        sess.run(train_step, feed_dict={xs: train_x, ys: train_y})
        if i % 5 == 0:
            print('num %d loss'%i,sess.run(loss, feed_dict={xs: train_x, ys: train_y}))