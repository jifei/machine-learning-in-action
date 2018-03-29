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
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 生成与加载数据
# 构造满足一元二次方程的函数
def Build_Data():

        x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
        # 为了使点更密一些,我们构建了300个点,分布在-1到1 的区间,直接曹永np生成等差数列的方式,并将结果为300个点的一维函数,转换为300 * 1 的二维函数
        noise = np.random.normal(0, 0.05, x_data.shape)
        # 加入一些噪声点,使它与x_data的维度一致,并且拟合为均值为0,方差为0.05的正态分布
        y_data = np.square(x_data) - 0.5 + noise
        # y = x^2 - 0.5 + 噪声
        return (x_data,y_data)


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 构建网络模型

def add_layer(inputs, in_size, out_size, activation_function=None):
    # 构建权重 : in_size * out)_sieze 大小的矩阵
    weights = tf.Variable(tf.zeros([in_size, out_size]))
    # 构建偏置 : 1 * out_size 的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 矩阵相乘
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs  # 得到输出数据

neurons_size = 20
# 构建输入层到隐藏层,假设隐藏层有 hidden_layers 个神经元
l1 = add_layer(xs, 1, neurons_size, activation_function=tf.nn.sigmoid)
# 构建隐藏层到隐藏层
# h2 = add_layer(h1, neurons_size, neurons_size, activation_function=tf.nn.sigmoid)
# 构建隐藏层到隐藏层
# h3 = add_layer(h2, neurons_size, neurons_size, activation_function=tf.nn.sigmoid)

# 构建隐藏层到输出层
prediction = add_layer(l1, neurons_size, 1, activation_function=None)

# 接下来构建损失函数: 计算输出层的预测值和真是值间的误差,对于两者差的平方求和,再取平均,得到损失函数.运用梯度下降法,以0.1的效率最小化损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 优化算法选取SGD,随机梯度下降

print('将计算图写入事件文件,在TensorBoard里查看')
writer = tf.summary.FileWriter(logdir='logs/8_2_BP', graph=tf.get_default_graph())
writer.close()

# 训练模型
(x_data,y_data) = Build_Data()
# 我们让TensorFlow训练1000次,每50次输出训练的损失值:
with tf.Session() as sess:
    tf.global_variables_initializer().run()  # 初始化所有变量

    for i in range(10000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 500 == 0:
            print('num %d loss'%i,sess.run(loss, feed_dict={xs: x_data, ys: y_data}))