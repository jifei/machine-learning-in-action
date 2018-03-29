import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('./data/water_quality_testing.csv')
print df.shape
df = df.drop(["site_code","monitor_time"], axis=1)
# print df.head(10)
# print df.values[:, :3]
# Prepare training data
datasize = 26226
train_X = df.values[:, :3]
# print train_X[:,0]
# exit()
# print train_X.shape
train_Y =  df.values[:, 3]
# print train_Y
# exit()

# Define the model  
X1 = tf.placeholder(tf.float32, shape=(datasize,))
X2 = tf.placeholder(tf.float32, shape=(datasize,))
X3 = tf.placeholder(tf.float32, shape=(datasize,))
Y = tf.placeholder(tf.float32, shape=(datasize,))
w1 = tf.Variable(0.0, name="weight1")
w2 = tf.Variable(0.0, name="weight2")
w3 = tf.Variable(0.0, name="weight3")
b = tf.Variable(tf.zeros([1], dtype=np.float32), name="bias")

y1 = tf.add(w1 * X1 + w2 * X2 + w3 * X3,b)
loss = tf.reduce_mean(tf.square(Y - y1))
# use adam method to optimize
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# Create session to run  
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(5000):
        _, ww1, ww2, ww3, loss_ = sess.run([train, w1, w2, w3, loss],
                                                feed_dict={X1: train_X[:,0], X2: train_X[:,1], X3: train_X[:,2],
                                                            Y: train_Y})

plt.plot(train_X[:,0], train_Y, "+", label='data')
plt.plot(train_X[:,0], ww1 * train_X[:,0] + (ww2) * (train_X[:,1]) + ww3 * (train_X[:,2]), label='curve')
plt.savefig('1.png', dpi=200)
plt.axis([0, np.pi, -2, 2])
plt.legend(loc=1)
plt.show()  