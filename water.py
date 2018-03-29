import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Prepare training data  
datasize = 100
train_X = np.linspace(0, np.pi, datasize)
train_Y = np.sin(train_X)
# Define the model
X1 = tf.placeholder(tf.float32, shape=(datasize,))
X2 = tf.placeholder(tf.float32, shape=(datasize,))
X3 = tf.placeholder(tf.float32, shape=(datasize,))
X4 = tf.placeholder(tf.float32, shape=(datasize,))
Y = tf.placeholder(tf.float32, shape=(datasize,))
w1 = tf.Variable(0.0, name="weight1")
w2 = tf.Variable(0.0, name="weight2")
w3 = tf.Variable(0.0, name="weight3")
w4 = tf.Variable(0.0, name="weight4")



y1 = w1 * X1 + w2 * X2 + w3 * X3 + w4 * X4
loss = tf.reduce_mean(tf.square(Y - y1))
# use adam method to optimize
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# Create session to run  
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(5000):
        _, ww1, ww2, ww3, ww4, loss_ = sess.run([train, w1, w2, w3, w4, loss],
                                                feed_dict={X1: train_X, X2: train_X ** 3, X3: train_X ** 5,
                                                           X4: train_X ** 7, Y: train_Y})

plt.plot(train_X, train_Y, "+", label='data')
plt.plot(train_X, ww1 * train_X + (ww2) * (train_X ** 3) + ww3 * (train_X ** 5) + ww4 * (train_X ** 7), label='curve')
plt.savefig('1.png', dpi=200)
plt.axis([0, np.pi, -2, 2])
plt.legend(loc=1)
plt.show()  