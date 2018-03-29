
# -*- coding=utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn import linear_model
from sklearn import preprocessing

# Read x and y
x_data = np.loadtxt("ex3x.dat").astype(np.float32)
y_data = np.loadtxt("ex3y.dat").astype(np.float32)


# We evaluate the x and y by sklearn to get a sense of the coefficients.
reg = linear_model.LinearRegression()
reg.fit(x_data, y_data)
print "Coefficients of sklearn: K=%s, b=%f" % (reg.coef_, reg.intercept_)


# Now we use tensorflow to get similar results.

# Before we put the x_data into tensorflow, we need to standardize it
# in order to achieve better performance in gradient descent;
# If not standardized, the convergency speed could not be tolearated.
# Reason:  If a feature has a variance that is orders of magnitude larger than others,
# it might dominate the objective function
# and make the estimator unable to learn from other features correctly as expected.
scaler = preprocessing.StandardScaler().fit(x_data)
print scaler.mean_, scaler.scale_
x_data_standard = scaler.transform(x_data)


W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1, 1]))
y = tf.matmul(x_data_standard, W) + b

loss = tf.reduce_mean(tf.square(y - y_data.reshape(-1, 1)))/2
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)
for step in range(100):
    sess.run(train)
    if step % 10 == 0:
        print step, sess.run(W).flatten(), sess.run(b).flatten()

print "Coefficients of tensorflow (input should be standardized): K=%s, b=%s" % (sess.run(W).flatten(), sess.run(b).flatten())
print "Coefficients of tensorflow (raw input): K=%s, b=%s" % (sess.run(W).flatten() / scaler.scale_, sess.run(b).flatten() - np.dot(scaler.mean_ / scaler.scale_, sess.run(W)))