import numpy as np
import tensorflow as tf

def model(x):
  weights = []

  # layer 1
  W_conv = tf.get_variable("conv_01_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9)))
  b_conv = tf.get_variable("conv_01_b", [64], initializer=tf.constant_initializer(0))
  y = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W_conv, strides=[1,1,1,1], padding='SAME'), b_conv))
  weights.append(W_conv)
  weights.append(b_conv)

  # layer 2~19
  for i in range(18):
    W_conv = tf.get_variable("conv_%02d_w" % (i + 2), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
    b_conv = tf.get_variable("conv_%02d_b" % (i + 2), [64], initializer=tf.constant_initializer(0))
    y = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(y, W_conv, strides=[1,1,1,1], padding='SAME'), b_conv))
    weights.append(W_conv)
    weights.append(b_conv)

  # layer 20
  W_conv = tf.get_variable("conv_20_w", [3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
  b_conv = tf.get_variable("conv_20_b", [1], initializer=tf.constant_initializer(0))
  y = tf.nn.bias_add(tf.nn.conv2d(y, W_conv, strides=[1,1,1,1], padding='SAME'), b_conv)
  weights.append(W_conv)
  weights.append(b_conv)

  y = tf.add(y, x)

  return y, weights

