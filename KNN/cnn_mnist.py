# coding:utf-8
# 使用卷积神经网络来识别mnist数据集中的图片，

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 数据转换成图片
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# 初始化全局变量
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './MNIST_data/', 'Directory for storing data')

print(FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

# 第一层卷积和池化
# 卷积核为 5*5， 输入通道为 1，输出通道为 32
# 卷积前图像的尺寸为 [1, 28, 28, 1] 卷积后的图像尺寸为 [1, 28, 28, 32]
# 池化后的图像尺寸为 [1, 14, 14, 32]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)   # [1, 28, 28, 32]
h_pool1 = max_pool_2x2(h_conv1)                         # [1, 14, 14, 32]


# 第二层卷积和池化
# 卷积核为 5*5， 输入通道为 32，输出通道为 64
# 卷积前图像的尺寸为 [1, 14, 14, 32] 卷积后的图像尺寸为 [1, 14, 14, 64]
# 池化后的图像尺寸为 [1, 7, 7, 64]
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层是全链接层，输入图像维度是 7*7*64， 输出维度是 1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层，输入1024维，输出10维，也就是具体的0~9分类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 使用softmax作为多分类激活函数


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 使用adam优化
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    #print(batch_xs)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})


    #result = sess.run(h_pool2_flat, feed_dict={x: batch_xs})
    #print(np.shape(result))
    #print(result[0])

test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print("test accuracy %g" % test_accuracy)




