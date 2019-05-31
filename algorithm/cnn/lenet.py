# -*- coding: UTF-8 -*-
"""
    手写数字识别
    算法：lenet
    数据：
"""

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def get_variable(name, shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    return tf.get_variable(name, shape, dtype, initializer)

# 构建LeNet
def le_net(x, y):
    # 输入层
    with tf.variable_scope('input'):
        # [None, input_size] -> [None, height, weight, channels]
        net = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 卷积层
    with tf.variable_scope('conv2'):
        net = tf.nn.conv2d(input=net, filter=get_variable('w', [5, 5, 1, 20]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [20]))
        net = tf.nn.relu(net)

    # 池化层
    with tf.variable_scope('pool3'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 卷积层
    with tf.variable_scope('conv4'):
        net = tf.nn.conv2d(input=net, filter=get_variable('w', [5, 5, 20, 50]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [50]))
        net = tf.nn.relu(net)

    # 池化层
    with tf.variable_scope('pool5'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层
    with tf.variable_scope('fc6'):
        net = tf.reshape(net, shape=[-1, 7 * 7 * 50])
        net = tf.add(tf.matmul(net, get_variable('w', [7 * 7 * 50, 500])), get_variable('b', [500]))
        net = tf.nn.relu(net)

    # 全连接层
    with tf.variable_scope('fc7'):
        net = tf.add(tf.matmul(net, get_variable('w', [500, classes_size])), get_variable('b', [classes_size]))
        actual = tf.nn.softmax(net)
    return actual

if __name__ == '__main__':
    mnist = input_data.read_data_sets('../data', one_hot=True)

    train_images = mnist.train.images
    train_labels = mnist.train.labels

    learn_rate = 1e-2

    input_size = train_images.shape[1]
    classes_size = train_labels.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes_size], name='y')

    actual = le_net(x, y)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=actual, labels=y))
    train = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
    predict = tf.equal(tf.argmax(actual, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

    init = tf.global_variables_initializer()
    batch_size = 100
    display_step = 4
    with tf.Session() as session:
        session.run(init)

        # 模型保存
        saver = tf.train.Saver()
        epoch = 0
        while True:
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                feeds = {x: batch_xs, y: batch_ys}
                session.run(train, feed_dict=feeds)
                avg_cost += session.run(loss, feed_dict=feeds)
            avg_cost = avg_cost / total_batch

            if (epoch + 1) % display_step == 0:
                print('次数：%03d 损失函数值：%.9f' % (epoch, avg_cost))
                feeds = {x: mnist.train.images, y: mnist.train.labels}
                train_accuracy = session.run(accuracy, feed_dict=feeds)
                print('训练集准确率：%.3f' % train_accuracy)

                feeds = {x: mnist.test.images, y: mnist.test.labels}
                test_accuracy = session.run(accuracy, feed_dict=feeds)
                print('测试集准确率：%.3f' % test_accuracy)

                if train_accuracy > 0.90 and test_accuracy > 0.90:
                    saver.save(session, 'mnist/model')
                    break
            epoch += 1
        writer = tf.summary.FileWriter('mnist/graph', tf.get_default_graph())
        writer.close()

