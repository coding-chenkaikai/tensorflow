# -*- coding: UTF-8 -*-
"""
    手写数字识别
    算法：BP神经网络
    数据：
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# 构建BP
def bp(x, y):
    hidden1_size = 256  # 第一层hidden中神经元数目
    hidden2_size = 128  #

    # 初始化w和b
    weights = {
        'w1': tf.Variable(tf.random_normal(shape=[input_size, hidden1_size], stddev=0.1)),
        'w2': tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size], stddev=0.1)),
        'out': tf.Variable(tf.random_normal(shape=[hidden2_size, classes_size], stddev=0.1))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal(shape=[hidden1_size], stddev=0.1)),
        'b2': tf.Variable(tf.random_normal(shape=[hidden2_size], stddev=0.1)),
        'out': tf.Variable(tf.random_normal(shape=[classes_size], stddev=0.1))
    }

    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['w2']), biases['b2']))
    actual = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    return actual

if __name__ == '__main__':
    mnist = input_data.read_data_sets('../data', one_hot=True)

    train_images = mnist.train.images
    train_labels = mnist.train.labels

    learn_rate = 0.01

    input_size = train_images.shape[1]
    classes_size = train_labels.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes_size], name='y')

    actual = bp(x, y)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=actual, labels=y))
    train = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)
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

                if train_accuracy > 0.95 and test_accuracy > 0.95:
                    saver.save(session, 'mnist/model')
                    break
            epoch += 1
        writer = tf.summary.FileWriter('mnist/graph', tf.get_default_graph())
        writer.close()
