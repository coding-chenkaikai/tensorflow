# -*- coding: UTF-8 -*-
"""
    手写数字识别
    算法：LSTM单层单向
    数据：
"""

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 数据加载，每个样本784维度
mnist = input_data.read_data_sets('../data', one_hot=True)

with tf.Session() as session:
    lr = 0.001
    # 每个时刻输入的数据维度大小
    input_size = 28
    # 时刻数目，总共输入多少次
    timestep_size = 28
    # 神经元数目
    hidden_size = 128
    # 隐藏层数目
    layer_num = 2
    # 输出类别数目
    class_num = 10

    _x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, class_num])

    batch_size = tf.placeholder(tf.int32, [])
    # dropout保存率
    keep_prob = tf.placeholder(tf.float32, [])

    # 构建网络
    # 输入格式转换
    # x格式：[batch_size, time_steps, input_size]
    x = tf.reshape(_x, shape=[-1, timestep_size, input_size])
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, reuse=tf.get_variable_scope().reuse)
    init_state = lstm_cell.zero_state(batch_size, tf.float32)

    # 输出格式
    # outputs格式：[batch_size, time_steps, input_size]
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=x, initial_state=init_state)
    output = outputs[:, -1, :]

    w = tf.Variable(tf.truncated_normal([hidden_size, class_num], mean=0.0, stddev=0.1), dtype=tf.float32, name='w')
    b = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32, name='b')
    y_predict = tf.nn.softmax(tf.add(tf.matmul(output, w), b))

    loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_predict), 1))
    # train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    cp = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(cp, 'float'))

    session.run(tf.global_variables_initializer())
    for i in range(1000):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)
        # 训练模型
        session.run(train, feed_dict={_x: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})
        if (i + 1) % 200 == 0:
            train_acc = session.run(accuracy, feed_dict={_x: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
            print('批次：{} 步骤：{} 训练集准确率：{}'.format(mnist.train.epochs_completed, (i + 1), train_acc))

        test_acc = session.run(accuracy, feed_dict={_x: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
        print('测试集准确率：{}'.format(test_acc))

