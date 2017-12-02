#-*-coding:utf-8-*-
#-------------------------------------------------------------------------------
# Name:        
# Purpose:
#
# Author:      BQH
#
# Created:     19/09/2017
# Copyright:   (c) BQH 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import yaml
import logging.config
import os
import time
import random

from SplitText import *
from NavieBayes import *

def SetupLogging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

# 添加层
def AddLayer(inputs, Weights, biases, activation_function=None):
    # add one more layer and return the output of this layer
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def ANN(sampleData,lableVec,root_path, st, hideLayerNum=5000):
    logger = logging.getLogger(__name__)
    Weights = {}
    biases = {}
    #hidden layer
    Weights['l1'] = tf.Variable(tf.random_normal([sampleData.shape[1], hideLayerNum]))
    biases['l1'] = tf.Variable(tf.zeros([1, hideLayerNum]) + 0.1)
    #output layer
    Weights['out'] = tf.Variable(tf.random_normal([hideLayerNum, len(lableVec)]))
    biases['out'] = tf.Variable(tf.zeros([1, len(lableVec)]) + 0.1)
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, sampleData.shape[1]])
    ys = tf.placeholder(tf.int32, [len(lableVec)])

    # 定义神经层：隐藏层和预测层
    # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
    l1 = AddLayer(xs, Weights['l1'], biases['l1'],tf.nn.softmax)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    prediction = AddLayer(l1, Weights['out'], biases['out'], tf.nn.softmax)

    # 定义 loss 表达式
    # the error between prediciton and real data
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels=ys))
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

    # 选择 optimizer 使 loss 达到最小
    # 这一行定义了用什么方式去减少 loss，学习率是 0.1
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    saver = tf.train.Saver()

    # important step 对所有变量进行初始化
    init = tf.initialize_all_variables()
    sess = tf.Session()
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)
    logger.info('start tarinning...')
    startTime = time.time()
    # 迭代 20000 次学习，sess.run optimizer
    for step in range(20000):
        # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
        sess.run(train_step, feed_dict={xs: sampleData, ys: lableVec})
        if step % 1000 == 0:
            saver.save(sess, os.path.join(os.getcwd(), 'my-model'),global_step=step)
            # to see the step improvement
            tempLoss = sess.run(loss, feed_dict={xs: sampleData, ys: lableVec})
            logger.info('loss: {0}'.format(tempLoss))
            if tempLoss < 0.1:
                break
    endTime = time.time()
    logger.info('tarinning cost {0}s'.format(endTime - startTime))

    #开始测试
    testFilePath = os.path.join(root_path, 'tarin_corpus_seg', 'verify')
    for className in os.listdir(testFilePath):
        logger.info('start create class {0} test data...'.format(className))
        trueLable = []
        testData = []
        startTime = time.time()
        testFileSet = os.listdir(os.path.join(testFilePath, className))
        testFileLable = [os.path.join(testFilePath, className, fileName) for fileName in testFileSet]
        trueLable.append([int(className)] * len(testFileSet))
        for testFile in testFileLable:
            testContent = st.ReadFile(testFile)
            testContent = testContent.replace("\r\n", "").strip()
            content = testContent.encode('utf-8')
            testVec = st.CreateDataVec(content.split(), '0')
            testData.append(testVec[0:-1])
        endTime = time.time()
        logger.info('create class {0} data cost {1}s'.format(className, endTime - startTime))

        logger.info('start pridict class {0}...'.format(className))
        predicted = tf.cast(tf.arg_max(sess.run(prediction,feed_dict={xs: np.array(testData, dtype=float)}), 1), tf.int32)
        correctRate = sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, trueLable), tf.float32)))
        logger.info('the class {0} correct rate is {1}%'.format(className, correctRate))

def main():
    SetupLogging()
    logger = logging.getLogger(__name__)
    root_path = os.path.abspath(os.curdir)
    logger.info('start create sample data...')
    # 先生成词袋，样本数据集
    wordset_fileName = os.path.join(root_path, 'wordSet.txt')
    st = SplitText(root_path)
    if os.path.exists(wordset_fileName) == False:
        st.CreateDataSet(wordset_fileName)
    else:
        st.wordsetvec = st.GetWordVec(wordset_fileName)
    # 生成计算所需的样本数据集
    sampleData = st.CreateSampleData()
    # logger.info('start create random data...')
    # # 随机抽取500条数据用于训练
    # randomData = random.sample(sampleData,500)
    nb = Bayes(sampleData)
    inputs = nb.dataMatrix
    lable = nb.labelVec
    logger.info('start run ANN....')
    ANN(inputs,lable,root_path,st)
    pass

if __name__ == '__main__':
    main()