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

import random
import tensorflow as tf
from numpy import linalg as la
import pickle as pkl
import numpy as np
import yaml

import logging.config
import os
import time

from SplitText import *


CKP_Files = '/home/bqh/Code/Text_categorization/checkpoints/'

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

# 利用SVD分解后的sigma和V矩阵对新向量做变换
def ConvertQuery2Eig(query, VT_K, sigma_matrix):
    return np.dot(np.dot(query,VT_K.T),sigma_matrix.I)

# 返回SVD分解后的前k个值
def LoadData():
    data = pkl.load(open('u.data', 'rb'))
    sigma = pkl.load(open('sigma.data', 'rb'))
    VT = pkl.load( open('vt.data', 'rb'))
    lables = pkl.load(open('lable.data','rb'))
    return data[:,0:k], np.matrix(sigma[0:k] * np.eye(k,dtype=int)), VT[0:k,:], lables

def CreateSVDDataSet(dataSample):
    inputs = dataSample[:, :-1]
    labels = dataSample[:,-1]

    # 进行SVD分解
    U, sigma, VT = la.svd(inputs, full_matrices=False)  # full_matrices=True 32G内存不够
    k = 0
    total = np.sum(sigma) * 0.8
    for i in range(1, len(sigma)):
        if np.sum(sigma[0:i]) >= total:
            k = i
            break
    print('k=',k)
    # pkl.dump(U, open('u.data', 'wb'))
    # pkl.dump(sigma, open('sigma.data', 'wb'))
    # pkl.dump(VT, open('vt.data', 'wb'))
    # pkl.dump(labels, open('lable.data', 'wb'))
    return U[:, 0:k], np.matrix(sigma[0:k] * np.eye(k, dtype=int)), VT[0:k, :], labels

def ANN(sample_data,root_path, st, hideLayerNum=500):
    logger = logging.getLogger(__name__)
    sampleData, sigma_matrix, VT_k, lableVec = CreateSVDDataSet(np.array(sample_data))
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
    l1 = AddLayer(xs, Weights['l1'], biases['l1'],tf.nn.relu)
    l1 = tf.nn.dropout(l1,0.1)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    prediction = AddLayer(l1, Weights['out'], biases['out'])

    # 定义 loss 表达式
    # the error between prediciton and real data
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels=ys))
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))


    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.01,global_step,2000,0.98,staircase = True)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    saver = tf.train.Saver()

    # important step 对所有变量进行初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    logger.info('start tarinning...')
    startTime = time.time()
    ckpt = tf.train.get_checkpoint_state(CKP_Files)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # 迭代 20000 次学习，sess.run optimizer
    for step in range(200000):
        sess.run(train_step, feed_dict={xs: sampleData, ys: lableVec})
        if step % 100 == 0:
            saver.save(sess, os.path.join(CKP_Files, 'my-ann-model'),global_step=step)
            # to see the step improvement
            tempLoss = sess.run(loss, feed_dict={xs: sampleData, ys: lableVec})
            logger.info('loss: {0}'.format(tempLoss))
            if tempLoss < 0.05:
                saver.save(sess, os.path.join(CKP_Files, 'my-ann-model'), global_step=step)
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
        testTemp = np.array(testData)
        eigData = np.array([ConvertQuery2Eig(ele, VT_k, sigma_matrix) for ele in testTemp])
        eigData = np.reshape(eigData,(eigData.shape[0], eigData.shape[2]))
        res = tf.nn.softmax(prediction)
        predicted = tf.cast(tf.arg_max(sess.run(res,feed_dict={xs: np.array(eigData, dtype=float)}), 1),tf.int32)
        # predicted = tf.cast(tf.arg_max(sess.run(tf.nn.softmax(prediction),feed_dict={xs: np.array(testData, dtype=float)}), tf.int32))
        correctRate = sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, trueLable), tf.float32)))
        logger.info('the class {0} correct rate is {1}%'.format(className, correctRate * 100))

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

    # 随机抽取500条数据用于训练
    # randomData = random.sample(sampleData,3000)
    ANN(sampleData,root_path,st)
    pass

if __name__ == '__main__':
    main()