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
from time import ctime

import tensorflow as tf
import yaml
import logging.config
from SplitText import *
from NavieBayes import *

def Inference(X,W,b):
    return tf.nn.softmax(CombineInputs(X,W,b))

def CombineInputs(X, W, b):
    return tf.matmul(X, W) + b

def Loss(X,Y,W,b):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = CombineInputs(X,W,b), labels=Y))

def Evaluate(sess, X, Y, W,b):
    predicted = tf.cast(tf.arg_max(Inference(X,W,b), 1), tf.int32)
    return sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))

def Train(totalLoss):
    learingRate = 0.01
    return tf.train.GradientDescentOptimizer(learingRate).minimize(totalLoss)

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

def main():
    SetupLogging()
    logger = logging.getLogger(__name__)
    logger.info( 'start at :{0}'.format(ctime()))
    root_path = os.path.abspath(os.curdir)
    #先生成词袋，样本数据集
    wordset_fileName = os.path.join(root_path , 'wordSet.txt')
    st = SplitText(root_path)
    if os.path.exists(wordset_fileName) == False:
        st.CreateDataSet(wordset_fileName)
    else:
        st.wordsetvec = st.GetWordVec(wordset_fileName)
    #生成计算所需的样本数据集
    sampleData = st.CreateSampleData()
    nb = Bayes(sampleData)
    X = nb.dataMatrix
    Y = nb.labelVec

    #训练模型
    with tf.Session() as sess:
        W = tf.Variable(tf.zeros([X.shape[1], len(Y)], dtype = tf.float64), name='weights')
        b = tf.Variable(tf.zeros([len(Y)], dtype = tf.float64), name='bias')
        tf.initialize_all_variables().run()
        totalLoss = Loss(X,Y,W,b)
        trainOp = Train(totalLoss)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        trainning_steps = 1000
        for step in range(trainning_steps):
            sess.run([trainOp])
            if step%10 == 0:
                print "loss: ", sess.run([totalLoss])
        # 开始测试
        testFilePath = os.path.join(root_path, 'tarin_corpus_seg', 'verify')
        for className in os.listdir(testFilePath):
            logger.info('start create class {0} test data...'.format(className))
            trueLable = []
            testData = []
            startTime =time.time()
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
            correctRate = Evaluate(sess,np.array(testData, dtype=float),trueLable,W,b)
            logger.info('the class {0} correct rate is {1}'.format(className, correctRate))
            logger.info('class {0} pridict end!'.format(className))

        coord.request_stop()
        coord.join(threads)
        sess.close()

    pass

if __name__ == '__main__':
    main()


