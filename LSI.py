#-*-coding:utf-8-*-
#-------------------------------------------------------------------------------
# Name:        利用SVD做奇异值分解，然后分类
# Purpose:
#
# Author:      BQH
#
# Created:     19/09/2017
# Copyright:   (c) BQH 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import logging.config
import yaml
import numpy as np
import pickle as pkl
from numpy import linalg as la
from SplitText import *

k = 0 #SVD后取前k个
# sample_data = [] #分解后的特征值矩阵
# VT_k = []
# sigma_matrix = []

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

# 计算两条曲线的相似性,相关系数
def GetPearsonDis(xa, ya):
    dis = np.corrcoef(xa, ya) * 0.5 + 0.5  # 由[-1,1]映射到[0,1]
    return dis[0, 1]

def GetDistance(xa,ya):
    x = np.array(xa)[0]
    y = np.array(ya)
    # print('x:',x.shape)
    # print('y:',y.shape)
    # print(np.dot(x,y.T))
    # print(np.sqrt(x.dot(x)))
    # print(np.sqrt(y.dot(y)))
    return np.dot(x,y.T) / np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))

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
    global k
    inputs = dataSample[:, :-1]
    labels = dataSample[:,-1]
    print(inputs.shape)
    # 进行SVD分解
    U, sigma, VT = la.svd(inputs, full_matrices=False)  # full_matrices=True 32G内存不够
    total = np.sum(sigma) * 0.9
    for i in range(1, len(sigma)):
        if np.sum(sigma[0:i]) >= total:
            k = i
            break
    print('k=', k)
    # pkl.dump(U, open('u.data', 'wb'))
    # pkl.dump(sigma, open('sigma.data', 'wb'))
    # pkl.dump(VT, open('vt.data', 'wb'))
    # pkl.dump(labels, open('lable.data', 'wb'))
    return U[:, 0:k], np.matrix(sigma[0:k] * np.eye(k, dtype=int)), VT[0:k, :], labels

def GetResult(trueLable, pridictLable):
    logger = logging.getLogger(__name__)
    correctRate=[0] * len(trueLable) #分别计算每个类的正确率
    correctTotal = 0
    # 计算正确率
    totalNum = 0
    for index in range(len(pridictLable)):
        temp = np.array(pridictLable[index]) - np.array(trueLable[index])
        correctRate[index] = np.sum(temp == 0) * 1.0/len(pridictLable[index]) * 100
        correctTotal += np.sum(temp == 0)
        totalNum += len(pridictLable[index])
    for index,rate in enumerate(correctRate):
        logger.info('the class {0} rate is {1}%'.format(index, rate))
    logger.info('the total correct rate is {0}%'.format(correctTotal * 1.0 / totalNum * 100))

def PridictWithSVD(root_path, sampleData, st):
    # global sample_data, sigma_matrix, VT_k
    logger = logging.getLogger(__name__)
    logger.info('start SVD...')
    sample_data, sigma_matrix, VT_k, lable = CreateSVDDataSet(np.array(sampleData))
    # 开始测试
    testFilePath = os.path.join(root_path, 'tarin_corpus_seg', 'verify')
    for className in os.listdir(testFilePath):
        trueLable = []
        testData = []
        testFileSet = os.listdir(os.path.join(testFilePath, className))
        testFileLable = [os.path.join(testFilePath, className, fileName) for fileName in testFileSet]
        trueLable.append([int(className)] * len(testFileSet))
        for testFile in testFileLable:
            testContent = st.ReadFile(testFile)
            testContent = testContent.replace("\r\n", "").strip()
            content = testContent.encode('utf-8')
            testVec = st.CreateDataVec(content.split(), '0')
            testData.append(testVec[0:-1])

        logger.info('start pridict class {0}...'.format(className))
        testTemp = np.array(testData)

        predicted = []
        eigData = [ConvertQuery2Eig(ele, VT_k, sigma_matrix) for ele in testTemp]
        for eig in eigData:
            testLable = lable[np.argmax([GetDistance(eig,vec) for vec in sample_data])]
            predicted.append(testLable)

        temp = np.array(predicted) - np.array(trueLable)
        correctRate = np.sum(temp == 0) * 1.0 / len(predicted) * 100
        logger.info('class {0} correct rate is {1}%'.format(className,correctRate))


def main():
    SetupLogging()
    global sample_data, VT_k, sigma_matrix
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
    # 生成计算所需的样本数据集,tf-idf权值矩阵
    sampleData = st.CreateSampleData()
    randomData = random.sample(sampleData, 3000)
    # CreateSVDDataSet(sampleData)
    PridictWithSVD(root_path,randomData,st)
    pass

if __name__ == '__main__':
    main()


