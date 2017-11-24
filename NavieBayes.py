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

from numpy import *
import numpy as np
import logging
import time
import sys

#import importlib
#importlib.reload(sys)
reload(sys)
sys.setdefaultencoding('utf-8')

class Bayes:
    #数据集的最后一列为类标签
    def __init__(self,sampleData):
        self.__sample_data__ = sampleData[:,:-1]
        self.__labelvec__ = [ ele[-1] for ele in sampleData]
        self.__classprob__ = self.CacaulteClassProb()
        #生成预测所需的权值矩阵和类标签向量
        self.dataMatrix, self.labelVec = self.CreateWeightMatrix(True)

    def CacaulteClassProb(self):
        logger = logging.getLogger(__name__)
        labeltemp = set(self.__labelvec__)
        labelvecSize = float(len(self.__labelvec__))
        classProb = {} #np.array([self.__labelvec__.count(label)/labelvecSize for label in self.__labelvec__])
        startTime  = time.time()
        for label in labeltemp:
            classProb[label] = np.log(self.__labelvec__.count(label)/labelvecSize)
        endTime = time.time()
        logger.info('Cacaulte Class Prob cost {0}s'.format(endTime - startTime))
        return classProb

    # def GeneratorDataArray(self,sampleData_ele):
    #     data_vec = [0] * (self.__data_dic_len__ + 1)
    #     for index in sampleData_ele:
    #         data_vec[index] = sampleData_ele[index]
    #     data_vec[-1] = sampleData_ele[-1]
    #     return data_vec

    def GeneratorDataCol(self,dic_length):
        logger = logging.getLogger(__name__)
        docNum = len(self.__labelvec__)
        idf = []
        startTime = time.time()
        idf = [log(docNum/(1.0 + docNum - np.sum(ele == 0))) for ele in self.__sample_data__.T]
        endTime = time.time()
        logger.info('generate idf cost {0}s'.format(endTime - startTime))
        return np.array(idf)

    #生成每个类的权值矩阵，即 P(X|C)
    #返回计算过了权值矩阵和矩阵中每行所对应的类标签
    def CreateWeightMatrix(self,tfIDF = False):
        logger = logging.getLogger(__name__)
        logger.info('start create weight matrix....')
        startTime = time.time()
        totalClassWordsNum = array([0.0] * self.__sample_data__.shape[1])
        docNum = len(self.__labelvec__)
        weightMatrix = []
        weightLable = []
        if tfIDF == False:
            return weightMatrix/(1 + totalClassWordsNum)
        else:
            #计算TF
            matrixDic = {}
            for index, className in enumerate(self.__labelvec__):
                if not matrixDic.has_key(className):
                    matrixDic[className] = [0] * self.__sample_data__.shape[1]
                matrixDic[className] += self.__sample_data__[index,:] / float(sum(self.__sample_data__[index,:]) + 1)
            for key in matrixDic.keys():
                weightMatrix.append(matrixDic[key])
                weightLable.append(key)
            endTime = time.time()
            logger.info('Create Weight Matrix cost {0}s '.format(endTime - startTime))
            #计算IDF
            logger.info( 'start caculte IDF....')
            startTime = time.time()
            idf = np.array([log(docNum / (1.0 + docNum - np.sum(ele == 0))) for ele in self.__sample_data__.T])
            endTime = time.time()
            logger.info( 'IDF caculte end cost {0}s'.format(endTime - startTime))
            return np.array(weightMatrix) * log(idf),weightLable

    def Pridict(self,testVec):
        classProb =  self.__classprob__
        test_array = array(testVec,dtype=float)
        vec = np.array([sum(np.log(self.dataMatrix[index,:] * test_array +1))+classProb[lable] for index, lable in enumerate(self.labelVec)])
        maxVal = vec.max()
        index = np.argwhere(vec == maxVal)
        lable = self.labelVec[index[0][0]]
        return lable