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

import logging
import time
import os
import multiprocessing
from gensim.models import Word2Vec
from SplitText import *
import matplotlib.pyplot as plt
from gensim.models.word2vec import LineSentence

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class MyWord2Vector:
    def __init__(self, root_path):
        self.rootPath = root_path
        self.__segpath__ = os.path.join(self.rootPath, 'tarin_corpus_seg', 'train')  # 分词后语料库路径
        self.__wvpath__ = os.path.join(self.rootPath, 'tarin_corpus_seg', 'word2vec')  # 词向量模型的输出路径

    def ReadFile(self,fileName):
        fp = open(fileName,"rb")
        content = fp.read()
        fp.close()
        return content

    def SaveFile(self,fileName,content):
        fp = open(fileName,"wb")
        fp.write(content)
        fp.close()

    def CreateWordVecModel(self):
        logger = logging.getLogger(__name__)
        classPathSet = os.listdir(self.__segpath__)
        logger.info('start create word vec....')
        for className in classPathSet:
            classDir = os.path.join(self.__segpath__, className)
            content = ''
            for fileName in os.listdir(classDir):
                filePath = os.path.join(classDir, fileName)
                content += self.ReadFile(filePath)
                # content = content.decode('utf-8')
                # wordslist = content.split()
                # wordData.extend(wordslist)
            tempFile = os.path.join(self.__wvpath__, className + '.wordset')
            self.SaveFile(tempFile,content)
            # 训练一个语言模型
            startTime = time.time()
            wordModel = Word2Vec(LineSentence(tempFile), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
            modelName = os.path.join(self.__wvpath__, className + '.model')
            vecName = os.path.join(self.__wvpath__, className + '.txt')
            wordModel.save(modelName)
            wordModel.wv.save_word2vec_format(vecName, binary=False)
            endTime = time.time()
            logger.info('train class {0} with word vector cost {1}s'.format(className, endTime - startTime))
        logger.info('create word vec end!')

    def LoadWordModel(self, wvModelPath = ''):
        wvModelDic = {}
        if wvModelPath == '':
            wvModelPath = os.path.join(self.rootPath, 'tarin_corpus_seg', 'word2vec')
        # 加载模型
        for modelName in os.listdir(wvModelPath):
            if modelName.find('.model') > 0:
                wvModelFile = os.path.join(wvModelPath, modelName)
                wvModel = Word2Vec.load(wvModelFile)
                className = os.path.splitext(modelName)[0]
                wvModelDic[className] = wvModel
        return wvModelDic

    #将训练好的语言模型转换成矩阵格式，最后一列为类标签
    def CreateWordModelDataMatrix(self, wvModelDic):
        dataMatrix = []
        for wvModelName in wvModelDic.keys():
            for key in wvModelDic[wvModelName].wv.vocab.keys():
                temp = list(wvModelDic[wvModelName].wv[key])
                temp.extend(wvModelName)
                dataMatrix.append(temp)
        return np.array(dataMatrix)

    def CreateFileWordVecModel(self, fileName,):
        # testContent = self.ReadFile(fileName)
        dataMatrix = []
        # testContent = testContent.replace("\r\n", "").strip()
        # testContent_seg = jieba.cut(testContent)
        # content = " ".join(testContent_seg)
        #contentList = testContent.encode('utf-8').split()
        wordModel = Word2Vec(LineSentence(fileName), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
        for key in wordModel.wv.vocab.keys():
            dataMatrix.append(wordModel.wv[key])
        return np.sum(dataMatrix)

    # testMatrix: 文章的词向量矩阵
    # 返回testMatrix到模型中每个词的余弦相似度的最大值
    def PridictWithCov(self, testMatrix, wvModel):
        score = []
        for testVec in testMatrix:
            for key in wvModel.wv.vocab:
                score.append(self.GetPearsonDis(testVec, wvModel.wv[key]))
        return np.max(score)

    # 计算两条曲线的相似性,相关系数
    def GetPearsonDis(self, xa, ya):
        dis = np.corrcoef([xa, ya]) * 0.5 + 0.5  # 由[-1,1]映射到[0,1]
        return dis[0, 1]