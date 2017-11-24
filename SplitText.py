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

import os
import logging
import re
import jieba
from myThread import *
import numpy as np

import sys
#import importlib
#importlib.reload(sys)
reload(sys)
sys.setdefaultencoding('utf-8')

lock = threading.Lock()
threadSeq = 0

class SplitText:
    def __init__(self,rootPath = ''):
        self.__rootpath__ = rootPath
        self.__stopwords__ = ''
        self.__sampleDataDic__= {}
        self.__key_list__ = {}
        self.sampleData = []
        self.wordsetvec = []  #训练样本的词集，不重复
        self.wordset = set()  #用于暂存词集，避免重复
        self.__oripath__ = os.path.join(rootPath , 'Data', 'train')  #未分词语料库路径
        self.__segpath__ = os.path.join(rootPath , 'tarin_corpus_seg' , 'train') #分词后语料库路径


    def ReadFile(self,fileName):
        fp = open(fileName,"rb")
        content = fp.read()
        fp.close()
        return content

    def SaveFile(self,fileName,content):
        fp = open(fileName,"wb")
        fp.write(content)
        fp.close()

    #分词的线程处理函数
    def __splitWordswithjieba__(self,class_dir_path,seg_class_path):
        file_list = os.listdir(class_dir_path)
        for file_path in file_list:
            filename = class_dir_path + file_path
            content = self.ReadFile(filename).strip()
            content = content.replace("\r\n","").strip()
            content_seg = jieba.cut(content)
            self.SaveFile(seg_class_path + file_path," ".join(content_seg))

    def SplitWords(self,inputpath='',outpath=''):
        if inputpath == '':
            inputpath = self.__oripath__
        if outpath == '':
            outpath = self.__segpath__

        catelist = os.listdir(self.__oripath__)
        cateNum = range(len(catelist))
        threads = []

        for dir in catelist:
            ori_class_path = os.path.join(inputpath,dir)
            seg_class_path = os.path.join(outpath , dir )
            if not os.path.exists(seg_class_path):
                os.makedirs(seg_class_path)
            t = MyThread(self.__splitWordswithjieba__,(ori_class_path,seg_class_path),self.__splitWordswithjieba__.__name__)
            threads.append(t)

        for index in cateNum:
            threads[index].start()

        for index in cateNum:
            threads[index].join()

    #filter的线程处理函数,去掉filename中的停用词
    def __dofilder__(self,class_dir_path):
        logger = logging.getLogger(__name__)
        global threadSeq
        threadSeq += 1
        localseq = threadSeq
        logger.info( 'thread ' + str(threadSeq) + 'start')
        file_list = os.listdir(class_dir_path)
        pattern = re.compile(r'^\w+$') #验证任意数字、字母、下划线组成的模式

        for file_path in file_list:
            filename = os.path.join(class_dir_path , file_path)
            content = self.ReadFile(filename)
            content = content.decode('utf-8')
            wordslist = content.split()
            newcontent = ''
            for word in wordslist:
                if self.__stopwords__.find(word.encode('utf-8')) != -1:
                    continue
                elif pattern.match(word.decode('utf-8')) != None:
                    continue
                if lock.acquire():
                    self.wordset.add(word)
                    lock.release()
                newcontent = newcontent + ' ' + word
            self.SaveFile(filename,newcontent)
            logger.info( 'thread ' + str(localseq) + ' end')

    #用中文停用词对分词后的文件做过滤
    def Filter(self,inputpath = ''):
        if inputpath == '':
            inputpath = os.path.join(self.__segpath__, "train")
        #读取停用词
        stop_words_path = os.path.join(self.__rootpath__, "stopwords.txt")
        stop_words = self.ReadFile(stop_words_path)
        stop_words = stop_words.replace("\r\n","").strip()
        self.__stopwords__ = stop_words.decode('utf-8')

        #起线程开始处理已分词的文件
        catelist = os.listdir(inputpath)
        cateNum = range(len(catelist))
        threads = []

        for dir in catelist:
            seg_class_path = os.path.join(inputpath , dir)
            t = MyThread(self.__dofilder__,(seg_class_path,),self.__dofilder__.__name__)
            threads.append(t)

        for index in cateNum:
            threads[index].start()

        for index in cateNum:
            threads[index].join()

    def CreateDataSet(self,wordset_fileName):
        self.SplitWords()
        self.Filter()
        #创建词集,去重
        content = ''
        for word in self.wordset:
            content = content + ' ' + word
        self.wordsetvec = content.split()
        self.SaveFile(wordset_fileName,content)

    #返回包含所有训练样本文本的词向量，不重复
    def GetWordVec(self,filename):
        content = self.ReadFile(filename)
        return content.split()

    #contentList是需要生成特征的文本向量，如：['你'，'我','他','听我的']
    def CreateDataVec(self,contentList,className):
        vec = [0] * len(self.__key_list__)
        for word in contentList:
            try:
                index = self.__key_list__[word]
                vec[index] += 1
            except:
                continue
        vec.append(int(className))
        return np.array(vec)

    def __do_create_datasample__(self,className):
        classPath = os.path.join(self.__segpath__, className )
        classPathSet = os.listdir(classPath)
        for filePath in classPathSet:
            fileName = os.path.join(classPath ,filePath)
            content = self.ReadFile(fileName)
            content = content.strip()
            vec = self.CreateDataVec(content.split(),className)
            self.sampleData.append(vec)

    #生成训练数据集,最后一列为类标签
    def CreateSampleData(self, sampleFileName = ''):
        data_dic = {}
        temp = []
        catelist = os.listdir(self.__segpath__)
        for word in self.wordsetvec:
            data_dic[word] = 0
        keys_list = data_dic.keys()
        num = 0
        #self.__key_list__[word] = val, val为word的索引
        for key in keys_list:
            self.__key_list__.setdefault(key,0)
            self.__key_list__[key] = num
            num += 1
        # 第一行为样本数据的标签，最后一列是类别
        for className in catelist:
            classPath = os.path.join(self.__segpath__, className)
            classPathSet = os.listdir(classPath)
            for filePath in classPathSet:
                fileName = os.path.join(classPath, filePath)
                content = self.ReadFile(fileName)
                content = content.strip()
                vec = self.CreateDataVec(content.split(), className)
                temp.append(vec)
        self.sampleData = np.array(temp)
        #存成cvs文件，文件太大1.4G，写硬盘很慢，而且读出需要22分钟28G内存
        if sampleFileName != '':
            np.savetxt(sampleFileName,   self.sampleData , fmt=['%s'] *   self.sampleData .shape[1], newline='\n', delimiter=',')
        return self.sampleData