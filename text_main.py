# coding=utf-8

import logging.config
import yaml
import random
import os
import sys
import time
import jieba
import shutil
from time import sleep,ctime

from NavieBayes import *
from SplitText import *
from SupportVectorMachines import *
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

file_num = 0
pridictLable = []
trueLable = []

#文件测试线程
def CaculteTestClass(st, nb, testFileLable, outPutPath):
    logger = logging.getLogger(__name__)
    lable = []
    for testFile in testFileLable:
        testContent = st.ReadFile(testFile)
        testContent = testContent.replace("\r\n","").strip()
        testContent_seg = jieba.cut(testContent)
        content = " ".join(testContent_seg)
        content = content.encode('utf-8')
        testVec = st.CreateDataVec(content.split(),'0')
        lable.append(nb.Pridict(testVec[0:-1]))
        #放入对应的文件夹中
        # outPutFileDir = os.path.join(outPutPath , lable)
        # outPutFile = os.path.join(outPutFileDir , os.path.basename(testFile))
        # if os.path.exists(outPutFileDir) == False:
        #     logger.info( 'create dir ' + lable + ' success!')
        #     os.mkdir(outPutFileDir)
        # shutil.copyfile(testFile,outPutFile)
        # logger.info( 'end the ' + testFile + ' file at: ' + ctime())
    pridictLable.append(lable)

def ReadSampleDataFromFile(sampleData_file):
    sampleData = []
    vec = []
    fp = open(sampleData_file,'r')
    lines = fp.readlines()
    for line in lines:
        line = line.replace('\r\n','')
        line = line[0:len(line) - 1] #去掉最后的','
        temp = line.split(',')
        for ele in temp:
            if len(ele.strip()) != 0:
                vec.append(float(ele))
        sampleData.append(vec)
        vec = []
    fp.close()
    return sampleData

def SaveSampleData(fileName,sampleData):
    count = ''
    fp = open(fileName,'w')
    for vec in sampleData:
        for ele in vec:
            count += str(ele) + ','
        count += '\r\n'
        fp.writelines(count)
        count = ''
    fp.close()

#抽取训练集中的70%用于建模，30%用于验证
def ExtrackTrainData(trainDataRootPath,  outVerifyDataPath):
    classPathSet = os.listdir(trainDataRootPath)
    for classPath in classPathSet:
        srcDir = os.path.join(trainDataRootPath, classPath)
        destDir = os.path.join(outVerifyDataPath, classPath)
        if os.path.exists(destDir) == False:
            #print  'create dir ' + destDir + ' success!'
            os.mkdir(destDir)
        fileList = [os.path.join(srcDir, fileName) for fileName in os.listdir(srcDir)]
        #抽取30%的数据到验证集下
        totalNum = len(fileList)
        desFileList = random.sample(fileList, int(totalNum * 0.3))
        for filePath in desFileList:
            fileName = os.path.basename(filePath)
            desPath = os.path.join(destDir, fileName)
            shutil.move(filePath, desPath)

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

def PridictWithBayes(root_path, sampleData, st):
    logger = logging.getLogger(__name__)
    startTime = time.time()
    if len(sampleData) == 0:
        logger.info('sampleData is null !!')
        return
    nb = Bayes(sampleData)
    testFilePath = os.path.join(root_path, 'tarin_corpus_seg', 'verify')
    outPutPath = os.path.join(root_path, 'Data', 'outPut')
    for className in os.listdir(testFilePath):
        startTime = time.time()
        logger.info('start pridict class {0}......'.format(className))
        testFileSet = os.listdir(os.path.join(testFilePath, className))
        testFileLable = [os.path.join(testFilePath, className, fileName) for fileName in testFileSet]
        trueLable.append([int(className)] * len(testFileSet))
        CaculteTestClass(st, nb, testFileLable, outPutPath)
        endTime = time.time()
        logger.info('pridict class {0} end, cost {1}s'.format(className, endTime - startTime))
    SaveSampleData('pridictLable.txt', pridictLable)
    SaveSampleData('trueLable.txt', trueLable)
    endTime = time.time()
    GetResult(trueLable, pridictLable)
    logger.info('cost time {0}s'.format(endTime - startTime))

def PridictWithSVM(root_path, sampleData, st):
    logger = logging.getLogger(__name__)
    svm = MySVM(sampleData)
    startTime1 = time.time()
    if len(sampleData) == 0:
        logger.info('sampleData is null !!')
        return
    testFilePath = os.path.join(root_path, 'tarin_corpus_seg', 'verify')
    lable = []
    for className in os.listdir(testFilePath):
        startTime = time.time()
        logger.info('start pridict class {0}......'.format(className))
        testFileSet = os.listdir(os.path.join(testFilePath, className))
        testFileLable = [os.path.join(testFilePath, className, fileName) for fileName in testFileSet]
        trueLable.append([int(className)] * len(testFileSet))
        for testFile in testFileLable:
            testContent = st.ReadFile(testFile)
            testContent = testContent.replace("\r\n", "").strip()
            testContent_seg = jieba.cut(testContent)
            content = " ".join(testContent_seg)
            content = content.encode('utf-8')
            testVec = st.CreateDataVec(content.split(), '0')
            lable.append(svm.Pridict(testVec[0:-1]))
        endTime = time.time()
        logger.info('pridict class {0} end, cost {1}s'.format(className, endTime - startTime))
        pridictLable.append(lable)
    SaveSampleData('pridictLable.txt', pridictLable)
    SaveSampleData('trueLable.txt', trueLable)
    endTime = time.time()
    GetResult(trueLable, pridictLable)
    logger.info('pridict cost {0}s'.format(endTime - startTime1))

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
    logger.info( 'strat CreateSampleData......')
    sampleData = st.CreateSampleData()
    logger.info( 'CreateSampleData end')
    #开始测试
    #PridictWithBayes(root_path, sampleData, st)
    PridictWithSVM(root_path, sampleData, st)
    pass

def test():
    SetupLogging()
    pL = ReadSampleDataFromFile('pridictLable.txt')
    tL = ReadSampleDataFromFile('trueLable.txt')
    GetResult(tL,pL)

if __name__ == '__main__':
    main()