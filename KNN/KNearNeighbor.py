# -*- coding: utf-8 -*-
"""
近邻分类算法概念：
在近邻分类算法中，对于预测的数据，将其与训练样本进行比较，找到最为相似的K个训练样本，
并以这K个训练样本中出现最多的标签作为最终的预测标签。
在近邻分类算法中，最主要的是K-近邻算法。
概述：
K-NN算法是最简单的分类算法，主要的思想是计算待分类样本与训练样本之间的差异性，
并将差异按照由小到大排序，选出前面K个差异最小的类别，并统计在K个中类别出现次数最多的类别为最相似的类，
最终将待分类样本分到最相似的训练样本的类中。与投票(Vote)的机制类似。
欧式距离：
P=(x1,x2,...,xn)
Q=(y1,y2,...,yn)
D=sqrt[SUM(xi-yi)^2]
"""
'''
Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label
'''
from numpy import *
import operator
from os import listdir

#输入参数：待分类向量、样本集、标签、k
def classify0(inX, dataSet, labels, k):
    #计算待分类样本与样本集中各个样本距离
    dataSetSize = dataSet.shape[0]   # 样本集中样本个数
    """
    title(inX,[rowNum,colNum]):把inX赋值colNum列,rowNum行
    例如:title([1,3,5],[2,3])
    [
    [1,3,5],[1,3,5],[1,3,5]
    [1,3,5],[1,3,5],[1,3,5]
    ]
    """
    diffMat = tile(inX, (dataSetSize,1)) - dataSet   # 待分类向量构成 样本集同形式（方便计算），作差
    # 两向量间距离（平方和开根号）
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]   # 返回排名第一的类别

# 创建简单的数据集和标签：
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 归一化特征值（在处理不同取值范围的特征值时，通常要归一化，以防止某个特征值过大造成距离计算完全取决于该特征值）
# 公式：newValue=（Value-min）/(max-min)，max和min为特征最大和最小值，该公式将特征值转化到0~1区间。
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))