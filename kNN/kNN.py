import numpy as np
import operator
from os import listdir


#创建数据集及标签 测试用
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) #数据集
    labels = ['A','A','B','B']          #标签
    return group, labels

#k-近邻法
    #inX：用于分类的数据输入
    #dataSet：输入的训练集
    #labels：标签向量
    #k:表示用于选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #返回的是dataset这个array的行数
    #把inX表示为和dataSetSize一样的格式，后相减
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet 
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #行相加
    distances = sqDistances**0.5
    #从小到大排列，提取索引排列
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #添加键值 相同的每次加1
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    
    for line in arrayOlines:
        line = line.strip() #去掉\n
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3] #去前三列转化为对应矩阵
        classLabelVector.append(int(listFromLine[-1])) #最后一列
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1 #抽取%10的数据为测试数据 剩下90%数据为训练样本
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                    datingLabels[numTestVecs:m],3)
        print("the classsifier came back with:%d,the real answer is:%d"\
              %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is:%f" %(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float (input("percentage of time spent playing games:"))
    ffMiles = float (input("frequent flier miles earned per year:"))
    iceCream = float(input("liter of oce cream consumed per year:"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult-1])
    
#手写识别系统
#准备数据：将图像转换为测试向量（图像格式处理为向量）
#将32*32的图像矩阵转换为1*1024的向量

#创建1*1024的Numpy数组，打开给定的文件，循环读出文件的前32行，并将每行的头32
#个字符值存储在numpy数组中，最后返回数组
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i + j] = int(lineStr[j])
    return returnVect
    
#手写数字识别系统测试
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i:] = img2vector('trainingDigits/%s' %fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d, the real answer is:%d"\
               %(classifierResult,classNumStr))
        if(classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total nember of errors is:%d" % errorCount)
    print("\nthe total error rate is: %f" %(errorCount/float(mTest)))
    
#画图
#datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
#plt.show()
    
    
    
    


