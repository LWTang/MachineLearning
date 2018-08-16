from numpy import *
import os
import operator

# 读取一张图片，转换为1*1024向量
def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

def classify0(inX, dataSet, labels, k):
	'''
	desc:
		KNN的分类函数
	args:
		inX -- 用于分类的输入向量/测试数据
		dataSet -- 训练数据集的向量
		labels -- 训练数据集的labels
		k -- 最近邻数目
	returns:
		预测分类的label
	'''
	dataSetSize = dataSet.shape[0]	# 训练数据向量的个数
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 测试向量与训练集向量之差
	sqDiffMat = diffMat ** 2 #shape 为 1000*1024
	sqDistances = sqDiffMat.sum(axis = 1) #按行加，按列加axis为0
	distances = sqDistances ** 0.5
	ascendIndex = distances.argsort()

	classCount = {}
	for i in range(k):
		votelabel = labels[ascendIndex[i]]
		classCount[votelabel] = classCount.get(votelabel, 0) + 1
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def handwritingClassTest():
	# 导入训练数据
	hwLabels = []
	trainingFileList = os.listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

	# 导入测试数据
	testFileList = os.listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print('return the %s, the real result is %s' % (classifierResult, classNumStr))
		if (classifierResult != classNumStr):errorCount += 1.0
	print('error totally occurs %d times' % errorCount)
	print('the error rate is %f' % (errorCount/float(mTest)))



if __name__ == '__main__':
	handwritingClassTest()