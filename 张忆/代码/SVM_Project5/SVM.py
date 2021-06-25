# -*- coding: utf-8 -*-
"""
使用 sklearn 的 SVM 处理手写数字识别问题。
此 SVM 只处理二分类问题，只取 MNIST 数据集中的 1 和 7 作二分类，其余数字不考虑。
"""

import numpy as np
import time
from os import listdir
from sklearn.svm import SVC


def img2vector(filename):
    """
    将32*32的二进制图像转换为1*1024向量

    Parameters:
        filename - 文件名

    Returns:
        returnVect - 返回二进制图像的1*1024向量
    """

    # 创建1*1024零向量
    returnVect = np.zeros((1, 1024))

    # 打开文件
    fr = open(filename)

    # 按行读取
    for i in range(32):
        # 读取一行数据
        lineStr = fr.readline()

        # 每一行的前32个数据依次存储到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect


def handwritingClassTest():
    """
    手写数字分类测试

    Parameters:
        None

    Returns:
        None
    """

    # 训练开始计时
    time_start = time.time()

    # 训练集的Labels
    hwLabels = []

    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')

    # 获取训练集数据个数
    m = len(trainingFileList)

    # 初始化训练集的Mat矩阵（全零阵）
    trainingMat = np.zeros((m, 1024))

    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获取文件名
        fileNameStr = trainingFileList[i]

        # 获取文件名中表示分类的数字
        classNumber = int(fileNameStr.split('_')[0])

        # 将获得的类别添加到hwLabels
        hwLabels.append(classNumber)

        # 将每一个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))

    # 生成SVM模型，指定超参数
    clf = SVC(C=200, kernel='rbf', tol=0.0001, max_iter=10000)

    # 训练模型
    clf.fit(trainingMat, hwLabels)

    # 训练结束计时，测试开始计时
    time_end_1 = time.time()
    print('训练用时:', time_end_1 - time_start)

    # 返回testDigits目录下的文件名
    testFileList = listdir('testDigits')

    # 错误检测计数
    errorCount = 0.0

    # 获取测试集数据个数
    mTest = len(testFileList)

    # 从文件名中解析出测试集的类别，进行分类测试
    for i in range(mTest):
        # 获取文件名
        fileNameStr = testFileList[i]

        # 获取文件名中表示分类的数字
        classNumber = int(fileNameStr.split('_')[0])

        # 将每一个文件的1*1024数据存储到vectorUndertest矩阵中
        vectorUndertest = img2vector('testDigits/%s' % (fileNameStr))

        # 获得预测结果
        classifierResult = clf.predict(vectorUndertest)

        # print("分类结果为%d\t真实结果为%d" % (classifierResult, classNumber))

        if classifierResult != classNumber:
            errorCount += 1.0

    print("误分类个数:%d\n错误率:%f%%" % (errorCount, (errorCount / mTest) * 100))

    # 测试结束计时
    time_end_2 = time.time()
    print('测试用时:', time_end_2 - time_end_1)


if __name__ == '__main__':
    handwritingClassTest()
