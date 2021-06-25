# -*- coding: utf-8 -*-
"""
简化版 SMO 算法
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time


def loadDataSet(fileName):
    """
    读取数据

    Parameters:
        fileName - 文件名

    Returns:
        dataMat - 数据矩阵
        labelMat - 数据标签
    """

    # 数据矩阵
    dataMat = []
    # 标签向量
    labelMat = []

    # 打开文件
    fr = open(fileName)

    # 逐行读取
    for line in fr.readlines():
        # 去掉每一行首尾的空白符，例如'\n','\r','\t',' '
        # 将每一行内容根据'\t'符进行切片
        lineArr = line.strip().split('\t')
        
        # 添加数据(100个元素排成一行)
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        
        # 添加标签(100个元素排成一行)
        labelMat.append(float(lineArr[2]))

    return dataMat, labelMat


def selectJrand(i, m):
    """
    随机选择alpha_j

    Parameters:
        i - alpha_i的索引值
        m - alpha参数个数

    Returns:
        j - alpha_j的索引值
    """

    j = i
    
    while j == i:
        # uniform()方法将随机生成一个实数，它在[x, y)范围内
        j = int(random.uniform(0, m))

    return j


def clipAlpha(aj, H, L):
    """
    修剪alpha_j

    Parameters:
        aj - alpha_j值
        H - alpha上限
        L - alpha下限

    Returns:
        aj - alpha_j值
    """

    if aj > H:
        aj = H

    if L > aj:
        aj = L

    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    计算 alpha，b

    Parameters:
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 惩罚参数
        toler - 容错率
        maxIter - 最大迭代次数

    Returns:
        None
    """

    # 转换为numpy的mat矩阵存储(100,2)
    dataMatrix = np.mat(dataMatIn)
    # 转换为numpy的mat矩阵存储并转置(100,1)
    labelMat = np.mat(classLabels).transpose()

    # 初始化b参数，设为0
    b = 0
    # 统计dataMatrix的维度,m:100行；n:2列
    m, n = np.shape(dataMatrix)
    # 初始化alpha参数，设为0
    alphas = np.mat(np.zeros((m, 1)))

    # 初始化迭代次数
    iter_num = 0

    # 最多迭代maxIter次
    while iter_num < maxIter:
        # 初始优化次数
        alphaPairsChanged = 0

        for i in range(m):

            # 步骤1：计算误差Ei
            # multiply(a,b)就是个乘法，如果a,b是两个数组，那么对应元素相乘
            fxi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 误差项计算公式
            Ei = fxi - float(labelMat[i])

            # 优化alpha，设定一定的容错率
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):

                # 随机选择另一个alpha_i成对比优化的alpha_j
                j = selectJrand(i, m)

                # 步骤1，计算误差Ej
                fxj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                # 误差项计算公式
                Ej = fxj - float(labelMat[j])

                # 保存更新前的alpha值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 步骤2：计算上下界H和L
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print("L == H")
                    continue

                # 步骤3：计算eta
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:
                    print("eta>=0")
                    continue

                # 步骤4：更新alpha_j
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta

                # 步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)

                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("alpha_j变化太小")
                    continue

                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                # 步骤7：更新b_1和b_2
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[i, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T

                # 步骤8：根据b_1和b_2更新b
                if (0 < alphas[i] < C):
                    b = b1
                elif (0 < alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                # 统计优化次数
                alphaPairsChanged += 1

                # 打印统计信息
                print("第%d次迭代 样本：%d， alpha优化次数：%d" % (iter_num, i, alphaPairsChanged))

        # 更新迭代次数
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数：%d" % iter_num)

    return b, alphas


def get_w(dataMat, labelMat, alphas):
    """
    计算w

    Returns:
        dataMat - 数据矩阵
        labelMat - 数据标签
        alphas - alphas值

    Returns:
        w - 直线法向量
    """

    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    # 通过labelMat.reshape(1, -1)，Numpy自动计算出有100行，新的数组shape属性为(100, 1)
    # np.tile(labelMat.reshape(1, -1).T, (1, 2))将labelMat扩展为两列(将第1列复制得到第2列)
    # w = sum(alpha_i * yi * xi)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)

    return w.tolist()


def showClassifer(dataMat, w, b):
    """
    分类结果可视化

    Parameters:
        dataMat - 数据矩阵
        w - 直线法向量
        b - 直线截距

    Returns:
        None
    """

    # 正样本
    data_plus = []
    # 负样本
    data_minus = []

    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])

    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)

    # 正样本散点图
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    # 负样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)

    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2

    plt.plot([x1, x2], [y1, y2])

    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        # 支持向量机的点
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')

    plt.show()


if __name__ == '__main__':

    # 开始计时
    time_start = time.time()

    dataMat, labelMat = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)

    # 结束计时
    time_end = time.time()
    print('运行时间：', time_end-time_start)
