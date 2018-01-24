#!/usr/bin/env python
# -*- coding:utf-8 -*-

#########################################
# kNN: k Nearest Neighbors
#             data: labels+sets
#             k:    number of neighbors to use for comparison

#########################################
from numpy import *
import operator
import numpy as np


class Knn(object):
    data = None  # 数据变量

    # 初始化方法，这里不需要用
    def __init__(self):
        super(Knn, self).__init__()

    # classify using kNN
    def kNNClassify(self, train, test, k, isEuclidian):
        rightCount = 0;
        labels_tr = [int(i) for i in train[:, 0]]
        dataSet_tr = train[:, 1:]
        labels_te = [int(i) for i in test[:, 0]]
        dataSet_te = test[:, 1:]

        numSamples = train.shape[0]  # shape[0] stands for the num of row

        ## step 1: calculate Euclidean distance
        # tile(A, reps): Construct an array by repeating A reps times
        # the following copy numSamples rows for dataSet
        for m in range(len(dataSet_te)):
            if isEuclidian:
                distance = self.euclidean(dataSet_te, dataSet_tr, m, numSamples)
            else:
                distance = self.cosine(dataSet_te, dataSet_tr, m, numSamples)

            ## step 2: sort the distance
            # argsort() returns the indices that would sort an array in a ascending order
            sortedDistIndices = argsort(distance)

            classCount = {}  # define a dictionary (can be append element)
            for i in range(k):
                ## step 3: choose the min k distance
                voteLabel = labels_tr[sortedDistIndices[i]]

                ## step 4: count the times labels occur
                # when the key voteLabel is not in dictionary classCount, get()
                # will return 0
                classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

                ## step 5: the max voted class will return
            maxCount = 0
            for key, value in classCount.items():
                if value > maxCount:
                    maxCount = value
                    maxIndex = key

            if labels_te[m] == maxIndex:
                rightCount += 1

        ratio = rightCount / len(dataSet_te)  # 最终的交叉验证的正确率

        return ratio

    def euclidean(self, dataSet_te, dataSet_tr, m, numSamples):
        diff = tile(dataSet_te[m, :], (numSamples, 1)) - dataSet_tr  # Subtract element-wise
        squaredDiff = diff ** 2  # squared for the subtract
        squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
        distance = squaredDist ** 0.5
        return distance

    def cosine(self, dataSet_te, dataSet_tr, m, numSamples):
        dui = tile(dataSet_te[m, :], (numSamples, 1))
        numerator = sum(dui * dataSet_tr,axis=1)
        denominator=(sum(dui**2,axis=1)**0.5)*(sum(dataSet_tr**2,axis=1)**0.5)
        distance = numerator/denominator
        return -distance#余弦距离是最大的时候最接近，这里取反以顺应代码最小的情况最近

    def getData(self, dir):
        f = open(dir, 'r')
        first_ele = True
        m = zeros([0, 14], dtype=float)
        for data in f.readlines():
            ## 去掉每行的换行符，"\n"
            data = data.strip('\n')
            nums = data.split(",")
            nums = [float(x) for x in nums]
            m = append(m, [nums], axis=0)  # 追加的类型把必须完全一样，这里是list中裹list
        self.data = m;
        f.close()
        return self

    def crossValidation(self, k, isEucli=True, order=5):  # 5 =order
        ratio = zeros([order, 1])
        dataNum = round(len(self.data) / order)
        for i in range(order):
            s = i * dataNum
            if i == order - 1:
                test = self.data[s:, :]
                train = delete(self.data, s_[s:], 0)
            else:
                test = self.data[s:s + dataNum, :]
                train = delete(self.data, s_[s:s + dataNum], 0)

            ratio[i] = self.kNNClassify(train, test, k, isEucli)
        rightRatio = mean(ratio)
        return rightRatio

    def pca(self, dim=6):  # percentage表示方差百分比，用以确定降到多少维合适，不过实验要求6维
        newData, meanVal = self.zeroMean()
        covMat = cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        eigVals, eigVects = linalg.eig(covMat)  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        n=dim
        eigValIndice = argsort(eigVals)  # 对特征值从小到大排序
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
        lowDDataMat = dot(newData, n_eigVect)  # 低维特征空间的数据
        # reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
        # 修改data，以便后续计算
        lowData = column_stack((self.data[:, 0], lowDDataMat))
        self.data = lowData
        return self  # , reconMat

    def zeroMean(self):
        dataT = self.data[:, 1:]
        meanVal = mean(dataT, axis=0)  # 按列求均值，即求各个特征的均值
        newData = dataT - meanVal
        return newData, meanVal

    def normalization(self):
        dataT = self.data[:, 1:]
        minVal = np.min(dataT, axis=0)  # 按列求
        maxVal = np.max(dataT, axis=0)  # 按列求
        newData = (dataT - minVal) / (maxVal - minVal)
        self.data = column_stack((self.data[:, 0], newData))
        return self

    def lda(self,dim=6):
        dataT = self.data[:, 1:]
        label = [int(x) for x in self.data[:,0]]
        clusters = 3
        S_W = self.within_class_SW(dataT, label, clusters)
        S_B = self.between_class_SB(dataT, label, clusters)

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)#按特征值由大到小排序
        W=zeros([len(eig_pairs[0][1]),0])
        for i in arange(dim):
            # W[:,i] = eig_pairs[i][1].reshape(len(eig_pairs[i][1]), 1)
            W=np.append(W,eig_pairs[i][1].reshape(len(eig_pairs[i][1]), 1),axis=1)
        W=np.real(W)
        self.data=np.append(self.data[:,0][:,np.newaxis],dataT.dot(W),axis=1)
        return self

    # 计算类内散度
    def within_class_SW(self, data,label,clusters):
        m = data.shape[1]
        S_W = np.zeros((m,m))
        mean_vectors = self.class_mean(data,label,clusters)
        for cl ,mv in zip(range(1,clusters+1),mean_vectors):#enumerator(mean_vectors),不过这种方式的索引是0开始的
            class_sc_mat = np.zeros((m,m))
            # 对每个样本数据进行矩阵乘法
            ind = [x for x, y in enumerate(label) if y == cl]
            for row  in data[ind]:
                row ,mv =row.reshape(row.shape[0],1),mv.reshape(row.shape[0],1)
                class_sc_mat += (row-mv).dot((row-mv).T)
            S_W +=class_sc_mat
        #print S_W
        return S_W

    def between_class_SB(self, data, label, clusters):
        m = data.shape[1]
        all_mean = np.mean(data, axis=0)
        S_B = np.zeros((m, m))
        mean_vectors = self.class_mean(data, label, clusters)
        for cl, mean_vec in enumerate(mean_vectors):
            ind = [x for x, y in enumerate(label) if y == cl]
            n = data[ind, :].shape[0]#shape[0]是行数
            mean_vec = mean_vec.reshape(len(mean_vec), 1)  # make column vector
            all_mean = all_mean.reshape(len(all_mean), 1)  # make column vector
            S_B += n * (mean_vec - all_mean).dot((mean_vec - all_mean).T)
        # print S_B
        return S_B

    # 特征均值,计算每类的均值，返回一个向量
    def class_mean(sef, data, label, clusters):
        mean_vectors = zeros([0,data.shape[1]])
        for cl in range(1, clusters + 1):
            ind=[x for x,y in enumerate(label) if y==cl]
            mean_vectors=np.vstack((mean_vectors,np.mean(data[ind],axis=0)))
        return mean_vectors