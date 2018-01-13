#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import Utils as ut
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

root = "./statisticLearning/proj2_data/"
logEpochFlag = "==============================================================================\n"


class BinClassfier(object):
    def __init__(self):
        super(BinClassfier, self).__init__()
        self.label = None
        self.trainData = None
        self.px = None
        self.pt_x = None
        self.testData = None

    def lineRegressioner(self, order, train, train_label, test, test_label, lamda=0):
        matrix = np.zeros([len(train), order * 2])
        matrix_test = np.zeros([len(test), order * 2])
        for i in range(order):
            matrix[:, 2 * i:2 * (i + 1)] = train ** (i + 1)
            matrix_test[:, 2 * i:2 * (i + 1)] = test ** (i + 1)
        w = np.matmul(
            np.matmul(np.linalg.inv(lamda * np.eye(order * 2, order * 2) + np.matmul(matrix.T, matrix)),
                      matrix.T), train_label)
        predict = np.abs(np.matmul(matrix_test, w))
        error = self.calculateLoss(predict, test_label)
        return w, error

    # default select 5-cross validation
    def lineRegressionerWithCrossValidation(self, order, k=5, lamda=1,isComputeTest=False):
        if isComputeTest:
            ww, error = self.lineRegressioner(order, self.trainData, self.label,self.testData, None,lamda)
        else:
            dataNum = int(len(self.label) / k)
            tmp = np.concatenate([self.trainData, self.label[:, np.newaxis]], 1)
            error = np.zeros(k)
            ww=[]
            for i in range(k):
                s = i * dataNum
                if i == k - 1:
                    test = tmp[s:, :]
                    train = np.delete(tmp, np.s_[s:], 0)
                else:
                    test = tmp[s:s + dataNum, :]
                    train = np.delete(tmp, np.s_[s:s + dataNum], 0)

                w, error[i] = self.lineRegressioner(order, train[:, :-1], train[:, -1], test[:, :-1], test[:, -1], lamda)
                ww.append(w)
            ww=ww[np.argmin(error)]
            error = np.mean(error)
        return ww,error

    def getData(self):
        self.label = pd.read_csv(root + "ctrain.txt", sep=',', dtype=float, header=None).dropna(axis=1,
                                                                                                how='all').as_matrix()
        self.trainData = pd.read_csv(root + "xtrain.txt", sep=',', dtype=float, header=None).dropna(axis=1,
                                                                                                    how='all').as_matrix()
        # disorder the trainData
        tmp = np.concatenate([self.trainData, self.label], 1)
        np.random.shuffle(tmp)
        self.label = tmp[:, -1].astype(int)
        self.trainData = tmp[:, :-1]
        self.px = pd.read_csv(root + "ptest.txt", sep=',', dtype=float, header=None).dropna(axis=1,
                                                                                            how='all').as_matrix()
        self.pt_x = pd.read_csv(root + "c1test.txt", sep=',', dtype=float, header=None).dropna(axis=1,
                                                                                               how='all').as_matrix()
        self.testData = pd.read_csv(root + "xtest.txt", sep=',', dtype=float, header=None).dropna(axis=1,
                                                                                                  how='all').as_matrix()

        return self

    # label is None representing that use test data to calculate loss,unless using cross validation or train set to calculate loss
    def calculateLoss(self, predict, label):
        if isinstance(label,np.ndarray):
            predict = np.around(predict)
            right = len(predict[predict == label])
            error = (len(label) - right) / len(label)
        else:
            n0 = [index for index, element in enumerate(predict) if element < 0.5]
            n1 = [index for index, element in enumerate(predict) if element >= 0.5]
            error = sum(self.px[n0] * self.pt_x[n0]) + sum(self.px[n1] * (1 - self.pt_x[n1]))
        return error

    def logisticRegression(self,order,train,train_label,test, test_label):
        matrix = np.zeros([len(train), order * 2])
        matrix_test = np.zeros([len(test), order * 2])
       #  init the weighted 'w' stocastically
        w=np.random.rand(order*2)
        for i in range(order):
            matrix[:, 2 * i:2 * (i + 1)] = train ** (i + 1)
            matrix_test[:, 2 * i:2 * (i + 1)] = test ** (i + 1)
        # use Newton Raphson algorithm to solve the best solution of 'w'
        for i in range(20):
            y=np.squeeze(1/(1+np.exp(-np.matmul(matrix,w))))
            # converge condition:
            if len(train)-(len(y[y<1e-3])+len(y[y>1-1e-3]))<30:
                break
            else:
                w=w-np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(matrix.T,np.diag(y*(1-y))),matrix)),matrix.T),(y-train_label))
        # model evidence
        predict = 1/(1+np.exp(-np.matmul(matrix,w)))
        label1,label0=np.where(self.label==1),np.where(self.label==0)
        product=np.prod(predict[label1])*np.prod(1-predict[label0])
        evi=np.sum(np.log(np.abs(product)))-order*np.log(len(train))/2
        # use test set to calculate the errorRate
        predict_test = 1/(1+np.exp(-np.matmul(matrix_test,w)))
        error = self.calculateLoss(predict_test, test_label)
        return w, error,evi






@ut.log(None)
def main():
    def plot(error, xValue):
        xmajorFormatter = FormatStrFormatter('%d')
        ax = plt.subplot(111)
        ax.xaxis.set_major_formatter(xmajorFormatter)
        index = np.argmin(error)
        value = error[index]
        plt.plot(np.arange(len(error)), error)
        plt.plot(index, value, 'gs')
        plt.grid()
        show_max = '[' + str(index) + ' ' + str(value) + ']'
        plt.annotate(show_max, xytext=(index, value), xy=(index, value))
        plt.xlabel(xValue)
        plt.ylabel("errorRate")
        plt.savefig("/home/jiangyiheng/fig.jpg")
        plt.show()

    binC = BinClassfier()
    binC.getData()

    # test different lambda value,result appearence that lambda is useless ,so we needn't use it
    # ==============================================================================
    # error=np.zeros(20)
    # for lamda in range(20):
    #     w,error[lamda] = binC.lineRegressioner(1,binC.trainData,binC.label,binC.trainData,binC.label,lamda)
    # print(error)
    # plot(error, "lambda")

    # cross validation to find out an optimal set of basis functions
    # ==============================================================================
    # error = np.zeros(20)
    # ww = []
    # for order in range(1, 21):
    #     w,error[order - 1] = binC.lineRegressionerWithCrossValidation(order)
    #     ww.append(w)
    # print(error)
    # plot(error, "order")
    # with ut.Log(str(ww)+"\n"+str(error)+"\n") as l:
    #     pass

    # varify the best choice of order using the test data
    # ==============================================================================
    # error = np.zeros(20)
    # for order in range(1, 21):
    #     w,error[order - 1] = binC.lineRegressionerWithCrossValidation(order,isComputeTest=True)
    # print(error)
    # plot(error, "order")

    # logistic regression
    # ==============================================================================
    error = np.zeros(20)
    for order in range(1, 21):
        w,error[order - 1],evidence = binC.logisticRegression(order,binC.trainData,binC.label,binC.testData,None)
        print(w)
    plot(error, "order")




    return logEpochFlag


main()
