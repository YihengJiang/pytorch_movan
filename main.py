#!/usr/bin/env python
# -*- coding:utf-8 -*-
#################################
# readme:
#       knn is the algorithm core ,main is begin of the project
#################################
import Knn
import numpy as n
import matplotlib as mpl
import matplotlib.pyplot as plt

knnDataDir = r'F:\project\proj1_data\wine.data'
k = Knn.Knn()

def plotPro():
    zhfont1 = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
    # mpl.rcParams['xtick.labelsize'] = 24
    # mpl.rcParams['ytick.labelsize'] = 24
    plt.grid()
    plt.title("交叉验证结果（k=1~140）", fontproperties=zhfont1)
    plt.xlabel("K")
    plt.ylabel("accuracy")
    plt.show()


###########################################各种做法：每次选择一种情况运行即可
# k = k.getData(knnDataDir)#最原始的处理
# k = k.getData(knnDataDir).pca()#默认时降6维
# k = k.getData(knnDataDir).normalization().pca()#归一化
# k = k.getData(knnDataDir).normalization().pca(3)#实验发现降3维最好，所以后续实验采用3维
k = k.getData(knnDataDir).normalization().lda(3)#lda降维
ratio = n.zeros([140, 1])
for i in range(1, 141):  # k=1到140
    ratio[i - 1] = k.crossValidation(i)  # 默认时欧几里得距离
#   ratio[i - 1] = k.crossValidation(i,False)
maxV = n.max(ratio, axis=0)
ra = [(x, float(y)) for x, y in enumerate(ratio) if y < maxV]
plt.plot(n.array(ra)[:, 0], n.array(ra)[:, 1], '.')

maxP = [(x, float(y)) for x, y in enumerate(ratio) if y == maxV]
plt.plot(n.array(maxP)[:, 0], n.array(maxP)[:, 1], linewidth=2, marker='o')

print("该情况下，最好的k选择是k=" + str(n.array(maxP)[:, 0] + 1) + "，对应的正确率是：" + str(float(maxV)))

plotPro()
#####################################

##############################测试pca降维的情况的代码，发现最好是降到3维
# pcaD = n.zeros([8, 1])
# for mm in range(3, 11):
#     k = k.getData(knnDataDir).normalization().pca(mm)
#     ratio = n.zeros([140, 1])
#     for i in range(1, 141):  # k=1到140
#         ratio[i - 1] = k.crossValidation(i)
#     maxV = n.max(ratio, axis=0)
#     pcaD[mm - 3] = maxV
#     for i in range(len(ratio)):
#         if maxV[0] == ratio[i, 0]:
#             print(str(i) + ":" + str(maxV))
#
# plt.plot(range(3, 11), pcaD, '.')
# plt.show()
##############################测试pca降维的情况的代码，发现最好是降到3维


############################数据可视化：箱线图####################
# k = k.getData(knnDataDir)
# # k = k.getData(knnDataDir).normalization()
# data=k.data[:,1:]
# plt.boxplot(data, labels =n.arange(1,14) , sym = "o")
# plt.show()
############################数据可视化：箱线图####################



