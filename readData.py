# #!/usr/bin/env python
# # -*- coding:utf-8 -*-
import pandas as pd
import pickle as pick
import numpy as np
import matplotlib.pyplot as plt

# resultFile="./statisticLearning/proj4_data/result.pkl"
#
#
# p=open(resultFile,'rb')
# p=pick.load(p)
# pass
#

def maxValue(y):
    index = np.argmin(y)
    value = y[index]
    plt.plot(index, value, 'rs')
    show_max = '[' + str(index) + ',' + str(value) + ']'
    plt.annotate(show_max, xytext=(index, value), xy=(index, value))


s = pd.read_csv("./log1.txt", header=None, sep=' ').as_matrix()
ind = s[:, 0]
ind = [i for i in range(np.shape(ind)[0]) if not i % 4]
col = 3
for i, j in zip(range(4), ['k=10', 'k=20', 'k=50', 'k=100']):
    t=s[:, col][(np.array(ind) + i).tolist()]
    indx=np.arange(106)
    jj=[]
    for jjj,ii in enumerate(t):
        if not np.isnan(ii):
            jj.append(jjj)
    jj=np.array(jj)
    t=t[jj]
    t=np.interp(indx,jj,t)

    plt.plot(t, label=j)
    maxValue(t)

plt.xlabel("epoch")
plt.ylabel('mean reciprocal rank')
plt.grid()
plt.title("global test evaluation")
plt.legend(loc='best')

plt.show()

