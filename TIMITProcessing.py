#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch as tc
import torchvision as tv
import torch.utils.data as tud
from torch.autograd import Variable as var
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import pandas as pd
import pickle as pk
import re  # regex module
import Utils as ut
import multiprocessing as mp

logSplit="==========================================================================="
with ut.Log('./log.txt', logSplit) as t:
    pass

# 路径数据
TRAIN_DIR = "/home/jiangyiheng/data/TIMIT/feature/TRAIN/"
TEST_DIR = "/home/jiangyiheng/data/TIMIT/feature/TEST/"
MEAN_DIR = "/home/jiangyiheng/data/TIMIT/feature/tmp/mean"
STD_DIR = "/home/jiangyiheng/data/TIMIT/feature/tmp/std"

EPOCH = 10
LearningRate = 1e-3
htk = ut.HTKFile()
batchSize = 32


def meanValue(data_path):
    tmpS = np.zeros([1, 39])
    rows = 0
    files = os.listdir(data_path)
    for fname in files:
        ffpath = data_path + fname
        df = pd.read_csv(ffpath, sep=' ', dtype=float)
        # dropna is method to remove the NaN data
        tmp = df.dropna(axis=1, how='all').as_matrix()
        rows += np.size(tmp, axis=0)
        tmpS += np.sum(tmp, axis=0)
    return tmpS / rows


def stdValue(data_path, meanValue):
    tmpS = np.zeros([1, 39])
    rows = 0
    files = os.listdir(data_path)
    for fname in files:
        ffpath = data_path + fname
        df = pd.read_csv(ffpath, sep=' ', dtype=float)
        # dropna is method to remove the NaN data
        tmp = df.dropna(axis=1, how='all').as_matrix()
        rows += np.size(tmp, axis=0)
        tmpS += np.sum(((tmp - meanValue) ** 2), axis=0)
    return np.sqrt(tmpS / rows)


class dataGet(tud.Dataset):
    def init(self,l):
            global lock
            lock = l

    @ut.timing(None)
    def __init__(self, root, mean, std, maxR=0):
        self.mean = mean
        self.std = std
        files = os.listdir(root)
        files.sort()
        self.train_data = []
        self.train_labels = []
        self.minR = 1000
        self.maxR = 0
        self.count = 0
        self.root=root
        # for i, fname in enumerate(files):
        #     getFileData(fname)

        lock = mp.Lock()
        # change lock to global variable when init the pool
        pool = mp.Pool(16, initializer=self.init, initargs=(lock,))
        pool.map(self.getFileData, files)
        pool.close()
        pool.join()

        if maxR:
            self.maxR = maxR


    # declare global lock,it will be use in pool
    def getFileData(self,fname):
        p = re.compile("_")
        regexRslt = re.split(p, fname)
        if regexRslt[2][:2] == "SA":
            return
        ffpath = self.root + fname
        # tmp=htk(ffpath).data
        df = pd.read_csv(ffpath, sep=' ', dtype=float)
        tmp = df.dropna(axis=1, how='all').as_matrix()
        row = np.size(tmp, axis=0)
        lock.acquire() # add lock to source as follow:
        self.count += 1
        print(str(self.count),"\n")
        self.train_labels.append([np.floor(self.count / 8.1), regexRslt[1]])
        self.train_data.append(tmp)
        if row > self.maxR:
            self.maxR = row
        lock.release() # release the lock


    def __getitem__(self, index):
        data = self.train_data[index]
        data = (data - self.mean) / self.std
        diffR = self.maxR - np.size(data, axis=0)
        if diffR > 0:
            while diffR:
                data = np.concatenate([data, data[:diffR]])
                diffR = self.maxR - np.size(data, axis=0)
        elif diffR < 0:
            data = data[:diffR]
        data = data[np.newaxis, :, :]
        self.train_labels[index][0] = int(self.train_labels[index][0])
        return tc.from_numpy(data).type(tc.FloatTensor), self.train_labels[index]

    def __len__(self):
        return self.count

if not os.path.exists(MEAN_DIR) or not os.path.exists(STD_DIR):
    meanV = meanValue(TRAIN_DIR)
    stdV = stdValue(TRAIN_DIR, meanV)
    output_meanV = open(MEAN_DIR, 'wb')
    pk.dump(meanV, output_meanV)
    output_stdV = open(STD_DIR, 'wb')
    pk.dump(stdV, output_stdV)
else:
    output_meanV = open(MEAN_DIR, 'rb')
    output_stdV = open(STD_DIR, 'rb')
    meanV = pk.load(output_meanV)
    stdV = pk.load(output_stdV)
output_meanV.close()
output_stdV.close()


# globle init by external method,sort by order of class definition in the __init__()
def weights_init(m):
    # classname=m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.normal(m.weight.data, 0, 0.0001)
        nn.init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0)


class cnn(nn.Module):
    def __init__(self, frameNum, personNum):  # frameNum=776
        super(cnn, self).__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),  # 16 * 388 * 20
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(16, 48, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48*194*10
        )
        self.fc = nn.Linear(int(frameNum / 4) * 48 * 10, int(personNum))
        # nn.init.normal(self.s1.parameters(), 0, 0.0001)
        # nn.init.normal(self.s2.parameters(), 0, 0.001)
        # nn.init.normal(self.fc.parameters(), 0, 0.01)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


train_set = dataGet(TRAIN_DIR, meanV, stdV)
train_loader = tud.DataLoader(dataset=train_set, batch_size=batchSize, shuffle=True, num_workers=16)

test_set = dataGet(TEST_DIR, meanV, stdV, train_set.maxR)
# must call __getitem__ () explicitly to transform the test_data,while we can concatenate all test data
test_x = []
test_y = []
for i in np.arange(test_set.__len__()):
    test_x_tmp, test_y_tmp = test_set.__getitem__(i)
    test_x.append(test_x_tmp)
    test_y.append(test_y_tmp)
test_x = tc.from_numpy(np.concatenate(test_x)[:, np.newaxis, :, :])[:batchSize]
test_x = var(test_x).cuda()
test_y = tc.from_numpy(np.array([i[0] for i in test_y]))[:batchSize].type(tc.LongTensor).cuda()
cnn = cnn(train_set.maxR, train_set.count / 8)
cnn.apply(weights_init)  # apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上。
lossFunc = nn.CrossEntropyLoss().cuda()
optimizer = tc.optim.Adam(cnn.parameters(), lr=LearningRate)


for epoch in range(EPOCH):
    with ut.Timing("epoch: %d" % epoch) as t:
        for step, (train_x, train_y) in enumerate(train_loader):
            train_x = var(train_x).cuda()
            train_y = var(train_y[0]).cuda()
            out = cnn(train_x)
            loss = lossFunc(out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step % 50 == 0):
                # test(epoch, step)
                test_out = cnn(test_x)
                pre_y = tc.max(test_out, 1)[1].data.squeeze()
                accuracy = sum(pre_y == test_y) / len(test_y)
                s = "epoch: %d   step: %d   accuracy: %.3f\n" % (epoch, step, accuracy)
                with ut.Log('./log.txt',s) as t:
                    pass
                print(s)
