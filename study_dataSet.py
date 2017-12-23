#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch as tc
import torchvision as tv
import torch.nn as nn
from cffi.model import global_lock
from torch.autograd import Variable as var
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pickle as pk

TRAIN_SUM = tc.zeros((3, 32, 32)).float()
TRAIN_VARIANCE = tc.zeros((3, 32, 32)).float()

# super parameters
isNeedDownload = False
batchSize = 100
learningRate = 1e-3
EPOCH = 15
varifyNum = 10
testNum = 200


def showImg(img):
    img = img / 2 + 0.5  # unnormalize
    npImg = img.numpy()
    ss = np.transpose(npImg, (1, 2, 0))
    plt.imshow(ss)  # permute the dim of array,row*column*channel
    plt.show()


# import data from pytorch's torchvision
if not (os.path.exists('./CIFAR10/')) or not os.listdir('./CIFAR10/'):
    isNeedDownload = True
    # transform will normalize the data to 0~1 and then transform what you want by specified the value of para(transform).

trainSet = tv.datasets.CIFAR10(root='./CIFAR10', transform=tv.transforms.ToTensor(), train=True,
                               download=isNeedDownload)
trainLoader = data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, num_workers=16)
# get mean and variance###########################################
if not os.path.exists("./CIFAR10/average.pkl") or not os.path.exists("./CIFAR10/variance.pkl"):
    for x, y in trainLoader:
        tc.sum(x, 0)
        TRAIN_SUM += tc.sum(x, 0)
    TRAIN_SUM /= len(trainSet)  # mean value
    for x, y in trainLoader:
        tmp = -TRAIN_SUM
        tmp.repeat(x.size(0), 1, 1, 1)  # repeat()'s using way is very especial,i should understand and remember it
        TRAIN_VARIANCE += tc.sum((tmp + x) ** 2, 0)
    TRAIN_VARIANCE = tc.sqrt(TRAIN_VARIANCE / len(trainSet))  # variance value
    output_average = open("./CIFAR10/average.pkl", 'wb')
    pk.dump(TRAIN_SUM, output_average)
    output_variance = open("./CIFAR10/variance.pkl", 'wb')
    pk.dump(TRAIN_VARIANCE, output_variance)
else:
    output_average = open("./CIFAR10/average.pkl", 'rb')
    output_variance = open("./CIFAR10/variance.pkl", 'rb')
    TRAIN_SUM = pk.load(output_average)
    TRAIN_VARIANCE = pk.load(output_variance)

testSet = tv.datasets.CIFAR10('./CIFAR10', False, tv.transforms.ToTensor())
# print(len(testSet))#10000
tmp_test_x = tc.from_numpy(np.transpose((testSet.test_data[:testNum] / 255), (0, 3, 1, 2))).type(tc.FloatTensor)
tmp_average = TRAIN_SUM.repeat(len(tmp_test_x), 1, 1, 1)
tmp_variance = TRAIN_VARIANCE.repeat(len(tmp_test_x), 1, 1, 1)
tmp_test_x = (tmp_test_x - tmp_average) / tmp_variance

test_x = var(tmp_test_x)
test_xx = tc.from_numpy(np.transpose((testSet.test_data[:varifyNum] / 255), (0, 3, 1, 2))).type(tc.FloatTensor)
showImg(tv.utils.make_grid(test_xx))  # make grid to compose all the pics to 1 pic
test_x = test_x.cuda()
test_y = tc.from_numpy(np.array(testSet.test_labels[:testNum])).type(tc.LongTensor).cuda()
# data labels is number 0-9,corresponding to the trainLabel as below, respectively
trainLabel = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# <editor-fold desc="Description">
# this simple constructor is not flexible(elastic) sometimes,so i think it is worse than class constructor
cnn = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 48*16*16
    nn.Conv2d(48, 96, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 96*8*8
    nn.Linear(96 * 8 * 8, 10)  # have 10 classes ,so fully connect to 10 output neurens
)


# </editor-fold>


class cnnC(nn.Module):
    def __init__(self):
        super(cnnC, self).__init__()
        # self.cnnC1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 48*16*16
        # )
        # self.cnnC2 = nn.Sequential(
        #     nn.Conv2d(48, 96, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 96*8*8
        # )
        self.fulC = nn.Linear(96 * 8 * 8, 10)

        self.s1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)
        self.s2 = nn.ReLU()
        self.s3 = nn.MaxPool2d(2)
        self.s4 = nn.Conv2d(48, 96, 3, 1, 1)
        self.s5 = nn.AvgPool2d(2)
        nn.init.normal(self.s1.weight, 0, 0.001)
        nn.init.normal(self.s4.weight, 0, 0.01)
        nn.init.constant(self.s1.bias, 0)
        nn.init.constant(self.s4.bias, 0)

    def forward(self, x):
        # x = self.cnnC1(x)
        # x = self.cnnC2(x)
        # x = x.view(x.size(0), -1)
        # out = self.fulC(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s2(x)
        x = self.s5(x)
        x = x.view(x.size(0), -1)
        out = self.fulC(x)

        return out


class cnnC2(nn.Module):
    def __init__(self):
        super(cnnC2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.active = nn.ReLU()
        self.poolMax = nn.MaxPool2d(3, 2)  # 32*15*15
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.poolMean = nn.AvgPool2d(3, 2)  # conv2:32*7*7   conv3:64*3*3
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.fullConn = nn.Linear(64 * 3 * 3, 10)
        self.outActive = nn.Softmax(dim=1)
        nn.init.normal(self.conv1.weight, 0, 0.0001)
        nn.init.normal(self.conv2.weight, 0, 0.01)
        nn.init.normal(self.conv3.weight, 0, 0.01)
        nn.init.normal(self.fullConn.weight, 0, 0.1)
        nn.init.constant(self.conv1.bias, 0)
        nn.init.constant(self.conv2.bias, 0)
        nn.init.constant(self.conv3.bias, 0)
        nn.init.constant(self.fullConn.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.active(x)
        x = self.poolMax(x)
        x = self.conv2(x)
        x = self.active(x)
        x = self.poolMean(x)
        x = self.conv3(x)
        x = self.active(x)
        x = self.poolMean(x)
        x = self.fullConn(x.view(x.size(0), -1))
        x = self.outActive(x)
        return x

    # def iniPara(self):
    #     #classname=m.__class__.__name__    can get the name of class
    #     nn.init.normal(self.conv1.weight,)
    # should init the parameters in instant the class


cnn2 = cnnC().cuda()

optimizer = tc.optim.Adam(params=cnn2.parameters(), lr=learningRate)
lossFunc = nn.CrossEntropyLoss().cuda()
for epoch in range(EPOCH):
    for step, (train_x, train_y) in enumerate(trainLoader):
        tmp_average = TRAIN_SUM.repeat(len(train_x), 1, 1, 1)
        tmp_variance = TRAIN_VARIANCE.repeat(len(train_x), 1, 1, 1)
        tmp_x = (train_x - tmp_average) / tmp_variance

        train_x = var(tmp_x).cuda()
        train_y = var(train_y).cuda()
        out = cnn2(train_x)
        loss = lossFunc(out, train_y)
        # backPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step % 500 == 0):
            test_out = cnn2(test_x)
            pre_y = tc.max(test_out, 1)[1].data.squeeze()
            accuracy = sum(pre_y == test_y) / len(test_y)
            print("epoch:", epoch, " step:", step, " accuracy:", accuracy)

prin_out = cnn2(test_x[:varifyNum])
pre_y = tc.max(prin_out, 1)[1].data.squeeze()

print(" ".join("%5s" % trainLabel[pre_y[j]] for j in range(varifyNum)))

# test if cuda can use
# print(tc.cuda.is_available())
