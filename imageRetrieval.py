#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch as tc
import torch.nn as nn
from torch.autograd import Variable as var
import torch.utils.data as tud
import torchvision as tv
import Utils as ut
import matplotlib.pyplot as plot
import numpy as np
import os
from PIL import Image  as img
import pickle as pick

BATCH_SIZE = 64
EPOCH = 20
LEARNING_RATE = 0.01
IMAGE_SIZE = [128, 128]
IMAGE_TOTAL = 2000
IMAGE_FOLDERS=40
PRE_TRAIN_NET_PARAMETERS=True

RE_TRAIN=True # default training is complete that i use vector.pkl file to get data;
               # if need re-train network,set this parameter to True,then the code would
               # run to train the network, instead
logEpochFlag = "==============================================================================\n"
root = "~/pycharmProjects/pytorch_movan/statisticLearning/proj4_data/proj4/"
vectorFile = "./statisticLearning/proj4_data/vector.pkl"
resultFile="./statisticLearning/proj4_data/result.pkl"
cnnFile="./statisticLearning/proj4_data/cnn.pkl"


class Data(tud.Dataset):
    def __init__(self, root):
        self.train_data, self.train_labels, self.mean = self.getData(root)

    @ut.timing("read data")
    def getData(self, root):
        average = np.zeros((3,IMAGE_SIZE[0], IMAGE_SIZE[1]))
        images, label = np.zeros((IMAGE_TOTAL, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])), np.zeros(
            IMAGE_TOTAL)  # compress to 1/4 of origin image
        iter = [i for i in ut.eachFile(root) if i != "clutter"]
        for index, i in enumerate(sorted(iter)):

            for index1, j in enumerate(sorted(ut.eachFile("%s%s/" % (root, i)))):
                s = img.open(os.path.expanduser("%s%s/%s" % (root, i, j))).resize(
                    (IMAGE_SIZE[0], IMAGE_SIZE[1]))#convert('L')-->tranform gray pic
                s = np.array(s) / 255
                if len(np.shape(s))==3:# 3 dims
                    s=np.transpose(s,[2,0,1])
                else:#1 dims,that means it is gray picture
                    s=np.concatenate([s[np.newaxis, :, :]]*3,axis=0)
                average += s
                label[index * 50 + index1] = index
                images[index * 50 + index1] = s
        average /= IMAGE_TOTAL
        average = tc.from_numpy(average).type(tc.FloatTensor)
        images = tc.from_numpy(images).type(tc.FloatTensor)
        label = tc.from_numpy(label).type(tc.LongTensor)

        # average = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]))
        # images, label = np.zeros((IMAGE_TOTAL, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])), np.zeros(
        #     IMAGE_TOTAL)  # compress to 1/4 of origin image
        # iter = [i for i in ut.eachFile(root) if i != "clutter"]
        # for index, i in enumerate(sorted(iter)):
        #
        #     for index1, j in enumerate(sorted(ut.eachFile("%s%s/" % (root, i)))):
        #         s = img.open(os.path.expanduser("%s%s/%s" % (root, i, j))).convert('L').resize(
        #             (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        #         s = np.array(s) / 255
        #         average += s
        #         label[index * 50 + index1] = index
        #         images[index * 50 + index1] = np.array(s)[np.newaxis, :, :]
        # average /= IMAGE_TOTAL
        # average = tc.from_numpy(average).type(tc.FloatTensor)
        # images = tc.from_numpy(images).type(tc.FloatTensor)
        # label = tc.from_numpy(label).type(tc.LongTensor)

        return images, label, average

    def __getitem__(self, index):
        return self.train_data[index], self.train_labels[index]

    def __len__(self):
        return len(self.train_data)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.s1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)
        self.s2 = nn.ReLU()
        self.s3 = nn.MaxPool2d(4)  # n 48 32 32

        self.s4 = nn.Conv2d(48, 96, 3, 1, 1)
        self.s5 = nn.ReLU()
        self.s6 = nn.AvgPool2d(2)  # n 96 16 16

        self.s7 = nn.Conv2d(96, 192, 3, 1, 1)
        self.s8 = nn.ReLU()
        self.s9 = nn.MaxPool2d(2)  # n 192 8 8

        self.fulC = nn.Linear(192 * 8 * 8, 40)

        nn.init.normal(self.s1.weight, 0, 0.001)
        nn.init.normal(self.s4.weight, 0, 0.01)
        nn.init.normal(self.s7.weight, 0, 0.01)
        nn.init.normal(self.fulC.weight,0.01)
        nn.init.constant(self.s1.bias, 0)
        nn.init.constant(self.s4.bias, 0)
        nn.init.constant(self.s7.bias, 0)
        nn.init.constant(self.fulC.bias, 0)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.s6(x)
        x = self.s7(x)
        x = self.s8(x)
        x = self.s9(x)
        x = x.view(x.size(0), -1)  # n*(128 * 8 * 8)
        out = self.fulC(x)

        return out


class ImageRetrieval(object):
    def __init__(self):
        super(ImageRetrieval, self).__init__()
        self.data = Data(root)

    def euclideanDistances(self,A, B):
        if not isinstance(A,np.matrix):
            A=np.matrix(A)
        if not isinstance(B, np.matrix):
            B=np.matrix(B)
        BT = B.transpose()
        vecProd = A * BT
        SqA = A.getA() ** 2
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
        SqB = B.getA() ** 2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
        SqED = sumSqBEx + sumSqAEx - 2 * vecProd

        ED = SqED.getA()
        ED[ED<0]=0#some value close to zero but is negative,so do this step
        ED=np.nan_to_num(ED)#may has some nan value,this method used to transform nan to zero
        ED = ED ** 0.5
        # ED=np.matrix(ED) # return matrix,if comment this code,return array.
        return ED

    def computeResult(self,data,k):
        # use euclid distance
        rlt=self.euclideanDistances(data,data)
        indexRlt=np.argsort(rlt,axis=1)# return index but not value

        global_mrr_k=0
        class_mrr_k=np.zeros(40)
        globalRlt=0
        classRlt=np.zeros(40)

        result={}
        for i in range(IMAGE_TOTAL):
            downBound=np.floor(i/50)*50
            upBound=downBound+49
            tmp=indexRlt[i][:k]
            tmp=[j for j,i in enumerate(tmp) if i >=downBound and i <=upBound]

            classRlt[int(np.floor(i/50))]+=len(tmp)
            globalRlt+=len(tmp)

            tmp=np.array(tmp)+1# indice begin with 0 ,so it should add 1 to begin with 1
            tmp=np.sum(1/tmp)/np.size(tmp)
            class_mrr_k[int(np.floor(i/50))]+=tmp
            global_mrr_k+=tmp

        globalRlt/=IMAGE_TOTAL
        classRlt/=50

        global_mrr_k/=IMAGE_TOTAL
        class_mrr_k/=50

        class_p_k=classRlt/k
        global_p_k=globalRlt/k
        class_r_k = classRlt / 50
        global_r_k = globalRlt / 50
        class_f_k = (2*class_p_k*class_r_k) / (class_p_k+class_r_k)
        global_f_k = (2*global_p_k*global_r_k) / (global_p_k+global_r_k)

        result['class_mrr_k']=class_mrr_k
        result['global_mrr_k']=global_mrr_k
        result['class_p_k']=class_p_k
        result['global_p_k']=global_p_k
        result['class_r_k']=class_r_k
        result['global_r_k']=global_r_k
        result['class_f_k']=class_f_k
        result['global_f_k']=global_f_k

        return result

    @ut.timing("Total")
    def dataHandle(self):
        vector = np.zeros((IMAGE_TOTAL, IMAGE_FOLDERS))

        if not RE_TRAIN:
            tmp = open(vectorFile, 'rb')
            vector=pick.load(tmp)
        else:
            cnn = CNN()
            if PRE_TRAIN_NET_PARAMETERS:
               cnn.load_state_dict(tc.load(cnnFile))

            cnn.cuda()
            trainLoader = tud.DataLoader(dataset=self.data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
            test = var(self.data.train_data - self.data.mean,volatile=True)  # broadcast
            test=test.cuda()

            optimizer = tc.optim.Adam(params=cnn.parameters(), lr=LEARNING_RATE)
            lossFunc = nn.CrossEntropyLoss().cuda()
            for epoch in range(EPOCH):
                with ut.Timing("epoch" + str(epoch)):
                    for step, (train_x, train_y) in enumerate(trainLoader):
                        train_x = train_x - self.data.mean
                        train_x = var(train_x).cuda()
                        train_y = var(train_y).cuda()
                        out = cnn(train_x)
                        loss = lossFunc(out, train_y)
                        # backPropagation
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    for i in range(20):
                        # vector[i*100:(i+1)*100,:].data.numpy()
                        rlt = cnn(test[i*100:(i+1)*100])
                        vector[i * 100:(i + 1) * 100, :]=rlt.cpu().data.numpy()

                    result = {}
                    for k in [10, 20, 50, 100]:
                        result[str(k)] = self.computeResult(vector, k=k)
                        with ut.Log("epoch: %d | global_p_k: %.4f\n" % (epoch,result[str(k)]['global_p_k'])):
                            print("epoch: %d | global_p_k: %.4f" % (epoch,result[str(k)]['global_p_k']))

            tc.save(cnn.state_dict(), cnnFile)

            # for i in range(20):
            #     rlt = cnn(test[i*100:(i+1)*100])
            #     vector[i * 100:(i + 1) * 100, :]=rlt.cpu().data.numpy()
            # tmp = open(vectorFile, 'wb')
            # pick.dump(vector, tmp)
        return vector

@ut.log(None)
def main():
    imgR = ImageRetrieval()
    vector=imgR.dataHandle()
    # result={}
    # for k in [10,20,50,100]:
    #     result[str(k)]=imgR.computeResult(vector,k=k)
    #     with ut.Log("global_p_k: %.4f\n"%result[str(k)]['global_p_k']):
    #         print("global_p_k: %.4f"%result[str(k)]['global_p_k'])
    #
    # tmp=open(resultFile,'wb')
    # pick.dump(result,tmp)

    return logEpochFlag

main()
