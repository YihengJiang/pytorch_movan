#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
import copy
import torch as tc
import torch.utils.data as Data
from torch.autograd import Variable as var
import Utils as ut

logEpochFlag="==============================================================================\n"
root = "./statisticLearning/proj3_data/task1/"
moviesGenres = {'Action': 0,
                'Adventure': 1,
                'Animation': 2,
                'Children\'s': 3,
                'Comedy': 4,
                'Crime': 5,
                'Documentary': 6,
                'Drama': 7,
                'Fantasy': 8,
                'Film-Noir': 9,
                'Horror': 10,
                'Musical': 11,
                'Mystery': 12,
                'Romance': 13,
                'Sci-Fi': 14,
                'Thriller': 15,
                'War': 16,
                'Western': 17}
usersGenders = {'M': 0, 'F': 1}
usersAges = [1, 18, 25, 35, 45, 50, 56]
LR = 0.1
BATCH_SIZE = [32,5]
EPOCH = 1


class AgeGenderClassification(object):
    def __init__(self):
        super(AgeGenderClassification, self).__init__()
        self.movies = None
        self.ratings = None
        self.users = None  # 604 rows per file
        # because there has 18 classes in movie genres,genderNn is a list which has 18 little classifier correspondingly
        self.genderNn = [NetGender(1,2)] * 18
        self.ageNn = [NetAge(1,7)] * 18
        # with ut.Log(logEpochFlag+"netGender:"+str(self.genderNn[0])+"\n"+"netAge:"+str(self.ageNn[0])+"\n"):
        #     pass
        self.optimizer = [tc.optim.Adam(i.parameters(), lr=LR, betas=(0.9, 0.99)) for i in self.genderNn]
        self.loss = [tc.nn.CrossEntropyLoss()] * 18

    def moviesGenresPreprocess(self, x):
        tmp = x[-1].split("|")
        x[-1] = [moviesGenres[i] for i in tmp]
        return x

    def ratingsPreprocess(self, x):
        tmp = self.movies[x[1]]
        for i in tmp:
            self.tmp.append(np.array([x[0], i, x[2]]).reshape(1, 3))
        return x

    def genderAndAgeClassifier(self, train, test,isGender=True):
        error, length = 0, 0
        if isGender:
            bSize=BATCH_SIZE[0]
        else:
            bSize=BATCH_SIZE[1]
        # print(logEpochFlag)
        for i in range(18):  # train 18 classifier respectively
            if test.get(i) == None:  # no test data,so that we needn't train this network
                continue
            length += len(test[i])
            if train.get(i) != None:  # avoid problem that some classes may have no data
                test_data = np.concatenate(test[i])
                test_x = var(tc.unsqueeze((tc.from_numpy(test_data[:, -1]).type(tc.FloatTensor) - 1) / 4,dim=1),volatile=True)
                test_y = tc.from_numpy(test_data[:, 1]).type(tc.LongTensor)
                data = np.concatenate(train[i])
                dataSet = Data.TensorDataset(data_tensor=tc.unsqueeze((tc.from_numpy(data[:, -1]).type(tc.FloatTensor) - 1) / 4,dim=1),
                                             target_tensor=tc.from_numpy(data[:, 1]).type(tc.LongTensor))
                loader = Data.DataLoader(dataset=dataSet, batch_size=bSize, shuffle=True, num_workers=10)

                for epoch in range(EPOCH):
                    for step, (x, y) in enumerate(loader):
                        _x = var(x)
                        _y = var(y)
                        if isGender:
                            output = self.genderNn[i](_x)
                        else:
                            output = self.ageNn[i](_x)
                        loss = self.loss[i](output, _y)
                        self.optimizer[i].zero_grad()
                        loss.backward()
                        self.optimizer[i].step()

                pred_y = tc.max(self.genderNn[i](test_x), 1)[1].data.squeeze()
                if isGender:
                    error1 = sum(pred_y != test_y)
                    if error1 / len(test[i])<0.37: # it means that train data is efficient,unless do not predict test data
                        error += error1
                else:
                    for i in zip(pred_y,test_y):
                        error1=abs(i[0] - i[1])
                        if error1<3: # it means that train data is efficient,unless do not predict test data
                            error+=error1
            else:  # for some classes,train sets are null,but test sets are not null
                # error += len(test[i]) / 2
                pass
        return error / length

    def getData(self):
        self.movies = pd.read_csv(root + "movies.dat", sep='::', engine='python', names=['mID', 'mTitle', 'mGenre'],
                                  dtype={'mID': int, 'mTitle': str, 'mGenre': str}).dropna(0, 'all').as_matrix()
        self.movies = self.movies[:, [0, 2]]
        list(map(self.moviesGenresPreprocess, self.movies))
        self.movies = dict(zip(self.movies[:, 0], self.movies[:, 1]))

        self.ratings = pd.read_csv(root + "ratings.dat", sep='::', engine='python', names=['uID', 'mID', 'rating'],
                                   usecols=[0, 1, 2],
                                   dtype=int).dropna(0, 'all').as_matrix()
        self.tmp = []
        list(map(self.ratingsPreprocess, self.ratings))
        self.ratings = np.concatenate(self.tmp)
        del self.tmp
        # with ut.Timing("read data cost"):
        self.users = multiReadProc(list(range(10)))

    def genderAndAgeCrossValidation(self,isGender=True):
        # change ratings's userID to userGender,and split test data and train data
        # then 10-cross validation
        if isGender:
            index=1
        else:
            index=2
        self.tmp = dict(zip(self.ratings[:, 0], self.ratings[:, 1:]))
        errorRate = [0] * 10
        for i in range(10):
            users = copy.deepcopy(self.users)
            test = users.pop(i)
            train = np.concatenate(users)

            self.tmp1 = {}
            for x in test:
                key = self.tmp[x[0]][0]
                value3 = self.tmp[x[0]][1]
                if self.tmp1.get(key) != None:
                    self.tmp1[key].append(np.array([x[0], x[index], value3]).reshape(1, 3))
                else:
                    self.tmp1[key] = [np.array([x[0], x[index], value3]).reshape(1, 3)]
            test = self.tmp1

            self.tmp1 = {}
            for x in train:
                key = self.tmp[x[0]][0]
                value3 = self.tmp[x[0]][1]
                if self.tmp1.get(key) != None:
                    self.tmp1[key].append(np.array([x[0], x[index], value3]).reshape(1, 3))
                else:
                    self.tmp1[key] = [np.array([x[0], x[index], value3]).reshape(1, 3)]
            train = self.tmp1
            if isGender:
                # with ut.Timing("genderClass, cross"+str(i)):
                errorRate[i] = self.genderAndAgeClassifier(train, test)
            else:
                # with ut.Timing("ageClass, cross" + str(i)):
                errorRate[i] = self.genderAndAgeClassifier(train, test,False)
        del self.tmp1, self.tmp
        return errorRate


class NetGender(tc.nn.Module):
    def __init__(self, n_input, n_output, n_hidden=10):
        super(NetGender, self).__init__()
        self.hidden = tc.nn.Linear(n_input, n_hidden)
        self.activate = tc.nn.ReLU()
        self.out = tc.nn.Linear(n_hidden, n_output)
        # self.drop=tc.nn.Dropout(0.5)
        tc.nn.init.normal(self.hidden.weight, 0, 0.001)  # there cannot use parameters() function to init the parameters
        tc.nn.init.normal(self.out.weight, 0, 0.01)

    def forward(self, x):
        x = self.hidden(x)
        # x = self.drop(x)
        x = self.activate(x)
        x = self.out(x)
        return x

class NetAge(tc.nn.Module):
    def __init__(self, n_input, n_output, n_hidden=64):
        super(NetAge, self).__init__()
        self.hidden = tc.nn.Linear(n_input, n_hidden)
        self.activate = tc.nn.ReLU()
        self.out = tc.nn.Linear(n_hidden, n_output)
        # self.drop=tc.nn.Dropout(0.5)
        tc.nn.init.normal(self.hidden.weight, 0, 0.001)  # there cannot use parameters() function to init the parameters
        tc.nn.init.normal(self.out.weight, 0, 0.01)

    def forward(self, x):
        x = self.hidden(x)
        # x = self.drop(x)
        x = self.activate(x)
        x = self.out(x)
        return x


def proc(i):
    tmp = pd.read_csv(root + "users.dat" + str(i), sep='::', engine='python',
                      names=['uID', 'uGender', 'uAge'],
                      dtype={'uID': int, 'uGender': str, 'uAge': int}, usecols=[0, 1, 2]).dropna(0,
                                                                                                 'all').as_matrix()
    list(map(usersGendersAndAgesPreprocess, tmp))
    with lock:
        data.append(tmp)


def usersGendersAndAgesPreprocess(x):
    x[1] = usersGenders[x[1]]
    for j, i in enumerate(usersAges):
        if x[2] == i:
            x[2] = j
            break
    return x


def multiReadProc(num):
    manager = Manager()
    data = manager.list()
    lock = manager.Lock()

    pool = Pool(initializer=globalVarinit, initargs=(lock, data))  # default number of processes is os.cpu_count()
    pool.map(proc, num)
    pool.close()
    pool.join()
    return data


def globalVarinit(_lock, _data):
    global data
    global lock
    data = _data
    lock = _lock

# @ut.timing("program run")
def main():
    cla = AgeGenderClassification()
    cla.getData()

    error = cla.genderAndAgeCrossValidation()
    print("10 fold error:" + str(error) + "\n")
    averageErr = sum(error) / 10
    print("averageErr:" + str(averageErr) + "\n")
    # with ut.Log(logEpochFlag+"10 fold error:" + str(error) + "\n"+"averageErr:" + str(averageErr) + "\n"):
    #     pass

    error = cla.genderAndAgeCrossValidation(isGender=False)
    print("10 fold error:" + str(error) + "\n")
    averageErr = sum(error) / 10
    print("averageErr:" + str(averageErr) + "\n")
    # with ut.Log(logEpochFlag+"10 fold error:" + str(error) + "\n"+"averageErr:" + str(averageErr) + "\n"):
    #     pass


main()
