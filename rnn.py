# coding=utf-8
from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM



inputfile = '/home/yongcai/fuheyuce/data.csv'  # 销量数据路径


 # 读入数据


def create_interval_dataset(dataset, xback, yback):
    """
    :param dataset: input array of time intervals
    :param look_back: each training set feature length
    :return: convert an array of values into a dataset matrix.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - xback -yback):
        dataX.append(dataset[i:i+xback])
        dataY.append(dataset[i+xback:i+xback+yback])
    return np.asarray(dataX), np.asarray(dataY)


df = pd.read_csv(inputfile)
dataset = np.asarray(df)   # if only 1 column
print(dataset)
dataX, dataY = create_interval_dataset(dataset, xback=6,yback=2)
print (len(dataY))
print (dataY[1])
print ('done!')
def change(data1):
    print data1[1]
    data2=[]
    for i in range(len(data1)):
        tmp = []
        for j in range(1):
            tmp.append(data1[i][j][0])
        data2.append(tmp)

    return data2
datapra=change(dataY)
print  (len(datapra))
print  datapra[2]
dataY=datapra
print ('dXXXXXX!')
print (len(dataY))
print ('dXXXXXX!')
#print (dataY[1])
print ('done!')
#数据归一化
maxtrain1 = np.amax(dataX)
mintrain1 = np.amin(dataX)
dataX = (dataX-mintrain1)/(maxtrain1-mintrain1)
maxtrain2 = np.amax(dataY)
mintrain2 = np.amin(dataY)
dataY = (dataY-mintrain2)/(maxtrain2-mintrain1)
trainin = np.asarray(dataX[0:4799])
testin = np.asarray(dataX[4799:4896])
trainout = dataY[0:4799]
testout = dataY[4799:4896]
print (len(dataX))
print (trainout)
print ('done!')
print (len(dataY))
print ('di0')
print (dataY[-1])
print ('done!')
print (maxtrain1)
print (maxtrain2)
print (len(trainin))
print (len(trainout))
print (len(testin))
print (len(testout))
model = Sequential()
model.add(LSTM(output_dim=3,
               input_shape=(6,1),
                activation='tanh',
                dropout_U=0.6))


model.add(Dense(output_dim=1,
                activation='linear'))
#rmsprop
model.compile(optimizer='rmsprop',
              loss='mape')
model.fit(trainin, trainout , batch_size=60, nb_epoch=10)
print('训练完成')
#score = model.evaluate(trainin, trainout , batch_size=200)

#print('训练精度')
#print (score)
#futureout = model.predict(testin,batch_size=20,verbose=1)
#print(futureout)
#print (testout)

score = model.evaluate(testin , testout , batch_size=10)
print('测试精度')
print (score)
