#coding:utf-8

# 反向传播求导联系
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import math


# 构造各个分类数据
def gen_sample():
    data = []
    radius = [0, 50]
    for i in range(1000):  # 生成10k个点
        catg = random.randint(0, 1)  # 随机生成 n 个类别标签
        r = random.random()*10
        arg = random.random()*360
        len = r + radius[catg]    # 如果catg=0 则在r上加 0 ，如果 catg=1，则在r上加 50 赋值给 len
        x_c = math.cos(math.radians(arg))*len
        y_c = math.sin(math.radians(arg))*len
        x = random.random()*30 + x_c
        y = random.random()*30 + y_c
        data.append((x, y, catg))
    return data

def plot_dots(data):
    data_asclass = [[] for i in range(2)]
    for d in data:
        data_asclass[int(d[2])].append((d[0],d[1]))
    colors = ['r.', 'b.', 'r.', 'g.']
    for i, d in enumerate(data_asclass):
        # print(d)
        nd = np.array(d)
        plt.plot(nd[:, 0], nd[:, 1], colors[i])
    plt.draw()



def train(input, output, Whx, Wyh, bh, by):
    """
    完成神经网络的训练过程
    :param input:   输入列向量， 例如 [x,y].T
    :param output:  输出列向量, 例如[0,1,0,0].T
    :param Whx:     x->h 的参数矩阵
    :param Wyh:     h->y 的参数矩阵
    :param bh:      x->h 的偏置向量
    :param by:      h->y 的偏置向量
    :return:
    """
    # forward
    h_z = np.dot(Whx, input) + bh     # 线性求和 hz = Whx * X +bh
    h_a = 1/(1+np.exp(-1*h_z))        # 经过sigmoid激活函数   ha = sigmoid(hz)
    y_z = np.dot(Wyh, h_a) + by       # yz = Wyh * ha + by
    y_a = 1/(1+np.exp(-1*y_z))        # ya = sigmoid(yz)

    # backward
    c_y = (y_a-output)*y_a*(1-y_a)        # ∂C/∂yz = (ya-y)*ya*(1-ya)
    dWyh = np.dot(c_y, h_a.T)             # ∂C/Wyh = ((ya-y)*ya*(1-ya))*ha.T
    dby = c_y                             # ∂C/by = (ya-y)*ya*(1-ya)
    c_h = np.dot(Wyh.T, c_y)*h_a*(1-h_a)  # ∂C/ha = ((ya-y)*ya*(1-ya)) * Wyh
    dWhx = np.dot(c_h, input.T)           # ∂C/Whx = ∂C/ha * x.T
    dbh = c_h                             # ∂C/bh = ∂C/ha
    return dWhx, dWyh, dbh, dby, c_y

def predict(input, Whx, Wyh, bh, by):   # 正向传播，从前到后得到结果
    # print('-----------------')
    # print(input)
    h_z = np.dot(Whx, input) + bh   # (1) hz = Whx * X +bh
    h_a = 1/(1+np.exp(-1*h_z))      # (2) ha = sigmoid(hz)
    y_z = np.dot(Wyh, h_a) + by     # (3) yz = Wyh * ha + by
    y_a = 1/(1+np.exp(-1*y_z))      # (4) ya = sigmoid(yz)
    # print(y_a)
    tag = np.argmax(y_a)   # 选择 y_a 中最大值所对应的下标，返回
    return tag

def test(train_set, test_set, Whx, Wyh, bh, by):
    train_tag = [int(x) for x in train_set[:, 2]]   # 提取训练集中的每个标签放到一个列表中
    test_tag = [int(x) for x in test_set[:, 2]]     # 提取测试集中的每个标签放到一个列表中
    train_pred = []
    test_pred = []
    # 根据 train_set 中每个数据预测得到每一个标签，并添加到 train_pred 列表中
    for i, d in enumerate(train_set):
        input = train_set[i:i+1, 0:2].T         # 每次取出一个数据
        tag = predict(input, Whx, Wyh, bh, by)
        train_pred.append(tag)
    for i, d in enumerate(test_set):
        input = test_set[i:i+1, 0:2].T
        tag = predict(input, Whx, Wyh, bh, by)
        test_pred.append(tag)
    # print(train_tag)
    # print(train_pred)
    train_err = 0   # 训练集中预测错误的个数
    test_err = 0    # 测试集中预测错误的个数
    for i in range(train_pred.__len__()):
        if train_pred[i] != int(train_tag[i]):   # 在测试集中，判断对应的真实标签和预测标签是否相等
            train_err += 1
    for i in range(test_pred.__len__()):
        if test_pred[i] != int(test_tag[i]):
            test_err += 1
    # print(test_tag)
    # print(test_pred)
    train_ratio = train_err / train_pred.__len__()

    test_ratio = test_err / test_pred.__len__()

    return train_err, train_ratio, test_err, test_ratio



if __name__=='__main__':
    input_dim = 2
    output_dim = 2
    hidden_size = 200
    # 初始化 Whx， Wyh， bh ， by
    Whx = np.random.randn(hidden_size, input_dim)*0.01   # Whx = [200, 2]
    print("00000", np.shape(Whx))
    Wyh = np.random.randn(output_dim, hidden_size)*0.01  # Wyh = [2, 200]
    bh = np.zeros((hidden_size, 1))                      # bh = [200, 1]
    by = np.zeros((output_dim, 1))                       # by = [2, 1]
    data = gen_sample()     # 得到分类数据集
    plt.subplot(221)
    plot_dots(data)
    ndata = np.array(data)
    print("1111", ndata[:5])
    train_set = ndata[0:800, :]    # 800 个作为训练集
    test_set = ndata[800:1000, :]  # 200 个作为测试集
    print("22222", np.shape(train_set))
    print("33333", len(test_set))
    train_ratio_list = []
    test_ratio_list = []

    for times in range(10000):
        i = times % train_set.__len__()   # [ 16.22444335,  14.06925116,   0.        ]
        input = train_set[i:i+1, 0:2].T   # 提取数据 [ 16.22444335,  14.06925116]
        ##print("555", input)
        tag = int(train_set[i, 2])        # 提取数据对应的标签 0 整型
        ##print("666", tag)
        output = np.zeros((2, 1))         # 初始化 2 行1列全0 的矩阵

        output[tag, 0] = 1       # 若标签为 0 这output=[[1.],[0.0]], 若标签为 1 这output=[[0.],[1.]],
        dWhx, dWyh, dbh, dby, c_y = train(input, output, Whx, Wyh, bh, by)  # 每迭代一次对参数更新一次

        # 每迭代100次执行一次
        if times % 100 == 0:
            train_err, train_ratio, test_err, test_ratio = test(train_set, test_set, Whx, Wyh, bh, by)
            print('times:{t}\t train ratio:{tar}\t test ratio: {ter}'.format(tar=train_ratio, ter=test_ratio, t=times))
            train_ratio_list.append(train_ratio)
            test_ratio_list.append(test_ratio)

        # 参数权值和偏重更新
        # Whx = Whx - η * dWhx
        # bh = bh - η * dbh
        # Wyh = Wyh - η * dWyh
        # by = by - η * dby
        # 每次迭代都会更新一次参数
        for param, dparam in zip([Whx, Wyh, bh, by], [dWhx, dWyh, dbh, dby]):
            param -= 0.01*dparam


    # 当上面训练迭代结束后，将会得到最后一次更新参数，将更新的参数重新对 ndata 数据集预测得到对应的标签
    for i, d in enumerate(ndata):      # 用原来的数据集作为测试，得到对应的标签
        input = ndata[i:i+1, 0:2].T
        tag = predict(input, Whx, Wyh, bh, by)
        ndata[i, 2] = tag

    plt.subplot(222)
    plot_dots(ndata)
    # plt.figure()
    plt.subplot(212)
    plt.plot(train_ratio_list)
    plt.plot(test_ratio_list)
    plt.show()













