#coding:utf-8
import csv
import random
import math
import operator

'''
对于每一个测试数据点
	计算所有训练集中的点到测试数据点的距离 dist
	把所有训练数据点和对应的距离添加到列表distances中
	根据距离由大到小降序排列
	计算测试数据点的k个邻近点
		选取前K个距离作为测试数据点的邻近点neighors
	对于每个邻近点
		得到该邻近点的对应的类别标签 response
		统计每个类别标签出现的次数
		把类别标签作为key,次数作为value 放到字典classVotes中
	按照类别次数进行降序排序
	选取最大的次数，即第一个[0][0]，作为测试数据点的预测标签
'''

def loadDataset(filename, split, trainingSet, testSet):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)  # to list

        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])  # change data type float
            if random.random() < split:  # random split dataset to trainingSet and testSet
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    # 4维数据点求距离
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):  # testInstance 每个测试集中的一个数据点
    distances = []
    length = len(testInstance)-1  # last cloumn is label
    # print("lenght::", length)
    for x in range(len(trainingSet)):
        #testinstance
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        # 把每个训练数据点,和到测试集中的一个数据点的距离加到列表distances中
        distances.append((trainingSet[x], dist))
        #distances.append(dist)
    distances.sort(key=operator.itemgetter(1))   # desc

    neighbors = []
    for x in range(k):
        # print("distance:", distances[x])
        neighbors.append(distances[x][0])
    # #print("*******************************")
    # #print("neighbors:", neighbors)
    return neighbors




def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]  # get neighbors label
        # #print("repose::", response)
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    # #print("classVotes::", classVotes)
    # 通过投票的方式，少数服从多少原则，选择
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    # #print("classVotes::", classVotes)
    return sortedVotes[0][0]  # 选择最大的一个类别


def getAccuracy(testSet, predictions):
    correct = 0

    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    #prepare data
    trainingSet = []
    testSet = []
    split = 0.7
    loadDataset('./irisdata.txt', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    #generate predictions
    predictions = []

    k = 3

    for x in range(len(testSet)):
        # trainingsettrainingSet[x]
        neighbors = getNeighbors(trainingSet, testSet[x], k)

        result = getResponse(neighbors)  # 已经判断出测试数据点属于哪个类，用result表示
        # print("result::", result)

        predictions.append(result)
        print ('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()







