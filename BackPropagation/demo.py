#coding:utf-8

import mnist_loader
import network


training_data, validating_data, test_data = mnist_loader.load_data_wrapper()

print("training_data ....")
print(type(training_data))
print(len(training_data))
print(training_data[0][0].shape)


net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


