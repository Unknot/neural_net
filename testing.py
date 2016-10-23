#! /bin/usr/python3

import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(s, 1) for s in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[1:], sizes[:-1])]
        self.sizes = sizes

    def eval(self, input_):
        """
        Feed the input to the input neurons and feed it through the network.
        Return the NN output.
        """
        if len(input_) != self.sizes[0]:
            print("Unable to evaluate! Wrong number of inputs!")
            return
        input_ = np.array(input_).reshape(len(input_), 1)
        # print(input_)
        # print(np.dot(self.weights[0], input_) + self.biases[0])
        # print(self.weights[0]@input_)
        for i in range(self.num_layers-1):
            input_ = np.dot(self.weights[i], input_) + self.biases[i]
        return input_

    def feedforward(self, a):
        """
        This function will use the sigmoid function to evaluate the neurons.
        Other than that, it's the same as eval.
        """
        assert len(a) == self.sizes[0]
        a = np.array(a).reshape(len(a), 1)
        for w, b in zip(self.weights[0:], self.biases[0:]):
            a = sigmoid(np.dot(w, a) + b)
        return a

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

net = Network([2, 3, 4])
print(net.num_layers)
print(net.biases)
print(net.weights)
print("eval\n", net.eval([1, 0]))
print("feedforward\n", net.feedforward([1, 0]))
