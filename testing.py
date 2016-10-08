import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(s, 1) for s in sizes[1:]]
        self.weights = [np.random.randn(x, y) 
                        for x, y in zip(sizes[1:], sizes[:-1])]

net = Network([2, 3, 4])
print(net.num_layers)
print(net.biases)
print(net.weights)