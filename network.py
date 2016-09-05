import math
import numpy as np
import gzip, cPickle
from numpy import linalg as LA
from collections import Counter

# Deriving my own simple neural net from
# https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

parameter_to_index_map = []  # map parameter index to (i,j) in biases or (i,j,k) in weights
first_weight_index = 0       # parameter index where weights start

def biases_size(topology):
    n = 0
    for layer in range(1,len(topology)):
        n += topology[layer]  # biases
    return n

def weights_size(topology):
    n = 0
    for layer in range(1,len(topology)):
        n += topology[layer] * topology[layer - 1]  # weights
    return n

def init_index_map(topology): # e.g., [784,15,10]
    global parameter_to_index_map, first_weight_index
    layers = len(topology)
    parameter_to_index_map = [0]*(biases_size(topology)+weights_size(topology))
    p = 0
    for layer in range(1,layers):
        for neuron in range(0,topology[layer]):
            parameter_to_index_map[p] = (layer-1,neuron)
            p += 1
    first_weight_index = p
    for layer in range(1, layers):
        for neuron in range(0, topology[layer]):
            for prev_neuron in range(0, topology[layer-1]):
                parameter_to_index_map[p] = (layer - 1, neuron, prev_neuron)
                p += 1
    # print "last index=", p-1
    # print '\n'.join([str(t) for t in parameter_to_index_map])


class Network(object):
    def __init__(self, layer_sizes, mu=None, sigma=None):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(layer_sizes)
        self.topology = layer_sizes
        if mu is None or sigma is None:
            # self.biases = np.array([np.random.randn(y) for y in layer_sizes[1:]])
            # http://cs231n.github.io/neural-networks-2/ says
            # "... common to simply use 0 bias initialization"
            self.biases = np.array([np.zeros(y) for y in layer_sizes[1:]])
            # E.g., weights[0] is the W weight matrix between input and 1st hidden layer
            # biases[0] is the b vector of biases for 1st hidden layer
            # W[j][k] is weight from neuron k to neuron j in prior layer
            # W . a + b is output of all neurons in a layer

            # http://cs231n.github.io/neural-networks-2/ says to "... normalize
            # the variance of each neuron's output to 1 by scaling its weight
            # vector by the square root of its fan-in (i.e. its number of inputs)."
            self.weights = np.array([np.random.randn(y, x)*np.sqrt(2.0/x)
                                     for x, y in zip(layer_sizes[:-1], layer_sizes[1:])])
        else:
            self.biases = np.array([sigma[0][i-1] * np.random.randn(layer_sizes[i])
                                    for i in range(1,self.num_layers)])
            self.biases = np.add(self.biases, mu[0])
            w = []
            for i in range(1, len(layer_sizes)):
                w.append( sigma[1][i-1] * np.random.randn(layer_sizes[i], layer_sizes[i-1]) )
            self.weights = np.array(w)
            # self.weights = np.array([sigma[1] * np.random.randn(y, x)
            #                          for x, y in zip(layer_sizes[:-1], layer_sizes[1:])])
            self.weights = np.add(self.weights, mu[1])\

    def size(self):
        n = 0
        for layer in range(1,len(self.topology)):
            n += self.topology[layer] # biases
            n += self.topology[layer] * self.topology[layer-1] # weights
        return n

    def add_to_parameter(self, i, v):
        if i<first_weight_index: # must be a bias
            ij = parameter_to_index_map[i]
            self.biases[ij[0]][ij[1]] += v
        else:
            ijk = parameter_to_index_map[i]
            self.weights[ijk[0]][ijk[1]][ijk[2]] += v

    def set_parameter(self, i, v):
        if i<first_weight_index: # must be a bias
            ij = parameter_to_index_map[i]
            self.biases[ij[0]][ij[1]] = v
        else:
            ijk = parameter_to_index_map[i]
            self.weights[ijk[0]][ijk[1]][ijk[2]] = v

    def get_parameter(self, i):
        if i<first_weight_index: # must be a bias
            ij = parameter_to_index_map[i]
            return self.biases[ij[0]][ij[1]]
        else:
            ijk = parameter_to_index_map[i]
            return self.weights[ijk[0]][ijk[1]][ijk[2]]

    def feedforward(self, activations):
        """Return the output of the network if ``a`` is input."""
        for layer in range(len(self.biases)):  # for each layer
            b = self.biases[layer]
            weights = self.weights[layer]
            weighted_outputs = np.dot(weights, activations) + b
            # activations = rectified_linear(weighted_outputs)
            activations = sigmoid(weighted_outputs) # not as good as reLU
        return activations

    def train(self, X, Y):
        "Train network on instances in X with predictions in one-hot-vectors of Y"
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            y_ = self.feedforward(x)
            print y_
            print [float("%4.3f" % s) for s in softmax(y_)]
            y_ = np.argmax(softmax(y_))
            print "predict", y_, "label is", y
        return

    def cost(self, X, labels):
        sum = 0.0
        for x,y in zip(X, labels):
            activations = self.feedforward(x)
            y_ = softmax(activations)
            diff = y_ - onehot(y)
            norm = LA.norm(diff)
            sum += norm * norm
        return sum #/len(X)

    def fitness(self,X, labels):
        MINIBATCH = len(X)
        # MINIBATCH = 1000
        # indexes = np.random.randint(0,len(X),size=MINIBATCH)
        # sample = X[indexes]
        # labels = labels[indexes]
        correct = 0
        # for i in indexes:
        for i in range(len(X)):
            x = X[i]
            y = labels[i]
            y_ = self.feedforward(x)
            # y_ = np.argmax(softmax(y_))
            y_ = np.argmax(y_)
            if y_==y: correct += 1
        return correct

def rectified_linear(activations):
    return np.array([max(0,a) for a in activations])

def onehot(i,N=10):
    v = np.zeros(N)
    v[i] = 1
    return v

# def softmax(x):
#     e = [np.exp(a) for a in x]
#     s = np.sum(e)
#     return [a / s for a in x]

def softmax(a):
    # from http://cs231n.github.io/linear-classify/#softmax
    a -= np.max(a) # "shift the values of 'a' so that the highest number is 0"
    return np.exp(a) / np.sum(np.exp(a))

def sigmoid(a):
    return 1.0/(1.0 + np.exp(-a))
