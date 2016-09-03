import math
import numpy as np
import gzip, cPickle
from numpy import linalg as LA
from collections import Counter

# Deriving my own simple neural net from
# https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

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
        self.sizes = layer_sizes
        if mu is None or sigma is None:
            self.biases = np.array([np.random.randn(y) for y in layer_sizes[1:]])
            # E.g., weights[0] is the W weight matrix between input and 1st hidden layer
            # biases[0] is the b vector of biases for 1st hidden layer
            # W[j][k] is weight from neuron k to neuron j in prior layer
            # W . a + b is output of all neurons in a layer
            self.weights = np.array([np.random.randn(y, x)
                                     for x, y in zip(layer_sizes[:-1], layer_sizes[1:])])
        else:
            self.biases = np.array([sigma * np.random.randn(y) for y in layer_sizes[1:]])
            self.biases = np.add(self.biases, mu[0])
            # E.g., weights[0] is the W weight matrix between input and 1st hidden layer
            # biases[0] is the b vector of biases for 1st hidden layer
            # W[j][k] is weight from neuron k to neuron j in prior layer
            # W . a + b is output of all neurons in a layer
            self.weights = np.array([sigma * np.random.randn(y, x)
                                     for x, y in zip(layer_sizes[:-1], layer_sizes[1:])])
            self.weights = np.add(self.weights, mu[1])

    def feedforward(self, activations):
        """Return the output of the network if ``a`` is input."""
        for layer in range(len(self.biases)):  # for each layer
            b = self.biases[layer]
            weights = self.weights[layer]
            weighted_outputs = np.dot(weights, activations) + b
            activations = rectified_linear(weighted_outputs)
            # activations = sigmoid(weighted_outputs) # not as good as reLU
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
        return sum/len(X)

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
    return np.exp(a) / np.sum(np.exp(a))

def sigmoid(a):
    return 1.0/(1.0 + np.exp(-a))
