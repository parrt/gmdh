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
            self.biases = [np.random.randn(y) for y in layer_sizes[1:]]
            # E.g., weights[0] is the W weight matrix between input and 1st hidden layer
            # biases[0] is the b vector of biases for 1st hidden layer
            # W[j][k] is weight from neuron k to neuron j in prior layer
            # W . a + b is output of all neurons in a layer
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        else:
            self.biases = [sigma*np.random.randn(y) for y in layer_sizes[1:]]
            self.biases = np.add(self.biases, mu[0])
            # E.g., weights[0] is the W weight matrix between input and 1st hidden layer
            # biases[0] is the b vector of biases for 1st hidden layer
            # W[j][k] is weight from neuron k to neuron j in prior layer
            # W . a + b is output of all neurons in a layer
            self.weights = [sigma*np.random.randn(y, x)
                            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
            self.weights = np.add(self.weights,mu[1])

    def feedforward(self, activations):
        """Return the output of the network if ``a`` is input."""
        for layer in range(len(self.biases)):  # for each layer
            b = self.biases[layer]
            weights = self.weights[layer]
            weighted_outputs = np.dot(weights, activations) + b
            activations = rectified_linear(weighted_outputs)
        return activations

    def train(self, X, Y):
        "Train network on instances in X with predictions in one-hot-vectors of Y"
        for i in range(len(X)):
            x = X[i]
            y = labels[i]
            y_ = net.feedforward(x)
            print y_
            print [float("%4.3f" % s) for s in softmax(y_)]
            y_ = np.argmax(softmax(y_))
            print "predict", y_, "label is", y
        return

    def cost(self,X, Y):
        sum = 0.0
        for x in X:
            y_ = self.feedforward(x)
            sum += LA.norm(y_ - Y)**2
        return sum/(2*len(X))

    def fitness(self,X, labels):
        MINIBATCH = len(X)
        # MINIBATCH = 1000
        indexes = np.random.randint(0,len(X),size=MINIBATCH)
        # sample = X[indexes]
        # labels = labels[indexes]
        correct = 0
        # for i in indexes:
        for i in range(len(X)):
            x = X[i]
            y = labels[i]
            y_ = self.feedforward(x)
            y_ = np.argmax(softmax(y_))
            if y_==y: correct += 1
        return correct

def rectified_linear(activations):
    return np.array([max(0,a) for a in activations])

def onehot(i,N=10):
    v = np.zeros(N)
    v[i] = 1
    return v

def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))

def sigmoid(a):
    return 1.0/(1.0 + np.exp(-a))

# Load the dataset
f = gzip.open('/Users/parrt/data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

images = train_set[0]
labels = train_set[1]
img = images[1]

# use just a few images
N = 1000
# N = len(images)
X = images[0:N]
# Make one-hot-vectors
Y = [onehot(lab) for lab in labels[0:N]]
print "N =",N,":",Counter(labels[0:N])
net = Network([784,15,10])
# net.train(X, Y)

# find max fitness by chance
(maxfit,whichnet) = (-1,None)
NGENERATIONS = 200
NPARTICLES = 20

mu = None
for gen in range(NGENERATIONS):
    for p in range(NPARTICLES):
        net = Network([784,15,10], mu=mu, sigma=1)
        fit = net.fitness(X, labels)
        # if fit>maxfit_this_gen:
        #     (maxfit_this_gen,whichnet) = (fit,net)
        if fit>maxfit:
            (maxfit,whichnet) = (fit,net)
    mu = whichnet.biases, whichnet.weights
    print "max fitness this gen %d" % maxfit

print "max fitness %d/%d" % (maxfit,N)

# print net.cost(X, Y)


