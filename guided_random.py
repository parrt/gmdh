import numpy as np
import gzip, cPickle
from numpy import linalg as LA
from collections import Counter

from network import Network

# Load the dataset
f = gzip.open('/Users/parrt/data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

images = train_set[0]
labels = train_set[1]
img = images[1]

# use just a few images
N = 100
# N = len(images)
X = images[0:N]
# Make one-hot-vectors
# Y = [onehot(lab) for lab in labels[0:N]]
print "N =",N,":",Counter(labels[0:N])
net = Network([784,15,10])
# net.train(X, Y)

# find max fitness by chance
(maxfit,whichnet) = (-1,None)
NGENERATIONS = 200
NPARTICLES = 20

learning_rate = 1

mu = None
sigma = 1
for gen in range(NGENERATIONS):
    (maxfit_this_gen,gennet) = (0,None)
    for p in range(NPARTICLES):
        net = Network([784,15,10], mu=mu, sigma=1)
        fit = net.fitness(X, labels)
        if fit>maxfit_this_gen:
            (maxfit_this_gen,gennet) = (fit,net)
        if fit>maxfit:
            (maxfit,whichnet) = (fit,net)
    delta = gennet.biases-whichnet.biases, gennet.weights-whichnet.weights
    # sigma = np.abs(delta)
    delta = learning_rate * delta
    # print delta
    mu = whichnet.biases+delta[0], whichnet.weights+delta[1] # adding delta seems to help convergence
    print "max fitness this gen %d" % maxfit

print "max fitness %d/%d" % (maxfit,N)

# print net.cost(X, Y)


