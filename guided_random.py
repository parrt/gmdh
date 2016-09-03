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
N = 200
# N = len(images)
X = images[0:N]
Y = labels[0:N]
# Make one-hot-vectors
# Y = [onehot(lab) for lab in labels[0:N]]
print "N =",N,":",Counter(labels[0:N])
net = Network([784,15,10])
# net.train(X, Y)

# find max fitness by chance
maxfit = -1
globalnet = None

NGENERATIONS = 2000
NPARTICLES = 100

learning_rate = 2

mu = None
sigma = 1
for gen in range(NGENERATIONS):
    maxfit_this_gen = 0
    gennet = None
    MINIBATCH = N
    # do true minibatch where we use all examples not purely random.
    # shuffle then take N/numbatches chunks
    # indexes = np.random.randint(0,len(X),size=MINIBATCH)
    for p in range(NPARTICLES):
        net = Network([784,15,10], mu=mu, sigma=sigma)
        fit = net.fitness(X, Y)
        # fit = net.fitness(X[indexes], Y[indexes])
        # c = net.cost(X,Y)
        # print "cost %3.4f" % c
        if fit>maxfit_this_gen:
            maxfit_this_gen = fit
            gennet = net
    if maxfit_this_gen>maxfit:
        maxfit = maxfit_this_gen
        globalnet = gennet
        delta = gennet.biases - globalnet.biases, gennet.weights - globalnet.weights
        #sigma = np.abs(delta)
        delta = learning_rate * delta
        # print delta
        mu = globalnet.biases + delta[0], globalnet.weights + delta[1] # adding delta seems to help convergence
        # mu = gennet
        sigma = 1 # scale back search area upon a win
    else:
        # if no match, widen search area
        sigma *= 1.01
        print "sigma "+str(sigma)
        pass # do reverse vector of worst in gen
    print "%d: max fitness %d" % (gen, globalnet.fitness(X, Y))

print "max fitness %d/%d" % (maxfit,N)

# print net.cost(X, Y)


