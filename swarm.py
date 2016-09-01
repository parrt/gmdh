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
labels = labels[0:N]

# Derived from https://gist.github.com/btbytes/79877
from numpy import array
from random import random
from math import sin, sqrt

ITERATIONS = 1000
SWARM_SIZE = 20
dimensions = len(images[0])

class Particle:
    def __init__(self, net):
        self.net = net
        self.best = net
        self.best_score = 0
    pass

#initialize the particles
particles = [Particle(Network([784,15,10])) for i in range(SWARM_SIZE)]
gbest = particles[0]

mu = None
sigma = None
for it in range(ITERATIONS):
    for p in range(SWARM_SIZE):
        ff = particles[p].best.weights.reshape(1)
        location = np.array(particles[p].best.biases + ff)
        pmu = particles[p].best.biases + gbest.best.biases, \
              particles[p].best.weights + gbest.best.weights
        # pmu = pmu[0] / 2.0, pmu[1] / 2.0
        # psigma = np.norm(particles[p].best.biases - gbest.best.biases  \
        #          particles[p].best.weights - gbest.best.weights
        # net = Network([784, 15, 10], mu=pmu, sigma=sigma)
        # fit = net.fitness(X, labels)
        # if fit > particles[p].best_score:
        #     (maxfit_this_gen, gennet) = (fit, net)
