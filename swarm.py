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
SWARM_SIZE = 10
dimensions = len(images[0])

class Particle:
    def __init__(self, net):
        self.pos = net
        self.best = net
        self.best_score = 0
    pass

#initialize the particles
particles = [Particle(Network([784,30,10])) for i in range(SWARM_SIZE)]

mu = particles[0].pos
sigma = None
for it in range(ITERATIONS):
    # update global best with best of all particles
    gbest = particles[0].best
    gbest_score = particles[0].best_score
    for i in range(SWARM_SIZE):
        p = particles[i]
        if p.best_score > gbest_score:
            gbest = p.best
            gbest_score = p.best_score
    print "global best score " + str(gbest_score)

    for i in range(SWARM_SIZE):
        p = particles[i]
        pmu = p.best.biases + gbest.biases, \
              p.best.weights + gbest.weights
        pmu = pmu[0] / 2.0, pmu[1] / 2.0
        psigma = np.abs(p.best.biases - gbest.biases), \
                 np.abs(p.best.weights - gbest.weights)
        pos = Network([784,30,10], mu=pmu, sigma=psigma)
        p.pos = pos
        fit = pos.fitness(X, labels)
        if fit > p.best_score:
            p.best = pos
            p.best_score = fit
