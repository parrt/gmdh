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

iter_max = 10000
pop_size = 100
dimensions = 2
c1 = 2
c2 = 2
err_crit = 0.00001

#initialize the particles
particles = []
for i in range(pop_size):
    p = Network([784,15,10])
    p.fitness_score = 0.0
    p.v = 0.0
    particles.append(p)

# let the first particle be the global best
gbest = particles[0]
err = 999999999
while i < iter_max :
    for p in particles:
        fitness = p.fitness(X, labels)
        # fitness,err = p.fitness(p.params)
        if fitness > p.fitness_score:
            p.fitness_score = fitness
            p.best_biases = p.biases
            p.best_weights = p.weights

        if fitness > gbest.fitness_score:
            gbest = p
        v = p.v + c1 * random() * (p.best_biases - p.best_biases) \
                + c2 * random() * (gbest.best_biases - p.best_biases)
        p.best_biases = p.best_biases + v
        v = p.v + c1 * random() * (p.best_weights - p.best_weights) \
                + c2 * random() * (gbest.best_weights - p.best_weights)
        p.best_weights = p.best_weights + v

    i  += 1
    if err < err_crit:
        break
    #progress bar. '.' = 10%
    if i % (iter_max/10) == 0:
        print '.'