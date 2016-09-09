import numpy as np
import gzip, cPickle
from numpy import linalg as LA
from collections import Counter
from decimal import Decimal
import random

from network2 import Network2, init_index_map

# Load the dataset
f = gzip.open('/Users/parrt/data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

images = train_set[0]
labels = train_set[1]
img = images[1]

# use just a few images
N = 30
# N = len(images)
X = images[0:N]
Y = labels[0:N]
# Make one-hot-vectors
# Y = [onehot(lab) for lab in labels[0:N]]
print "N =",N

# init_index_map([784,15,10])
init_index_map([784,15,10])
pos = Network2([784,15,10])

num_parameters = pos.size()
print "num parameters =", num_parameters

precision = 0.000000000001
eta = 40
steps = 0
h = 0.00001
cost = 1e20
NPARTIALS = 1
MINIBATCH = 30
print "NPARTIALS =", NPARTIALS
print "MINIBATCH =", MINIBATCH
print "eta =", eta

def compute_finite_diff(pos, d):
    save = pos.get_parameter(d)
    pos.add_to_parameter(d, h)
    right = pos.cost(samples, sample_labels)
    pos.set_parameter(d, save)
    pos.add_to_parameter(d, -h)
    left = pos.cost(samples, sample_labels)
    pos.set_parameter(d, save)  # restore position vector
    return (right - left) / (2 * h)

while True:
    steps += 1
    prevcost = cost

    # what is cost at current location?
    # indexes = np.random.randint(0,len(X),size=MINIBATCH)
    # samples = X[indexes]
    # sample_labels = labels[indexes]
    samples = X
    sample_labels = labels

    # compute finite difference for one parameter
    # (f(pos+h) - f(pos-h)) / 2h
    save = [0]*NPARTIALS
    d = [0]*NPARTIALS
    for i in range(NPARTIALS):
        d[i] = random.randint(0,num_parameters-1) # randint() is inclusive on both ends
        save[i] = pos.get_parameter(d[i])
        finite_diff = compute_finite_diff(pos,d[i])
        # move position in one direction
        pos.add_to_parameter(d[i], -eta * finite_diff)

    # delta = Decimal(cost) - Decimal(prevcost)

    cost = pos.cost(samples, sample_labels) # what is new cost
    if steps % 1000 == 0:
        correct = pos.fitness(X,Y)
        print "%d: cost = %3.5f, correct %d, weight norm neuron 0,0: %3.3f" % \
              (steps, cost, correct, LA.norm(pos.weights[0][0]))
    # print "%d: cost = %3.5f, weight norm neuron 0,0: %3.3f" %\
    #       (steps,cost,LA.norm(pos.weights[0][0]))
    if cost > prevcost:
        lossratio = (cost - prevcost) / prevcost
        if lossratio > 0.035: # even sigmoid seems to get these weird pop ups in energy so don't let it
            # print "lossratio by %3.5f" % lossratio
            for i in range(NPARTIALS):
                pos.set_parameter(d[i], save[i]) # restore so we can try again
            # print "resetting to cost %3.5f from pop up %3.5f" % (prevcost,cost)
            cost = prevcost # reset cost too lest it think it hadn't jumped much next iteration


    # stop when small change in vertical but not heading down
    # Sometimes subtraction wipes out precision and we get an actual 0.0
    # if delta >= 0 and abs(delta) < precision:
    #     break
