import numpy as np
import gzip, cPickle
from numpy import linalg as LA
from collections import Counter
from decimal import Decimal
import random

from network import Network, init_index_map

# Load the dataset
f = gzip.open('/Users/parrt/data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

images = train_set[0]
labels = train_set[1]
img = images[1]

# use just a few images
N = 300
# N = len(images)
X = images[0:N]
Y = labels[0:N]
# Make one-hot-vectors
# Y = [onehot(lab) for lab in labels[0:N]]
print "N =",N

# init_index_map([784,15,10])
init_index_map([784,15,10])
pos = Network([784,15,10])

num_parameters = pos.size()
print num_parameters

# print net.get_parameter(20)
# net.add_to_parameter(20, 99)
# print net.get_parameter(20)

precision = 0.000000000001
eta = 30
steps = 0
h = 0.0001
cost = 1e20

while True:
    steps += 1
    prevcost = cost
    # what is cost at current location?
    # compute finite difference for one parameter
    # (f(pos+h) - f(pos-h)) / 2h
    dir = random.randint(0,num_parameters-1) # randint() is inclusive on both ends
    pos.add_to_parameter(dir, h)
    right = pos.cost(X,Y)
    pos.add_to_parameter(dir, -2*h)
    left = pos.cost(X,Y)
    pos.add_to_parameter(dir, h)     # reset
    finite_diff = (right - left) / (2*h)
    # move position in one direction only
    pos.add_to_parameter(dir, -eta * finite_diff) # decelerates x jump as it flattens out
    # delta = Decimal(cost) - Decimal(prevcost)

    cost = pos.cost(X, Y) # what is new cost
    print "%d: cost = %3.5f, correct %d, weight norm neuron 0,0: %3.3f" %\
          (steps,cost,pos.fitness(X,Y),LA.norm(pos.weights[0][0]))
    if cost > prevcost:
        lossratio = (cost - prevcost) / prevcost
        print "lossratio by %3.5f" % lossratio
        if lossratio > 0.0035: # even sigmoid seems to get these weird pop ups in energy so don't let it
            pos.add_to_parameter(dir, eta * finite_diff)  # restore and try again
            print "resetting"

    # stop when small change in vertical but not heading down
    # Sometimes subtraction wipes out precision and we get an actual 0.0
    # if delta >= 0 and abs(delta) < precision:
    #     break
