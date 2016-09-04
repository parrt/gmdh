import numpy as np
import gzip, cPickle
from numpy import linalg as LA
from collections import Counter

from network import Network, init_index_map

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
Y = labels[0:N]
# Make one-hot-vectors
# Y = [onehot(lab) for lab in labels[0:N]]
print "N =",N

# init_index_map([784,15,10])
init_index_map([784,15,10])
net = Network([784,15,10])

print net.get_parameter(20)
net.add_to_parameter(20, 99)
print net.get_parameter(20)

c = net.cost(X, Y)

