import cPickle, gzip, numpy
import matplotlib.pyplot as plt

# Load the dataset
f = gzip.open('/Users/parrt/data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# display 2nd shape (it's a 0)
img = train_set[0][1]
img = numpy.reshape(img, (28,28))
plt.imshow(img,aspect="auto",cmap='Greys_r')
plt.show()
